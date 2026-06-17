# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from typing import Any, Tuple

import torch
from torch import nn
from torch.distributions import Beta
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.modules.dit import AlternateVLDiT, DiT, SelfAttentionTransformer
from gr00t.model.modules.embodiment_conditioned_mlp import (
    CategorySpecificMLP,
    MultiEmbodimentActionEncoder,
)


logger = logging.getLogger(__name__)


class Gr00tN1d7ActionHead(nn.Module):
    """Action head component for flow matching diffusion policy."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7Config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.input_embedding_dim = config.input_embedding_dim

        if config.use_alternate_vl_dit:
            self.model = AlternateVLDiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
                attend_text_every_n_blocks=config.attend_text_every_n_blocks,
            )
            logger.info("Using AlternateVLDiT for diffusion model")
        else:
            self.model = DiT(
                **config.diffusion_model_cfg,
                cross_attention_dim=config.backbone_embedding_dim,
            )
            logger.info("Using DiT for diffusion model")
        self.action_dim = config.max_action_dim
        self.action_horizon = config.action_horizon
        self.num_inference_timesteps = config.num_inference_timesteps

        self.state_encoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=config.max_state_dim * config.state_history_length,
            hidden_dim=self.hidden_size,
            output_dim=self.input_embedding_dim,
        )
        self.action_encoder = MultiEmbodimentActionEncoder(
            action_dim=self.action_dim,
            hidden_size=self.input_embedding_dim,
            num_embodiments=config.max_num_embodiments,
        )
        self.action_decoder = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=self.hidden_size,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )
        self.enable_progress_head = config.enable_progress_head
        self.progress_loss_weight = config.progress_loss_weight
        self.progress_head_source = getattr(config, "progress_head_source", "action").lower()
        self.progress_output_type = getattr(config, "progress_output_type", "scalar").lower()
        if self.progress_output_type not in {"scalar", "soft_bins", "hard_bins"}:
            raise ValueError(
                f"Unsupported progress_output_type={self.progress_output_type!r}; "
                "expected 'scalar', 'soft_bins', or 'hard_bins'."
            )
        self.progress_num_bins = int(getattr(config, "progress_num_bins", 10))
        if self.progress_num_bins < 2:
            raise ValueError("progress_num_bins must be at least 2.")
        self.progress_soft_label_sigma = float(getattr(config, "progress_soft_label_sigma", 0.08))
        if self.progress_soft_label_sigma <= 0:
            raise ValueError("progress_soft_label_sigma must be positive.")
        if self.progress_head_source not in {
            "action",
            "vlm",
            "vlm_dit",
            "vlm_pooled_dit",
            "state_multilayer_dit",
            "vlm_pooled",
            "vlm_pooled_state",
            "vlm_concat_linear",
            "vlm_concat_projected_linear",
            "vlm_concat_attention_pool",
            "vlm_layer_pooled",
            "vlm_layer_concat_linear",
            "vlm_layer_concat_projected_linear",
            "vlm_layer_concat_attention_pool",
        }:
            raise ValueError(
                f"Unsupported progress_head_source={self.progress_head_source!r}; "
                "expected 'action', 'vlm', 'vlm_dit', 'vlm_pooled_dit', "
                "'state_multilayer_dit', 'vlm_pooled', 'vlm_pooled_state', "
                "'vlm_concat_linear', 'vlm_concat_projected_linear', "
                "'vlm_concat_attention_pool', 'vlm_layer_pooled', "
                "'vlm_layer_concat_linear', 'vlm_layer_concat_projected_linear', "
                "or 'vlm_layer_concat_attention_pool'."
            )
        self.progress_vlm_layer = int(getattr(config, "progress_vlm_layer", -1))
        self.progress_concat_project_dim = int(getattr(config, "progress_concat_project_dim", 64))
        if self.progress_concat_project_dim <= 0:
            raise ValueError("progress_concat_project_dim must be positive.")
        self.isolate_progress_action_attention = getattr(
            config, "isolate_progress_action_attention", False
        )
        if self.enable_progress_head:
            progress_output_dim = (
                self.progress_num_bins
                if self.progress_output_type in {"soft_bins", "hard_bins"}
                else 1
            )
            if self.progress_output_type in {"soft_bins", "hard_bins"}:
                self.register_buffer(
                    "progress_bin_centers",
                    (torch.arange(self.progress_num_bins, dtype=torch.float32) + 0.5)
                    / self.progress_num_bins,
                    persistent=False,
                )
            use_progress_token = self.progress_head_source in {
                "action",
                "vlm",
                "vlm_dit",
                "vlm_pooled_dit",
                "state_multilayer_dit",
            }
            progress_token_dim = (
                config.backbone_embedding_dim
                if self.progress_head_source in {"vlm", "vlm_dit"}
                else self.input_embedding_dim
            )
            if self.progress_head_source in {"vlm", "vlm_pooled", "vlm_layer_pooled"}:
                progress_head_dim = config.backbone_embedding_dim
            elif self.progress_head_source == "vlm_pooled_state":
                progress_head_dim = config.backbone_embedding_dim + self.input_embedding_dim
            elif self.progress_head_source in {"vlm_concat_linear", "vlm_layer_concat_linear"}:
                progress_head_dim = config.max_seq_len * config.backbone_embedding_dim
            elif self.progress_head_source in {
                "vlm_concat_projected_linear",
                "vlm_layer_concat_projected_linear",
            }:
                progress_head_dim = config.max_seq_len * self.progress_concat_project_dim
            elif self.progress_head_source in {
                "vlm_concat_attention_pool",
                "vlm_layer_concat_attention_pool",
            }:
                progress_head_dim = self.progress_concat_project_dim
            elif self.progress_head_source in {"vlm_pooled_dit", "state_multilayer_dit"}:
                progress_head_dim = self.hidden_size
            else:
                progress_head_dim = self.hidden_size
            if use_progress_token:
                self.progress_token = nn.Parameter(torch.empty(1, 1, progress_token_dim))
                nn.init.normal_(self.progress_token, mean=0.0, std=1.0)
                self.progress_token_scale = 0.02
            if self.progress_head_source == "vlm_dit":
                self.progress_vlm_projector = nn.Linear(
                    config.backbone_embedding_dim,
                    self.input_embedding_dim,
                )
                nn.init.xavier_uniform_(self.progress_vlm_projector.weight)
                nn.init.zeros_(self.progress_vlm_projector.bias)
            if self.progress_head_source in {
                "vlm_concat_projected_linear",
                "vlm_layer_concat_projected_linear",
                "vlm_concat_attention_pool",
                "vlm_layer_concat_attention_pool",
            }:
                self.progress_vlm_token_norm = nn.LayerNorm(config.backbone_embedding_dim)
                self.progress_vlm_token_projector = nn.Linear(
                    config.backbone_embedding_dim,
                    self.progress_concat_project_dim,
                )
                nn.init.xavier_uniform_(self.progress_vlm_token_projector.weight)
                nn.init.zeros_(self.progress_vlm_token_projector.bias)
            if self.progress_head_source in {
                "vlm_concat_attention_pool",
                "vlm_layer_concat_attention_pool",
            }:
                self.progress_vlm_token_attention = nn.Linear(
                    self.progress_concat_project_dim,
                    1,
                )
                nn.init.zeros_(self.progress_vlm_token_attention.weight)
                nn.init.zeros_(self.progress_vlm_token_attention.bias)
            if self.progress_head_source in {
                "state_multilayer_dit",
                "vlm_concat_linear",
                "vlm_concat_projected_linear",
                "vlm_concat_attention_pool",
                "vlm_layer_concat_linear",
                "vlm_layer_concat_projected_linear",
                "vlm_layer_concat_attention_pool",
            }:
                self.progress_head = nn.Sequential(
                    nn.LayerNorm(progress_head_dim),
                    nn.Linear(progress_head_dim, progress_output_dim),
                )
                nn.init.zeros_(self.progress_head[1].weight)
                nn.init.zeros_(self.progress_head[1].bias)
            else:
                self.progress_head = nn.Sequential(
                    nn.LayerNorm(progress_head_dim),
                    nn.Linear(progress_head_dim, progress_head_dim),
                    nn.GELU(),
                    nn.Linear(progress_head_dim, progress_output_dim),
                )
                nn.init.xavier_uniform_(self.progress_head[1].weight)
                nn.init.zeros_(self.progress_head[1].bias)
                nn.init.zeros_(self.progress_head[3].weight)
                nn.init.zeros_(self.progress_head[3].bias)
            self._register_progress_gradient_sanitizers()

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )

        vl_self_attention_cfg = getattr(config, "vl_self_attention_cfg", None)
        if vl_self_attention_cfg and vl_self_attention_cfg.get("num_layers", 0) > 0:
            self.vl_self_attention = SelfAttentionTransformer(**vl_self_attention_cfg)
        else:
            self.vl_self_attention = nn.Identity()

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        # State dropout parameters
        self.state_dropout_prob = config.state_dropout_prob

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets
        self.set_trainable_parameters(
            config.tune_projector,
            config.tune_diffusion_model,
            config.tune_vlln,
            config.tune_progress_head,
        )

    def set_trainable_parameters(
        self,
        tune_projector: bool,
        tune_diffusion_model: bool,
        tune_vlln: bool,
        tune_progress_head: bool = True,
    ):
        self.tune_projector = tune_projector
        self.tune_diffusion_model = tune_diffusion_model
        self.tune_vlln = tune_vlln
        self.tune_progress_head = tune_progress_head
        self.optimize_action_loss = tune_projector or tune_diffusion_model or tune_vlln
        for p in self.parameters():
            p.requires_grad = True
        if not tune_projector:
            self.state_encoder.requires_grad_(False)
            self.action_encoder.requires_grad_(False)
            self.action_decoder.requires_grad_(False)
            if self.config.add_pos_embed:
                self.position_embedding.requires_grad_(False)
        if not tune_diffusion_model:
            self.model.requires_grad_(False)
        if not tune_vlln:
            self.vlln.requires_grad_(False)
            self.vl_self_attention.requires_grad_(False)
        if self.enable_progress_head and not tune_progress_head:
            if hasattr(self, "progress_token"):
                self.progress_token.requires_grad_(False)
            self.progress_head.requires_grad_(False)
            if hasattr(self, "progress_vlm_projector"):
                self.progress_vlm_projector.requires_grad_(False)
            if hasattr(self, "progress_vlm_token_norm"):
                self.progress_vlm_token_norm.requires_grad_(False)
            if hasattr(self, "progress_vlm_token_projector"):
                self.progress_vlm_token_projector.requires_grad_(False)
            if hasattr(self, "progress_vlm_token_attention"):
                self.progress_vlm_token_attention.requires_grad_(False)
        logger.debug(f"Tune action head projector: {self.tune_projector}")
        logger.debug(f"Tune action head diffusion model: {self.tune_diffusion_model}")
        logger.debug(f"Tune action head vlln: {self.tune_vlln}")
        logger.debug(f"Tune progress head: {self.tune_progress_head}")
        # Check if any parameters are still trainable. If not, log a warning.
        if (
            not tune_projector
            and not tune_diffusion_model
            and not tune_vlln
            and (not self.enable_progress_head or not tune_progress_head)
        ):
            for name, p in self.named_parameters():
                if p.requires_grad:
                    logger.debug(f"Action head trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No action head trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if not self.tune_projector:
                self.state_encoder.eval()
                self.action_encoder.eval()
                self.action_decoder.eval()
                if self.config.add_pos_embed:
                    self.position_embedding.eval()
            if not self.tune_diffusion_model:
                self.model.eval()
            if not self.tune_vlln:
                self.vlln.eval()
                self.vl_self_attention.eval()
        if self.enable_progress_head and not self.tune_progress_head:
            self.progress_head.eval()
            if hasattr(self, "progress_vlm_projector"):
                self.progress_vlm_projector.eval()
            if hasattr(self, "progress_vlm_token_norm"):
                self.progress_vlm_token_norm.eval()
            if hasattr(self, "progress_vlm_token_projector"):
                self.progress_vlm_token_projector.eval()
            if hasattr(self, "progress_vlm_token_attention"):
                self.progress_vlm_token_attention.eval()

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype)
        sample = (1 - sample) * self.config.noise_s
        return sample

    def process_backbone_output(self, backbone_output: BatchFeature) -> BatchFeature:
        backbone_features = backbone_output["backbone_features"]
        backbone_features = self.vlln(backbone_features)
        backbone_features = self.vl_self_attention(backbone_features)
        backbone_output["backbone_features"] = backbone_features
        return backbone_output

    def _uses_vlm_progress_head(self) -> bool:
        return self.enable_progress_head and self.progress_head_source == "vlm"

    def _uses_vlm_dit_progress_head(self) -> bool:
        return self.enable_progress_head and self.progress_head_source == "vlm_dit"

    def _uses_vlm_pooled_dit_progress_head(self) -> bool:
        return self.enable_progress_head and self.progress_head_source == "vlm_pooled_dit"

    def _uses_state_multilayer_dit_progress_head(self) -> bool:
        return self.enable_progress_head and self.progress_head_source == "state_multilayer_dit"

    def _uses_vlm_pooled_progress_head(self) -> bool:
        return self.enable_progress_head and self.progress_head_source in {
            "vlm_pooled",
            "vlm_pooled_state",
            "vlm_concat_linear",
            "vlm_concat_projected_linear",
            "vlm_concat_attention_pool",
        }

    def _uses_vlm_layer_pooled_progress_head(self) -> bool:
        return self.enable_progress_head and self.progress_head_source in {
            "vlm_layer_pooled",
            "vlm_layer_concat_linear",
            "vlm_layer_concat_projected_linear",
            "vlm_layer_concat_attention_pool",
        }

    def _uses_non_action_progress_head(self) -> bool:
        return (
            self._uses_vlm_progress_head()
            or self._uses_vlm_dit_progress_head()
            or self._uses_vlm_pooled_dit_progress_head()
            or self._uses_state_multilayer_dit_progress_head()
            or self._uses_vlm_pooled_progress_head()
            or self._uses_vlm_layer_pooled_progress_head()
        )

    def uses_backbone_progress_token(self) -> bool:
        return self._uses_vlm_dit_progress_head()

    def uses_vlm_layer_hidden_states(self) -> bool:
        return self._uses_vlm_layer_pooled_progress_head()

    def make_backbone_progress_token(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        return self._make_progress_features(
            batch_size=batch_size,
            position_index=0,
            device=device,
            dtype=self.progress_token.dtype,
        )

    def _compute_vlm_progress_hidden(self, backbone_output: BatchFeature) -> torch.Tensor:
        backbone_features = backbone_output["backbone_features"]
        progress_features = self._make_progress_features(
            batch_size=backbone_features.shape[0],
            position_index=backbone_features.shape[1],
            device=backbone_features.device,
            dtype=backbone_features.dtype,
        )
        vlm_progress_features = torch.cat((backbone_features, progress_features), dim=1)
        vlm_progress_features = self.vlln(vlm_progress_features)
        vlm_progress_features = self.vl_self_attention(vlm_progress_features)
        return vlm_progress_features[:, -1]

    def _compute_vlm_dit_progress_hidden(
        self,
        state_features: torch.Tensor,
        progress_backbone_output: BatchFeature,
    ) -> torch.Tensor:
        progress_backbone_output = self.process_backbone_output(progress_backbone_output)
        progress_vl_embeds = progress_backbone_output.backbone_features
        progress_token_index = progress_backbone_output.progress_token_index.to(
            device=progress_vl_embeds.device,
            dtype=torch.long,
        )
        batch_index = torch.arange(progress_vl_embeds.shape[0], device=progress_vl_embeds.device)
        progress_vlm_hidden = progress_vl_embeds[batch_index, progress_token_index]
        progress_query = self.progress_vlm_projector(progress_vlm_hidden).unsqueeze(1)
        progress_sa_embs = torch.cat((state_features, progress_query), dim=1)
        progress_timestep = torch.zeros(
            progress_vl_embeds.shape[0],
            dtype=torch.long,
            device=progress_vl_embeds.device,
        )
        progress_output = self._run_model(
            hidden_states=progress_sa_embs,
            vl_embeds=progress_vl_embeds,
            timestep=progress_timestep,
            backbone_output=progress_backbone_output,
        )
        return progress_output[:, state_features.shape[1]]

    def _compute_vlm_pooled_dit_progress_hidden(
        self,
        state_features: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        progress_features = self._make_progress_features(
            batch_size=state_features.shape[0],
            position_index=state_features.shape[1],
            device=state_features.device,
            dtype=state_features.dtype,
        )
        progress_sa_embs = torch.cat((state_features, progress_features), dim=1)
        progress_timestep = torch.zeros(
            state_features.shape[0],
            dtype=torch.long,
            device=state_features.device,
        )
        progress_output = self._run_model(
            hidden_states=progress_sa_embs,
            vl_embeds=backbone_output.backbone_features,
            timestep=progress_timestep,
            backbone_output=backbone_output,
        )
        progress_index = state_features.shape[1]
        return progress_output[:, progress_index]

    def _compute_state_multilayer_dit_progress_hidden(
        self,
        state_features: torch.Tensor,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        progress_features = self._make_progress_features(
            batch_size=state_features.shape[0],
            position_index=state_features.shape[1],
            device=state_features.device,
            dtype=state_features.dtype,
        )
        progress_sa_embs = torch.cat((state_features, progress_features), dim=1)
        progress_timestep = torch.zeros(
            state_features.shape[0],
            dtype=torch.long,
            device=state_features.device,
        )
        progress_output = self._run_model(
            hidden_states=progress_sa_embs,
            vl_embeds=backbone_output.backbone_features,
            timestep=progress_timestep,
            backbone_output=backbone_output,
        )
        progress_index = state_features.shape[1]
        return progress_output[:, progress_index]

    def _pool_vlm_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        backbone_features = backbone_output.backbone_features
        mask = backbone_output.backbone_attention_mask.to(
            device=backbone_features.device,
            dtype=backbone_features.dtype,
        )
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (backbone_features * mask).sum(dim=1) / denom

    def _pool_vlm_layer_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        hidden_states = backbone_output.get("backbone_hidden_states")
        if hidden_states is None:
            raise ValueError(
                "backbone_hidden_states is required for progress_head_source='vlm_layer_pooled'"
            )
        layer_index = self.progress_vlm_layer
        if layer_index < 0:
            layer_index = len(hidden_states) - 1
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise ValueError(
                f"progress_vlm_layer={self.progress_vlm_layer} is out of range for "
                f"{len(hidden_states)} hidden-state tensors."
            )
        features = hidden_states[layer_index]
        mask = backbone_output.backbone_attention_mask.to(
            device=features.device,
            dtype=features.dtype,
        )
        mask = mask.unsqueeze(-1)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (features * mask).sum(dim=1) / denom

    def _concat_vlm_layer_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        hidden_states = backbone_output.get("backbone_hidden_states")
        if hidden_states is None:
            raise ValueError(
                "backbone_hidden_states is required for "
                "progress_head_source='vlm_layer_concat_linear'"
            )
        layer_index = self.progress_vlm_layer
        if layer_index < 0:
            layer_index = len(hidden_states) - 1
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise ValueError(
                f"progress_vlm_layer={self.progress_vlm_layer} is out of range for "
                f"{len(hidden_states)} hidden-state tensors."
            )
        features = hidden_states[layer_index]
        mask = backbone_output.backbone_attention_mask.to(
            device=features.device,
            dtype=features.dtype,
        )
        max_tokens = self.config.max_seq_len
        if features.shape[1] > max_tokens:
            features = features[:, :max_tokens]
            mask = mask[:, :max_tokens]
        features = features * mask.unsqueeze(-1)
        if features.shape[1] < max_tokens:
            pad_tokens = max_tokens - features.shape[1]
            features = F.pad(features, (0, 0, 0, pad_tokens))
        return features.reshape(features.shape[0], -1)

    def _get_selected_vlm_layer_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        hidden_states = backbone_output.get("backbone_hidden_states")
        if hidden_states is None:
            raise ValueError(
                "backbone_hidden_states is required for "
                f"progress_head_source={self.progress_head_source!r}"
            )
        layer_index = self.progress_vlm_layer
        if layer_index < 0:
            layer_index = len(hidden_states) - 1
        if layer_index < 0 or layer_index >= len(hidden_states):
            raise ValueError(
                f"progress_vlm_layer={self.progress_vlm_layer} is out of range for "
                f"{len(hidden_states)} hidden-state tensors."
            )
        return hidden_states[layer_index]

    def _project_and_mask_vlm_tokens(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mask = mask.to(device=features.device, dtype=features.dtype)
        max_tokens = self.config.max_seq_len
        if features.shape[1] > max_tokens:
            features = features[:, :max_tokens]
            mask = mask[:, :max_tokens]
        features = self.progress_vlm_token_norm(features)
        features = self.progress_vlm_token_projector(features)
        features = F.gelu(features)
        features = features * mask.unsqueeze(-1)
        return features, mask

    def _concat_projected_vlm_layer_features(
        self,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        features = self._get_selected_vlm_layer_features(backbone_output)
        mask = backbone_output.backbone_attention_mask.to(
            device=features.device,
            dtype=features.dtype,
        )
        features, mask = self._project_and_mask_vlm_tokens(features, mask)
        max_tokens = self.config.max_seq_len
        if features.shape[1] < max_tokens:
            pad_tokens = max_tokens - features.shape[1]
            features = F.pad(features, (0, 0, 0, pad_tokens))
        return features.reshape(features.shape[0], -1)

    def _concat_vlm_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        backbone_features = backbone_output.backbone_features
        mask = backbone_output.backbone_attention_mask.to(
            device=backbone_features.device,
            dtype=backbone_features.dtype,
        )
        max_tokens = self.config.max_seq_len
        if backbone_features.shape[1] > max_tokens:
            backbone_features = backbone_features[:, :max_tokens]
            mask = mask[:, :max_tokens]
        backbone_features = backbone_features * mask.unsqueeze(-1)
        if backbone_features.shape[1] < max_tokens:
            pad_tokens = max_tokens - backbone_features.shape[1]
            backbone_features = F.pad(backbone_features, (0, 0, 0, pad_tokens))
        return backbone_features.reshape(backbone_features.shape[0], -1)

    def _concat_projected_vlm_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        backbone_features = backbone_output.backbone_features
        mask = backbone_output.backbone_attention_mask.to(
            device=backbone_features.device,
            dtype=backbone_features.dtype,
        )
        backbone_features, mask = self._project_and_mask_vlm_tokens(backbone_features, mask)
        max_tokens = self.config.max_seq_len
        if backbone_features.shape[1] < max_tokens:
            pad_tokens = max_tokens - backbone_features.shape[1]
            backbone_features = F.pad(backbone_features, (0, 0, 0, pad_tokens))
        return backbone_features.reshape(backbone_features.shape[0], -1)

    def _attention_pool_projected_vlm_tokens(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        features, mask = self._project_and_mask_vlm_tokens(features, mask)
        scores = self.progress_vlm_token_attention(features).squeeze(-1)
        scores = scores.masked_fill(mask <= 0, -1e4)
        weights = torch.softmax(scores.float(), dim=1).to(dtype=features.dtype)
        weights = weights * mask
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (features * weights.unsqueeze(-1)).sum(dim=1)

    def _attention_pool_vlm_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        return self._attention_pool_projected_vlm_tokens(
            features=backbone_output.backbone_features,
            mask=backbone_output.backbone_attention_mask,
        )

    def _attention_pool_vlm_layer_features(self, backbone_output: BatchFeature) -> torch.Tensor:
        return self._attention_pool_projected_vlm_tokens(
            features=self._get_selected_vlm_layer_features(backbone_output),
            mask=backbone_output.backbone_attention_mask,
        )

    def _compute_vlm_pooled_progress_hidden(
        self,
        backbone_output: BatchFeature,
        state_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.progress_head_source == "vlm_concat_linear":
            return self._concat_vlm_features(backbone_output)
        if self.progress_head_source == "vlm_concat_projected_linear":
            return self._concat_projected_vlm_features(backbone_output)
        if self.progress_head_source == "vlm_concat_attention_pool":
            return self._attention_pool_vlm_features(backbone_output)
        pooled_features = self._pool_vlm_features(backbone_output)
        if self.progress_head_source == "vlm_pooled_state":
            return torch.cat((pooled_features, state_features.squeeze(1)), dim=-1)
        return pooled_features

    def _compute_vlm_layer_pooled_progress_hidden(
        self,
        backbone_output: BatchFeature,
    ) -> torch.Tensor:
        if self.progress_head_source == "vlm_layer_concat_linear":
            return self._concat_vlm_layer_features(backbone_output)
        if self.progress_head_source == "vlm_layer_concat_projected_linear":
            return self._concat_projected_vlm_layer_features(backbone_output)
        if self.progress_head_source == "vlm_layer_concat_attention_pool":
            return self._attention_pool_vlm_layer_features(backbone_output)
        return self._pool_vlm_layer_features(backbone_output)

    def _make_progress_features(
        self,
        batch_size: int,
        position_index: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        bounded_token = torch.tanh(self.progress_token.float()) * self.progress_token_scale
        progress_features = bounded_token.to(device=device, dtype=dtype).expand(batch_size, -1, -1)
        if self.config.add_pos_embed and self.progress_head_source == "action":
            pos_id = torch.tensor([position_index], dtype=torch.long, device=device)
            progress_features = progress_features + self.position_embedding(pos_id).unsqueeze(0)
        return progress_features

    def _register_progress_gradient_sanitizers(self) -> None:
        def _sanitize_grad(grad: torch.Tensor) -> torch.Tensor:
            return torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

        if hasattr(self, "progress_token"):
            self.progress_token.register_hook(_sanitize_grad)
        for parameter in self.progress_head.parameters():
            parameter.register_hook(_sanitize_grad)
        if hasattr(self, "progress_vlm_projector"):
            for parameter in self.progress_vlm_projector.parameters():
                parameter.register_hook(_sanitize_grad)
        if hasattr(self, "progress_vlm_token_norm"):
            for parameter in self.progress_vlm_token_norm.parameters():
                parameter.register_hook(_sanitize_grad)
        if hasattr(self, "progress_vlm_token_projector"):
            for parameter in self.progress_vlm_token_projector.parameters():
                parameter.register_hook(_sanitize_grad)
        if hasattr(self, "progress_vlm_token_attention"):
            for parameter in self.progress_vlm_token_attention.parameters():
                parameter.register_hook(_sanitize_grad)

    def _make_progress_route_features(
        self,
        state_features: torch.Tensor,
        action_features: torch.Tensor,
        action_position_features: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, int]:
        progress_features = self._make_progress_features(
            batch_size=batch_size,
            position_index=action_features.shape[1],
            device=device,
            dtype=action_features.dtype,
        )
        if action_position_features is None:
            action_placeholders = torch.zeros_like(action_features)
        else:
            action_placeholders = action_position_features.to(
                device=device,
                dtype=action_features.dtype,
            )
        progress_index = state_features.shape[1] + action_placeholders.shape[1]
        return torch.cat(
            (state_features, action_placeholders, progress_features), dim=1
        ), progress_index

    def _run_model(
        self,
        hidden_states: torch.Tensor,
        vl_embeds: torch.Tensor,
        timestep: torch.Tensor,
        backbone_output: BatchFeature,
        return_all_hidden_states: bool = False,
    ):
        if self.config.use_alternate_vl_dit:
            return self.model(
                hidden_states=hidden_states,
                encoder_hidden_states=vl_embeds,
                attention_mask=None,
                encoder_attention_mask=backbone_output.backbone_attention_mask,
                timestep=timestep,
                return_all_hidden_states=return_all_hidden_states,
                image_mask=backbone_output.image_mask,
                backbone_attention_mask=backbone_output.backbone_attention_mask,
            )
        return self.model(
            hidden_states=hidden_states,
            encoder_hidden_states=vl_embeds,
            attention_mask=None,
            encoder_attention_mask=backbone_output.backbone_attention_mask,
            timestep=timestep,
            return_all_hidden_states=return_all_hidden_states,
        )

    def _progress_logits(self, progress_hidden: torch.Tensor) -> torch.Tensor:
        progress_head_dtype = next(self.progress_head.parameters()).dtype
        device_type = progress_hidden.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            progress_hidden = torch.nan_to_num(
                progress_hidden.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4,
            ).to(dtype=progress_head_dtype)
            progress_pred = self.progress_head(progress_hidden)
            progress_pred = torch.nan_to_num(
                progress_pred.float(),
                nan=0.0,
                posinf=30.0,
                neginf=-30.0,
            )
            if self.progress_output_type == "scalar":
                return progress_pred.squeeze(-1)
            return progress_pred

    def _predict_progress(self, progress_hidden: torch.Tensor) -> torch.Tensor:
        return self._progress_pred_from_logits(self._progress_logits(progress_hidden))

    def _progress_pred_from_logits(self, progress_logits: torch.Tensor) -> torch.Tensor:
        if self.progress_output_type == "scalar":
            return torch.sigmoid(progress_logits)
        bin_centers = self.progress_bin_centers.to(
            device=progress_logits.device,
            dtype=progress_logits.dtype,
        )
        if self.progress_output_type == "hard_bins":
            return bin_centers[self._progress_class_from_logits(progress_logits)]
        return (torch.softmax(progress_logits, dim=-1) * bin_centers).sum(dim=-1)

    def _progress_class_from_logits(self, progress_logits: torch.Tensor) -> torch.Tensor:
        if self.progress_output_type == "scalar":
            progress_pred = torch.sigmoid(progress_logits)
            return self._progress_target_classes(progress_pred)
        return progress_logits.argmax(dim=-1)

    def _progress_target_classes(self, progress_target: torch.Tensor) -> torch.Tensor:
        target = progress_target.clamp(0.0, 1.0)
        return torch.clamp(
            (target * self.progress_num_bins).long(),
            min=0,
            max=self.progress_num_bins - 1,
        )

    def _make_soft_progress_targets(self, progress_target: torch.Tensor) -> torch.Tensor:
        bin_centers = self.progress_bin_centers.to(
            device=progress_target.device,
            dtype=progress_target.dtype,
        )
        distances = progress_target.unsqueeze(-1) - bin_centers
        soft_targets = torch.exp(-0.5 * (distances / self.progress_soft_label_sigma).square())
        return soft_targets / soft_targets.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _progress_loss(
        self,
        progress_logits: torch.Tensor,
        progress_target: torch.Tensor,
    ) -> torch.Tensor:
        if self.progress_output_type == "scalar":
            return F.binary_cross_entropy_with_logits(progress_logits, progress_target)
        if self.progress_output_type == "hard_bins":
            return F.cross_entropy(progress_logits, self._progress_target_classes(progress_target))
        soft_targets = self._make_soft_progress_targets(progress_target)
        log_probs = F.log_softmax(progress_logits, dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1).mean()

    def forward(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        progress_backbone_output: BatchFeature | None = None,
    ) -> BatchFeature:
        """
        Forward pass through the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - action: [B, action_horizon, action_dim] (during training)
                - embodiment_id: [B] (embodiment IDs)
                - action_mask: [B, action_horizon, action_dim]

        Returns:
            BatchFeature containing:
                - loss: action prediction loss
        """
        # Set frozen modules to eval
        self.set_frozen_modules_to_eval_mode()

        vlm_progress_hidden = None
        if self._uses_vlm_progress_head():
            vlm_progress_hidden = self._compute_vlm_progress_hidden(backbone_output)
        elif self._uses_vlm_dit_progress_head() and progress_backbone_output is None:
            raise ValueError(
                "progress_backbone_output is required for progress_head_source='vlm_dit'"
            )

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        device = vl_embeds.device

        # Get embodiment ID.
        embodiment_id = action_input.embodiment_id

        # Handle state history
        assert action_input.state.shape[1] == self.config.state_history_length
        action_input.state = action_input.state.view(action_input.state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(action_input.state, embodiment_id)

        # Dropout state features (training only): zero out dropped states.
        if self.training and self.state_dropout_prob > 0:
            do_dropout = (
                torch.rand(state_features.shape[0], device=state_features.device)
                < self.state_dropout_prob
            )
            do_dropout = do_dropout[:, None, None].to(dtype=state_features.dtype)
            state_features = state_features * (1 - do_dropout)

        if self._uses_vlm_pooled_progress_head():
            vlm_progress_hidden = self._compute_vlm_pooled_progress_hidden(
                backbone_output=backbone_output,
                state_features=state_features,
            )
        elif self._uses_vlm_pooled_dit_progress_head():
            vlm_progress_hidden = self._compute_vlm_pooled_dit_progress_hidden(
                state_features=state_features,
                backbone_output=backbone_output,
            )
        elif self._uses_state_multilayer_dit_progress_head():
            vlm_progress_hidden = self._compute_state_multilayer_dit_progress_hidden(
                state_features=state_features,
                backbone_output=backbone_output,
            )
        elif self._uses_vlm_layer_pooled_progress_head():
            vlm_progress_hidden = self._compute_vlm_layer_pooled_progress_hidden(
                backbone_output=backbone_output,
            )

        # Embed noised action trajectory.
        actions = action_input.action
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        # Convert (continuous) t -> discrete if needed
        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized, embodiment_id)

        # Maybe add position embedding.
        action_position_features = None
        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_position_features = pos_embs.expand(action_features.shape[0], -1, -1)
            action_features = action_features + pos_embs

        # Join state, action tokens, and optional auxiliary progress token. In
        # isolated mode, the action route stays identical to the original model
        # and the progress route gets action-position placeholders only.
        action_start = state_features.shape[1]
        action_end = action_start + actions.shape[1]
        progress_index = None
        progress_sa_embs = None
        if self._uses_non_action_progress_head():
            sa_embs = torch.cat((state_features, action_features), dim=1)
        elif self.enable_progress_head and self.isolate_progress_action_attention:
            sa_embs = torch.cat((state_features, action_features), dim=1)
            progress_sa_embs, progress_index = self._make_progress_route_features(
                state_features=state_features,
                action_features=action_features,
                action_position_features=action_position_features,
                batch_size=actions.shape[0],
                device=device,
            )
        elif self.enable_progress_head:
            progress_index = action_end
            progress_features = self._make_progress_features(
                batch_size=actions.shape[0],
                position_index=action_features.shape[1],
                device=device,
                dtype=action_features.dtype,
            )
            sa_embs = torch.cat((state_features, action_features, progress_features), dim=1)
        else:
            sa_embs = torch.cat((state_features, action_features), dim=1)
        model_output, _ = self._run_model(
            hidden_states=sa_embs,
            vl_embeds=vl_embeds,
            timestep=t_discretized,
            backbone_output=backbone_output,
            return_all_hidden_states=True,
        )

        pred_actions = self.action_decoder(model_output[:, action_start:action_end], embodiment_id)

        # Slice out only the action portion of pred and target.
        action_mask = action_input.action_mask
        action_loss = F.mse_loss(pred_actions, velocity, reduction="none") * action_mask
        loss = action_loss.sum() / (action_mask.sum() + 1e-6)
        output_loss = loss if self.optimize_action_loss else loss.detach()
        output = {
            "loss": output_loss,
            "action_loss": action_loss,
            "action_mask": action_mask,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }

        if self.enable_progress_head:
            if (
                self._uses_vlm_progress_head()
                or self._uses_vlm_pooled_progress_head()
                or self._uses_vlm_pooled_dit_progress_head()
                or self._uses_state_multilayer_dit_progress_head()
                or self._uses_vlm_layer_pooled_progress_head()
            ):
                progress_logits = self._progress_logits(vlm_progress_hidden)
            elif self._uses_vlm_dit_progress_head():
                progress_hidden = self._compute_vlm_dit_progress_hidden(
                    state_features=state_features,
                    progress_backbone_output=progress_backbone_output,
                )
                progress_logits = self._progress_logits(progress_hidden)
            else:
                progress_output = model_output
                if progress_sa_embs is not None:
                    progress_output, _ = self._run_model(
                        hidden_states=progress_sa_embs,
                        vl_embeds=vl_embeds,
                        timestep=t_discretized,
                        backbone_output=backbone_output,
                        return_all_hidden_states=True,
                    )
                progress_logits = self._progress_logits(progress_output[:, progress_index])
            progress_pred = self._progress_pred_from_logits(progress_logits)
            output["progress_logits"] = progress_logits
            output["progress_pred"] = progress_pred
            if self.progress_output_type in {"soft_bins", "hard_bins"}:
                output["progress_class_pred"] = self._progress_class_from_logits(progress_logits)
            if "progress" in action_input:
                progress_target = action_input.progress.to(
                    device=progress_pred.device,
                    dtype=progress_pred.dtype,
                ).view_as(progress_pred)
                progress_target = progress_target.clamp(0.0, 1.0)
                progress_loss = self._progress_loss(progress_logits, progress_target)
                output["progress_loss"] = progress_loss
                if self.progress_output_type in {"soft_bins", "hard_bins"}:
                    output["progress_class_target"] = self._progress_target_classes(progress_target)
                if self.optimize_action_loss:
                    output["loss"] = loss + self.progress_loss_weight * progress_loss
                else:
                    output["loss"] = self.progress_loss_weight * progress_loss

        return output

    def _encode_features(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        progress_backbone_output: BatchFeature | None = None,
    ) -> BatchFeature:
        """
        Encode features for the action head.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_history_length, max_state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - state_features: [B, 1, input_embedding_dim]
        """
        vlm_progress_hidden = None
        if self._uses_vlm_progress_head():
            vlm_progress_hidden = self._compute_vlm_progress_hidden(backbone_output)
        elif self._uses_vlm_dit_progress_head() and progress_backbone_output is None:
            raise ValueError(
                "progress_backbone_output is required for progress_head_source='vlm_dit'"
            )

        backbone_output = self.process_backbone_output(backbone_output)

        # Get vision and language embeddings.
        vl_embeds = backbone_output.backbone_features
        embodiment_id = action_input.embodiment_id

        # Handle state history: if we have fewer timesteps than expected, repeat to fill
        state = action_input.state
        current_T = state.shape[1]
        assert current_T == self.config.state_history_length, "current_T != state_history_length"
        # Reshape state from [B, state_history_length, max_state_dim] to [B, 1, state_history_length * max_state_dim]
        state = state.view(state.shape[0], 1, -1)

        # Embed state.
        state_features = self.state_encoder(state, embodiment_id)

        if self._uses_vlm_pooled_progress_head():
            vlm_progress_hidden = self._compute_vlm_pooled_progress_hidden(
                backbone_output=backbone_output,
                state_features=state_features,
            )
        elif self._uses_vlm_pooled_dit_progress_head():
            vlm_progress_hidden = self._compute_vlm_pooled_dit_progress_hidden(
                state_features=state_features,
                backbone_output=backbone_output,
            )
        elif self._uses_state_multilayer_dit_progress_head():
            vlm_progress_hidden = self._compute_state_multilayer_dit_progress_hidden(
                state_features=state_features,
                backbone_output=backbone_output,
            )
        elif self._uses_vlm_layer_pooled_progress_head():
            vlm_progress_hidden = self._compute_vlm_layer_pooled_progress_hidden(
                backbone_output=backbone_output,
            )

        features = {"backbone_features": vl_embeds, "state_features": state_features}
        if vlm_progress_hidden is not None:
            features["progress_hidden"] = vlm_progress_hidden
        if self._uses_vlm_dit_progress_head():
            features["progress_backbone_output"] = progress_backbone_output
        return BatchFeature(data=features)

    @torch.no_grad()
    def get_action_with_features(
        self,
        backbone_features: torch.Tensor,
        state_features: torch.Tensor,
        embodiment_id: torch.Tensor,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        progress_hidden: torch.Tensor | None = None,
        progress_backbone_output: BatchFeature | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_features: [B, seq_len, backbone_embedding_dim]
            state_features: [B, state_horizon, input_embedding_dim]
            embodiment_id: [B] (embodiment IDs)
            backbone_output: Output from the backbone model
        """
        vl_embeds = backbone_features

        # Set initial actions as the sampled noise.
        batch_size = vl_embeds.shape[0]
        device = vl_embeds.device
        actions = torch.randn(
            size=(batch_size, self.config.action_horizon, self.action_dim),
            dtype=vl_embeds.dtype,
            device=device,
        )

        dt = 1.0 / self.num_inference_timesteps
        vel_strength = torch.ones_like(actions)
        progress_logits = None
        progress_pred = None
        progress_class_pred = None
        if (
            self._uses_vlm_progress_head()
            or self._uses_vlm_pooled_progress_head()
            or self._uses_vlm_pooled_dit_progress_head()
            or self._uses_state_multilayer_dit_progress_head()
            or self._uses_vlm_layer_pooled_progress_head()
        ):
            assert progress_hidden is not None
            progress_logits = self._progress_logits(progress_hidden)
            progress_pred = self._progress_pred_from_logits(progress_logits)
            if self.progress_output_type in {"soft_bins", "hard_bins"}:
                progress_class_pred = self._progress_class_from_logits(progress_logits)
        elif self._uses_vlm_dit_progress_head():
            assert progress_backbone_output is not None
            progress_hidden = self._compute_vlm_dit_progress_hidden(
                state_features=state_features,
                progress_backbone_output=progress_backbone_output,
            )
            progress_logits = self._progress_logits(progress_hidden)
            progress_pred = self._progress_pred_from_logits(progress_logits)
            if self.progress_output_type in {"soft_bins", "hard_bins"}:
                progress_class_pred = self._progress_class_from_logits(progress_logits)

        if "action" in action_input:
            # If action in input when doing get action, it means we want to use RTC.
            # action_horizon is the action horizon of the input action.
            # rtc_overlap_steps is the number of steps to overlap with the previous action chunks.
            # rtc_frozen_steps is the number of steps to freeze the action, which is the latency of the policy inference.
            # rtc_ramp_rate is the rate of the ramp of denoising the actions.
            assert options is not None, "options is not None"
            assert "action_horizon" in options, "action_horizon is not in options"
            assert "rtc_overlap_steps" in options, "rtc_overlap_steps is not in options"
            assert "rtc_frozen_steps" in options, "rtc_frozen_steps is not in options"
            assert "rtc_ramp_rate" in options, "rtc_ramp_rate is not in options"

            action_horizon_before_padding = options["action_horizon"]

            # Use previous action instead of pure noise to do inpainting
            actions[:, : options["rtc_overlap_steps"], :] = action_input["action"][
                :,
                action_horizon_before_padding
                - options["rtc_overlap_steps"] : action_horizon_before_padding,
                :,
            ]
            vel_strength[:, : options["rtc_frozen_steps"], :] = 0.0
            # NOTE: use an exponential ramp strength to set the remaining unfrozen rtc_steps
            intermediate_steps = options["rtc_overlap_steps"] - options["rtc_frozen_steps"]
            # Create exponential ramp from 0 to 1 over intermediate steps
            t = torch.linspace(0.0, 1.0, intermediate_steps + 2, device=device)
            ramp = 1 - torch.exp(-options["rtc_ramp_rate"] * t)
            ramp = ramp / ramp[-1].clamp_min(1e-8)  # normalize to [0,1]
            ramp = ramp[
                1:-1
            ]  # we will only take the middle part of the ramp, ignore the 0.0 and 1.0
            # Apply ramp to the intermediate steps [batch, intermediate_steps, action_dim]
            vel_strength[
                :,
                options["rtc_frozen_steps"] : options["rtc_overlap_steps"],
                :,
            ] = ramp[None, :, None].to(device)

        # Run denoising steps.
        for t in range(self.num_inference_timesteps):
            t_cont = t / float(self.num_inference_timesteps)  # e.g. goes 0, 1/N, 2/N, ...
            t_discretized = int(t_cont * self.num_timestep_buckets)

            # Embed noised action trajectory.
            timesteps_tensor = torch.full(
                size=(batch_size,), fill_value=t_discretized, device=device
            )
            action_features = self.action_encoder(actions, timesteps_tensor, embodiment_id)
            # Add position embedding.
            action_position_features = None
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_position_features = pos_embs.expand(action_features.shape[0], -1, -1)
                action_features = action_features + pos_embs

            # Join state, action tokens, and optional auxiliary progress token.
            action_start = state_features.shape[1]
            action_end = action_start + self.action_horizon
            progress_index = None
            progress_sa_embs = None
            if self._uses_non_action_progress_head():
                sa_embs = torch.cat((state_features, action_features), dim=1)
            elif self.enable_progress_head and self.isolate_progress_action_attention:
                sa_embs = torch.cat((state_features, action_features), dim=1)
                progress_sa_embs, progress_index = self._make_progress_route_features(
                    state_features=state_features,
                    action_features=action_features,
                    action_position_features=action_position_features,
                    batch_size=batch_size,
                    device=device,
                )
            elif self.enable_progress_head:
                progress_index = action_end
                progress_features = self._make_progress_features(
                    batch_size=batch_size,
                    position_index=action_features.shape[1],
                    device=device,
                    dtype=action_features.dtype,
                )
                sa_embs = torch.cat((state_features, action_features, progress_features), dim=1)
            else:
                sa_embs = torch.cat((state_features, action_features), dim=1)

            # Run model forward.
            model_output = self._run_model(
                hidden_states=sa_embs,
                vl_embeds=vl_embeds,
                timestep=timesteps_tensor,
                backbone_output=backbone_output,
            )
            pred_velocity = self.action_decoder(
                model_output[:, action_start:action_end],
                embodiment_id,
            )
            if self.enable_progress_head and not self._uses_non_action_progress_head():
                progress_output = model_output
                if progress_sa_embs is not None:
                    progress_output = self._run_model(
                        hidden_states=progress_sa_embs,
                        vl_embeds=vl_embeds,
                        timestep=timesteps_tensor,
                        backbone_output=backbone_output,
                    )
                progress_logits = self._progress_logits(progress_output[:, progress_index])
                progress_pred = self._progress_pred_from_logits(progress_logits)
                if self.progress_output_type in {"soft_bins", "hard_bins"}:
                    progress_class_pred = self._progress_class_from_logits(progress_logits)

            # Update actions using euler integration.
            actions = actions + dt * pred_velocity * vel_strength

        output = {
            "action_pred": actions,
            "backbone_features": vl_embeds,
            "state_features": state_features,
        }
        if self.enable_progress_head:
            output["progress_logits"] = progress_logits
            output["progress_pred"] = progress_pred
            if progress_class_pred is not None:
                output["progress_class_pred"] = progress_class_pred
        return BatchFeature(data=output)

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
        progress_backbone_output: BatchFeature | None = None,
    ) -> BatchFeature:
        """
        Generate actions using the flow matching diffusion process.

        Args:
            backbone_output: Output from the backbone model containing:
                - backbone_features: [B, seq_len, backbone_embedding_dim]
                - backbone_attention_mask: [B, seq_len]
            action_input: Input containing:
                - state: [B, state_dim]
                - embodiment_id: [B] (embodiment IDs)

        Returns:
            BatchFeature containing:
                - action_pred: [B, action_horizon, action_dim] predicted actions
        """
        features = self._encode_features(
            backbone_output,
            action_input,
            progress_backbone_output=progress_backbone_output,
        )
        return self.get_action_with_features(
            backbone_features=features.backbone_features,
            state_features=features.state_features,
            embodiment_id=action_input.embodiment_id,
            backbone_output=backbone_output,
            action_input=action_input,
            options=options,
            progress_hidden=features.get("progress_hidden"),
            progress_backbone_output=features.get("progress_backbone_output"),
        )

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype

    def prepare_input(self, batch: dict) -> BatchFeature:
        """Prepare input batch for the action head."""
        return BatchFeature(data=batch)


def get_backbone_cls(config: Gr00tN1d7Config):
    if "nvidia/Cosmos-Reason2" in config.model_name or "Qwen/Qwen3-VL" in config.model_name:
        # We import here as Qwen3Backbone depends on newer transformers versions than the rest of the code.
        from gr00t.model.modules.qwen3_backbone import Qwen3Backbone

        return Qwen3Backbone
    else:
        raise ValueError(f"Unsupported model name: {config.model_name}")


class Gr00tN1d7(PreTrainedModel):
    """Gr00tN1d7: VLA model with Cosmos-Reason2-2B (Qwen3-VL) backbone."""

    config_class = Gr00tN1d7Config
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7Config,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        """
        Initialize Gr00tN1d7 model.

        Args:
            config: Model configuration
            transformers_loading_kwargs: Dict with transformers loading parameters:
                - transformers_trust_remote_code: Whether to trust remote code when loading from HF Hub
                - transformers_local_files_only: Whether to only use local files
                - model_revision: Specific model revision to use
                - transformers_cache_dir: Directory to cache downloaded models
                - transformers_access_token: HuggingFace access token for gated models

        Note: During training, transformers parameters are passed from training config.
              During inference (e.g., from_pretrained), defaults are used.
        """
        super().__init__(config)
        self.config = config

        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

        # Initialize action head
        self.action_head = Gr00tN1d7ActionHead(config)
        from .processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> Tuple[BatchFeature, BatchFeature]:
        """Prepare inputs for backbone and action head."""

        # NOTE -- currently the eval code doesn't use collator, so we need to add it here
        # this should ideally be fixed upstream
        if "vlm_content" in inputs:
            # Fix for n_envs > 1: Process all environments' VLM content, not just the first
            vlm_content_list = inputs["vlm_content"]
            # Ensure vlm_content_list is always a list for consistent processing
            if not isinstance(vlm_content_list, list):
                vlm_content_list = [vlm_content_list]

            # Process all VLM contents through the collator
            prep = self.collator([{"vlm_content": vlm} for vlm in vlm_content_list])["inputs"]
            inputs.pop("vlm_content")
            inputs.update(prep)

        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        # Move to device and dtype
        def to_device_with_dtype(x):
            if torch.is_floating_point(x):
                return x.to(self.device, dtype=self.dtype)
            else:
                return x.to(self.device)

        backbone_inputs = tree.map_structure(to_device_with_dtype, backbone_inputs)
        action_inputs = tree.map_structure(to_device_with_dtype, action_inputs)

        return backbone_inputs, action_inputs

    def forward(self, inputs: dict) -> BatchFeature:
        """
        Forward pass through the complete model.

        Args:
            inputs: Dictionary containing:
                - Action inputs (state, action, embodiment_id, etc.)

        Returns:
            BatchFeature containing loss and other outputs
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        backbone_outputs = self.backbone(
            backbone_inputs,
            return_hidden_states=self.action_head.uses_vlm_layer_hidden_states(),
        )
        progress_backbone_outputs = None
        if self.action_head.uses_backbone_progress_token():
            progress_token = self.action_head.make_backbone_progress_token(
                batch_size=backbone_inputs.input_ids.shape[0],
                device=backbone_inputs.input_ids.device,
            )
            progress_backbone_outputs = self.backbone(
                backbone_inputs, progress_token=progress_token
            )
        action_outputs = self.action_head(
            backbone_outputs,
            action_inputs,
            progress_backbone_output=progress_backbone_outputs,
        )

        return action_outputs

    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        """
        Generate actions using the complete model.
        """
        # Prepare inputs for backbone and action head
        backbone_inputs, action_inputs = self.prepare_input(inputs)

        # Forward through backbone
        backbone_outputs = self.backbone(
            backbone_inputs,
            return_hidden_states=self.action_head.uses_vlm_layer_hidden_states(),
        )
        progress_backbone_outputs = None
        if self.action_head.uses_backbone_progress_token():
            progress_token = self.action_head.make_backbone_progress_token(
                batch_size=backbone_inputs.input_ids.shape[0],
                device=backbone_inputs.input_ids.device,
            )
            progress_backbone_outputs = self.backbone(
                backbone_inputs, progress_token=progress_token
            )
        action_outputs = self.action_head.get_action(
            backbone_outputs,
            action_inputs,
            options,
            progress_backbone_output=progress_backbone_outputs,
        )

        return action_outputs

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


# Register the model with HuggingFace
AutoConfig.register("Gr00tN1d7", Gr00tN1d7Config)
AutoModel.register(Gr00tN1d7Config, Gr00tN1d7)
