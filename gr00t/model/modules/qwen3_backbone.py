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

import torch
from transformers.feature_extraction_utils import BatchFeature


logger = logging.getLogger(__name__)


try:
    from transformers import Qwen3VLForConditionalGeneration

    _QWEN3VL_AVAILABLE = True
except ImportError:
    _QWEN3VL_AVAILABLE = False


class Qwen3Backbone(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "nvidia/Cosmos-Reason2-2B",
        tune_llm: bool = False,
        tune_visual: bool = False,
        select_layer: int = -1,
        reproject_vision: bool = True,
        use_flash_attention: bool = False,
        projector_dim: int = -1,
        load_bf16: bool = False,
        tune_top_llm_layers: int = 0,
        trainable_params_fp32: bool = False,
        transformers_loading_kwargs: dict = {},
    ):
        """
        Qwen3Backbone is to generate n_queries to represent the future action hidden states.
        Args:
            model_name: nvidia/Cosmos-Reason2-2B
            tune_llm: whether to tune the LLM model (default: False)
            tune_visual: whether to tune the visual model (default: False)
        """
        if not _QWEN3VL_AVAILABLE:
            raise ImportError(
                "Qwen3VLForConditionalGeneration is not available. "
                "Please upgrade transformers to a version that supports Qwen3-VL: "
                "pip install transformers>=4.57.0"
            )

        super().__init__()

        # Add attention kwargs
        extra_kwargs = {}
        if use_flash_attention:
            try:
                import flash_attn  # noqa: F401

                extra_kwargs["attn_implementation"] = "flash_attention_2"
            except ImportError:
                logger.warning(
                    "flash_attn is not installed. Falling back to sdpa attention. "
                    "Install flash-attn for better performance: pip install flash-attn"
                )
                extra_kwargs["attn_implementation"] = "sdpa"
        if load_bf16:
            extra_kwargs["torch_dtype"] = torch.bfloat16

        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name,
            **extra_kwargs,
            **transformers_loading_kwargs,
        ).eval()

        # needed since we don't use these layers. Also saves compute
        while len(self.model.language_model.layers) > select_layer:
            self.model.language_model.layers.pop(-1)

        self.select_layer = select_layer
        self.set_trainable_parameters(tune_llm, tune_visual, tune_top_llm_layers)
        if load_bf16 and trainable_params_fp32:
            # cast trainable parameters to fp32
            for n, p in self.named_parameters():
                if p.requires_grad:
                    p.data = p.data.to(torch.float32)
                    logger.debug(f"Casting trainable parameter {n} to fp32")

    def set_trainable_parameters(self, tune_llm: bool, tune_visual: bool, tune_top_llm_layers: int):
        self.tune_llm = tune_llm
        self.tune_visual = tune_visual
        for p in self.parameters():
            p.requires_grad = True
        if not tune_llm:
            self.model.language_model.requires_grad_(False)
        if not tune_visual:
            self.model.visual.requires_grad_(False)

        if tune_top_llm_layers > 0:
            for layer in self.model.language_model.layers[-tune_top_llm_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True

        logger.debug(f"Tune backbone llm: {self.tune_llm}")
        logger.debug(f"Tune backbone visual: {self.tune_visual}")
        # Check if any parameters are still trainable. If not, log a warning.
        for name, p in self.named_parameters():
            if p.requires_grad:
                logger.debug(f"Backbone trainable parameter: {name}")
        if not any(p.requires_grad for p in self.parameters()):
            logger.warning("No backbone trainable parameters found.")

    def set_frozen_modules_to_eval_mode(self):
        """
        Huggingface will call model.train() at each training_step. To ensure
        the expected behaviors for modules like dropout, batchnorm, etc., we
        need to call model.eval() for the frozen modules.
        """
        if self.training:
            if self.model.language_model and not self.tune_llm:
                self.model.language_model.eval()
            if self.model.visual and not self.tune_visual:
                self.model.visual.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _resolve_progress_token_id(self) -> int:
        token_id = getattr(self.model.config, "eos_token_id", None)
        text_config = getattr(self.model.config, "text_config", None)
        if token_id is None:
            token_id = getattr(text_config, "eos_token_id", None)
        if isinstance(token_id, (list, tuple)):
            token_id = token_id[0]
        if token_id is None:
            token_id = 0
        return int(token_id)

    def _resolve_pad_token_id(self) -> int:
        token_id = getattr(self.model.config, "pad_token_id", None)
        text_config = getattr(self.model.config, "text_config", None)
        if token_id is None:
            token_id = getattr(text_config, "pad_token_id", None)
        if token_id is None:
            token_id = 0
        return int(token_id)

    def _forward_with_progress_token(
        self,
        vl_input: dict[str, torch.Tensor],
        progress_token: torch.Tensor,
    ) -> BatchFeature:
        input_ids = vl_input["input_ids"]
        attention_mask = vl_input["attention_mask"]
        batch_size, seq_len = input_ids.shape
        embedding_layer = self.model.model.get_input_embeddings()
        inputs_embeds = embedding_layer(input_ids)

        progress_token = progress_token.to(
            device=inputs_embeds.device,
            dtype=inputs_embeds.dtype,
        )
        if progress_token.shape[0] == 1:
            progress_token = progress_token.expand(batch_size, -1, -1)

        pad_token_id = self._resolve_pad_token_id()
        progress_token_id = self._resolve_progress_token_id()
        progress_input_ids = input_ids.new_full((batch_size, seq_len + 1), pad_token_id)
        progress_attention_mask = attention_mask.new_zeros((batch_size, seq_len + 1))
        progress_inputs_embeds = inputs_embeds.new_zeros(
            batch_size,
            seq_len + 1,
            inputs_embeds.shape[-1],
        )
        progress_token_index = attention_mask.to(torch.long).sum(dim=1)

        for batch_idx in range(batch_size):
            insert_at = int(progress_token_index[batch_idx].item())
            progress_input_ids[batch_idx, :insert_at] = input_ids[batch_idx, :insert_at]
            progress_input_ids[batch_idx, insert_at] = progress_token_id
            progress_input_ids[batch_idx, insert_at + 1 :] = input_ids[batch_idx, insert_at:]
            progress_attention_mask[batch_idx, :insert_at] = attention_mask[batch_idx, :insert_at]
            progress_attention_mask[batch_idx, insert_at] = 1
            progress_attention_mask[batch_idx, insert_at + 1 :] = attention_mask[
                batch_idx, insert_at:
            ]
            progress_inputs_embeds[batch_idx, :insert_at] = inputs_embeds[batch_idx, :insert_at]
            progress_inputs_embeds[batch_idx, insert_at] = progress_token[batch_idx, 0]
            progress_inputs_embeds[batch_idx, insert_at + 1 :] = inputs_embeds[
                batch_idx, insert_at:
            ]

        position_ids, _ = self.model.model.get_rope_index(
            progress_input_ids,
            vl_input.get("image_grid_thw"),
            None,
            attention_mask=progress_attention_mask,
        )
        outputs = self.model.model(
            input_ids=None,
            inputs_embeds=progress_inputs_embeds,
            attention_mask=progress_attention_mask,
            position_ids=position_ids,
            pixel_values=vl_input.get("pixel_values"),
            image_grid_thw=vl_input.get("image_grid_thw"),
        )
        image_mask = progress_input_ids == self.model.config.image_token_id
        return BatchFeature(
            data={
                "backbone_features": outputs.last_hidden_state,
                "backbone_attention_mask": progress_attention_mask == 1,
                "image_mask": image_mask,
                "progress_token_index": progress_token_index,
            }
        )

    def forward(
        self,
        vl_input: BatchFeature,
        progress_token: torch.Tensor | None = None,
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        # 0. Set frozen module to eval
        keys_to_use = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vl_input = {k: vl_input[k] for k in keys_to_use}
        if progress_token is not None:
            return self._forward_with_progress_token(vl_input, progress_token)

        outputs = self.model(**vl_input, output_hidden_states=True)
        outputs = outputs.hidden_states[-1]
        image_mask = vl_input["input_ids"] == self.model.config.image_token_id
        attention_mask = vl_input["attention_mask"] == 1
        return BatchFeature(
            data={
                "backbone_features": outputs,
                "backbone_attention_mask": attention_mask,
                "image_mask": image_mask,
            }
        )  # [B, T2, hidden_size]
