# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test Gr00tN1d7ActionHead: flow matching forward, get_action, feature encoding.

These tests instantiate the action head directly (no backbone required)
and feed it synthetic backbone output tensors.
"""

from gr00t.configs.model.gr00t_n1d7 import Gr00tN1d7Config
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import Gr00tN1d7ActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _small_config(**overrides) -> Gr00tN1d7Config:
    defaults = dict(
        backbone_embedding_dim=64,
        hidden_size=64,
        input_embedding_dim=64,
        max_state_dim=7,
        max_action_dim=7,
        action_horizon=4,
        state_history_length=1,
        num_inference_timesteps=2,
        max_num_embodiments=4,
        add_pos_embed=True,
        use_vlln=True,
        max_seq_len=32,
        use_alternate_vl_dit=False,
        attend_text_every_n_blocks=2,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
        state_dropout_prob=0.0,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        noise_s=0.999,
        num_timestep_buckets=1000,
        attn_dropout=0.0,
        diffusion_model_cfg={
            "positional_embeddings": None,
            "num_layers": 2,
            "num_attention_heads": 2,
            "attention_head_dim": 32,
            "norm_type": "ada_norm",
            "dropout": 0.0,
            "final_dropout": False,
            "output_dim": 64,
            "interleave_self_attention": True,
        },
    )
    defaults.update(overrides)
    return Gr00tN1d7Config(**defaults)


@pytest.fixture
def action_head():
    config = _small_config()
    head = Gr00tN1d7ActionHead(config)
    head.eval()
    return head, config


def _make_backbone_output(config, batch_size=2, seq_len=8, num_hidden_states=0):
    data = {
        "backbone_features": torch.randn(batch_size, seq_len, config.backbone_embedding_dim),
        "backbone_attention_mask": torch.ones(batch_size, seq_len, dtype=torch.long),
        "image_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
    }
    if num_hidden_states:
        data["backbone_hidden_states"] = tuple(
            torch.randn(batch_size, seq_len, config.backbone_embedding_dim)
            for _ in range(num_hidden_states)
        )
    return BatchFeature(data=data)


def _make_progress_backbone_output(config, batch_size=2, seq_len=9):
    output = _make_backbone_output(config, batch_size=batch_size, seq_len=seq_len)
    output["progress_token_index"] = torch.full((batch_size,), seq_len - 1, dtype=torch.long)
    output["image_mask"][:, -1] = False
    return output


def _make_action_input(config, batch_size=2):
    data = {
        "state": torch.randn(batch_size, config.state_history_length, config.max_state_dim),
        "action": torch.randn(batch_size, config.action_horizon, config.max_action_dim),
        "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
        "action_mask": torch.ones(batch_size, config.action_horizon, config.max_action_dim),
    }
    if config.enable_progress_head:
        data["progress"] = torch.linspace(0.0, 1.0, batch_size)
    return BatchFeature(data=data)


class TestActionHeadForward:
    """Test training forward pass."""

    def test_forward_returns_loss(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "loss" in out
        assert out["loss"].dim() == 0
        assert torch.isfinite(out["loss"])

    def test_forward_loss_shape(self, action_head):
        head, config = action_head
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert out["action_loss"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_forward_with_state_dropout(self):
        config = _small_config(state_dropout_prob=0.5)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert torch.isfinite(out["loss"])

    def test_forward_with_progress_head(self):
        config = _small_config(enable_progress_head=True, progress_loss_weight=0.2)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "progress_pred" in out
        assert "progress_loss" in out
        assert out["progress_pred"].shape == (2,)
        assert torch.allclose(out["progress_pred"], torch.full_like(out["progress_pred"], 0.5))
        assert torch.isfinite(out["loss"])

    def test_forward_with_progress_head_and_alternate_vl_dit(self):
        config = _small_config(enable_progress_head=True, use_alternate_vl_dit=True)
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))
        assert "progress_pred" in out
        assert out["progress_pred"].shape == (2,)
        assert torch.isfinite(out["loss"])

    @pytest.mark.parametrize(
        "source",
        [
            "vlm_pooled",
            "vlm_pooled_state",
            "vlm_concat_linear",
            "vlm_concat_projected_linear",
            "vlm_concat_attention_pool",
            "vlm_layer_pooled",
            "vlm_layer_concat_linear",
            "vlm_layer_concat_projected_linear",
            "vlm_layer_concat_attention_pool",
        ],
    )
    def test_forward_with_vlm_pooled_progress_head(self, source):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source=source,
            progress_vlm_layer=1,
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(
            _make_backbone_output(config, num_hidden_states=2),
            _make_action_input(config),
        )

        assert "progress_pred" in out
        assert "progress_loss" in out
        assert out["progress_pred"].shape == (2,)
        assert torch.isfinite(out["loss"])
        assert not hasattr(head, "progress_token")
        expected_dim = config.backbone_embedding_dim
        if source == "vlm_pooled_state":
            expected_dim += config.input_embedding_dim
        elif source == "vlm_concat_linear":
            expected_dim *= config.max_seq_len
            assert len(head.progress_head) == 2
        elif source == "vlm_layer_concat_linear":
            expected_dim *= config.max_seq_len
            assert len(head.progress_head) == 2
        elif source in {"vlm_concat_projected_linear", "vlm_layer_concat_projected_linear"}:
            expected_dim = config.max_seq_len * config.progress_concat_project_dim
            assert len(head.progress_head) == 2
        elif source in {"vlm_concat_attention_pool", "vlm_layer_concat_attention_pool"}:
            expected_dim = config.progress_concat_project_dim
            assert len(head.progress_head) == 2
        assert head.progress_head[0].normalized_shape == (expected_dim,)

    def test_vlm_concat_linear_progress_head_masks_and_pads_tokens(self):
        config = _small_config(enable_progress_head=True, progress_head_source="vlm_concat_linear")
        head = Gr00tN1d7ActionHead(config)
        backbone_output = _make_backbone_output(config, batch_size=1, seq_len=4)
        backbone_output.backbone_features = torch.ones_like(backbone_output.backbone_features)
        backbone_output.backbone_attention_mask[:, 2:] = 0

        hidden = head._concat_vlm_features(backbone_output)

        expected = torch.zeros(1, config.max_seq_len, config.backbone_embedding_dim)
        expected[:, :2] = 1
        assert hidden.shape == (1, config.max_seq_len * config.backbone_embedding_dim)
        assert torch.equal(hidden, expected.reshape(1, -1))

    def test_vlm_concat_projected_linear_progress_head_masks_and_pads_tokens(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_concat_projected_linear",
            progress_concat_project_dim=8,
        )
        head = Gr00tN1d7ActionHead(config)
        backbone_output = _make_backbone_output(config, batch_size=1, seq_len=4)
        backbone_output.backbone_attention_mask[:, 2:] = 0

        hidden = head._concat_projected_vlm_features(backbone_output)

        expected_shape = (1, config.max_seq_len * config.progress_concat_project_dim)
        assert hidden.shape == expected_shape
        hidden_tokens = hidden.reshape(1, config.max_seq_len, config.progress_concat_project_dim)
        assert torch.equal(hidden_tokens[:, 2:], torch.zeros_like(hidden_tokens[:, 2:]))

    def test_vlm_concat_attention_pool_progress_head_masks_tokens(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_concat_attention_pool",
            progress_concat_project_dim=8,
        )
        head = Gr00tN1d7ActionHead(config)
        backbone_output = _make_backbone_output(config, batch_size=1, seq_len=4)
        backbone_output.backbone_attention_mask[:, 2:] = 0

        hidden = head._attention_pool_vlm_features(backbone_output)

        assert hidden.shape == (1, config.progress_concat_project_dim)
        assert torch.isfinite(hidden).all()

    def test_vlm_layer_concat_linear_progress_head_masks_and_pads_tokens(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_layer_concat_linear",
            progress_vlm_layer=1,
        )
        head = Gr00tN1d7ActionHead(config)
        backbone_output = _make_backbone_output(
            config,
            batch_size=1,
            seq_len=4,
            num_hidden_states=2,
        )
        hidden_state = torch.ones_like(backbone_output.backbone_hidden_states[1])
        backbone_output["backbone_hidden_states"] = (
            torch.zeros_like(hidden_state),
            hidden_state,
        )
        backbone_output.backbone_attention_mask[:, 2:] = 0

        hidden = head._concat_vlm_layer_features(backbone_output)

        expected = torch.zeros(1, config.max_seq_len, config.backbone_embedding_dim)
        expected[:, :2] = 1
        assert hidden.shape == (1, config.max_seq_len * config.backbone_embedding_dim)
        assert torch.equal(hidden, expected.reshape(1, -1))

    def test_vlm_layer_concat_projected_linear_progress_head_masks_and_pads_tokens(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_layer_concat_projected_linear",
            progress_vlm_layer=1,
            progress_concat_project_dim=8,
        )
        head = Gr00tN1d7ActionHead(config)
        backbone_output = _make_backbone_output(
            config,
            batch_size=1,
            seq_len=4,
            num_hidden_states=2,
        )
        hidden_state = torch.ones_like(backbone_output.backbone_hidden_states[1])
        backbone_output["backbone_hidden_states"] = (
            torch.zeros_like(hidden_state),
            hidden_state,
        )
        backbone_output.backbone_attention_mask[:, 2:] = 0

        hidden = head._concat_projected_vlm_layer_features(backbone_output)

        expected_shape = (1, config.max_seq_len * config.progress_concat_project_dim)
        assert hidden.shape == expected_shape
        hidden_tokens = hidden.reshape(1, config.max_seq_len, config.progress_concat_project_dim)
        assert torch.equal(hidden_tokens[:, 2:], torch.zeros_like(hidden_tokens[:, 2:]))

    def test_forward_with_vlm_pooled_dit_progress_head(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_pooled_dit",
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))

        assert "progress_pred" in out
        assert "progress_loss" in out
        assert out["progress_pred"].shape == (2,)
        assert torch.isfinite(out["loss"])
        assert hasattr(head, "progress_token")
        assert not hasattr(head, "progress_vlm_projector")
        assert len(head.progress_head) == 4
        assert head.progress_head[0].normalized_shape == (config.hidden_size,)

    def test_forward_with_soft_bin_progress_head(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_pooled_dit",
            progress_output_type="soft_bins",
            progress_num_bins=10,
            progress_soft_label_sigma=0.08,
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))

        assert "progress_pred" in out
        assert "progress_loss" in out
        assert out["progress_pred"].shape == (2,)
        assert torch.allclose(out["progress_pred"], torch.full_like(out["progress_pred"], 0.5))
        assert torch.isfinite(out["loss"])
        assert head.progress_head[3].out_features == config.progress_num_bins

    def test_forward_with_hard_bin_progress_head(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_pooled_dit",
            progress_output_type="hard_bins",
            progress_num_bins=10,
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))

        assert "progress_pred" in out
        assert "progress_class_pred" in out
        assert "progress_class_target" in out
        assert "progress_loss" in out
        assert out["progress_pred"].shape == (2,)
        assert out["progress_class_pred"].shape == (2,)
        assert torch.equal(out["progress_class_pred"], torch.zeros(2, dtype=torch.long))
        assert torch.equal(out["progress_class_target"], torch.tensor([0, 9]))
        assert torch.allclose(out["progress_pred"], torch.full_like(out["progress_pred"], 0.05))
        assert torch.isfinite(out["loss"])
        assert head.progress_head[3].out_features == config.progress_num_bins

    def test_soft_progress_targets_are_normalized(self):
        config = _small_config(
            enable_progress_head=True,
            progress_output_type="soft_bins",
            progress_num_bins=10,
        )
        head = Gr00tN1d7ActionHead(config)
        target = torch.tensor([0.0, 0.37, 1.0])

        soft_targets = head._make_soft_progress_targets(target)

        assert soft_targets.shape == (3, config.progress_num_bins)
        assert torch.allclose(soft_targets.sum(dim=-1), torch.ones(3))
        assert soft_targets[0].argmax() == 0
        assert soft_targets[-1].argmax() == config.progress_num_bins - 1

    def test_forward_with_state_multilayer_dit_progress_head(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="state_multilayer_dit",
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))

        assert "progress_pred" in out
        assert "progress_loss" in out
        assert out["progress_pred"].shape == (2,)
        assert torch.isfinite(out["loss"])
        assert hasattr(head, "progress_token")
        assert not hasattr(head, "progress_vlm_projector")
        assert len(head.progress_head) == 2
        assert head.progress_head[0].normalized_shape == (config.hidden_size,)

    def test_progress_only_training_loss_excludes_action_loss(self):
        config = _small_config(
            enable_progress_head=True,
            progress_loss_weight=0.2,
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
        )
        head = Gr00tN1d7ActionHead(config)
        head.train()
        out = head.forward(_make_backbone_output(config), _make_action_input(config))

        assert torch.allclose(out["loss"], out["progress_loss"] * config.progress_loss_weight)

    def test_progress_token_features_are_bounded(self):
        config = _small_config(enable_progress_head=True, add_pos_embed=False)
        head = Gr00tN1d7ActionHead(config)
        head.progress_token.data.fill_(1e6)

        features = head._make_progress_features(
            batch_size=2,
            position_index=0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        assert features.abs().max() <= head.progress_token_scale

    def test_progress_head_outputs_sigmoid_value(self):
        config = _small_config(enable_progress_head=True)
        head = Gr00tN1d7ActionHead(config)
        progress_hidden = torch.randn(2, config.hidden_size)

        with torch.no_grad():
            head.progress_head[3].weight.zero_()
            head.progress_head[3].bias.fill_(2.0)

        progress_pred = head._predict_progress(progress_hidden)
        assert torch.allclose(
            progress_pred,
            torch.full_like(progress_pred, torch.sigmoid(torch.tensor(2.0))),
        )
        assert torch.all((progress_pred >= 0.0) & (progress_pred <= 1.0))

    def test_progress_token_is_appended_after_action_tokens(self):
        class ConstantStateEncoder(torch.nn.Module):
            def forward(self, state, embodiment_id):
                return torch.ones(
                    state.shape[0],
                    1,
                    config.input_embedding_dim,
                    device=state.device,
                    dtype=state.dtype,
                )

        class ConstantActionEncoder(torch.nn.Module):
            def forward(self, action, timestep, embodiment_id):
                return torch.full(
                    (action.shape[0], config.action_horizon, config.input_embedding_dim),
                    2.0,
                    device=action.device,
                    dtype=action.dtype,
                )

        class IdentityBackboneAttention(torch.nn.Module):
            def forward(self, backbone_features):
                return backbone_features

        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_states = None
                self.attention_mask = None

            def forward(self, hidden_states, **kwargs):
                self.hidden_states = hidden_states.detach().clone()
                self.attention_mask = kwargs.get("attention_mask")
                return hidden_states, None

        class CaptureActionDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_states = None

            def forward(self, hidden_states, embodiment_id):
                self.hidden_states = hidden_states.detach().clone()
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    config.max_action_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        config = _small_config(enable_progress_head=True, add_pos_embed=False)
        head = Gr00tN1d7ActionHead(config)
        head.state_encoder = ConstantStateEncoder()
        head.action_encoder = ConstantActionEncoder()
        head.vl_self_attention = IdentityBackboneAttention()
        head.model = CaptureModel()
        head.action_decoder = CaptureActionDecoder()

        head.forward(_make_backbone_output(config), _make_action_input(config))

        hidden_states = head.model.hidden_states
        assert hidden_states.shape[1] == 1 + config.action_horizon + 1
        assert torch.allclose(hidden_states[:, :1], torch.ones_like(hidden_states[:, :1]))
        assert torch.allclose(
            hidden_states[:, 1 : 1 + config.action_horizon],
            torch.full_like(hidden_states[:, 1 : 1 + config.action_horizon], 2.0),
        )
        assert torch.allclose(
            hidden_states[:, -1:],
            head._make_progress_features(
                batch_size=hidden_states.shape[0],
                position_index=config.action_horizon,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            ),
        )
        assert torch.allclose(
            head.action_decoder.hidden_states,
            hidden_states[:, 1 : 1 + config.action_horizon],
        )

        attention_mask = head.model.attention_mask
        assert attention_mask is None

    def test_vlm_progress_head_does_not_append_progress_to_action_tokens(self):
        class CaptureBackboneAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_states = []

            def forward(self, backbone_features):
                self.hidden_states.append(backbone_features.detach().clone())
                return backbone_features

        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_states = None

            def forward(self, hidden_states, **kwargs):
                self.hidden_states = hidden_states.detach().clone()
                return hidden_states, None

        class CaptureActionDecoder(torch.nn.Module):
            def forward(self, hidden_states, embodiment_id):
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    config.max_action_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm",
            add_pos_embed=False,
        )
        head = Gr00tN1d7ActionHead(config)
        backbone_attention = CaptureBackboneAttention()
        head.vl_self_attention = backbone_attention
        head.model = CaptureModel()
        head.action_decoder = CaptureActionDecoder()

        head.forward(_make_backbone_output(config), _make_action_input(config))

        assert len(backbone_attention.hidden_states) == 2
        assert backbone_attention.hidden_states[0].shape[1] == 9
        assert backbone_attention.hidden_states[1].shape[1] == 8
        assert head.model.hidden_states.shape[1] == 1 + config.action_horizon

    def test_vlm_dit_progress_head_uses_separate_progress_route(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_states = []
                self.encoder_hidden_states = []

            def forward(self, hidden_states, **kwargs):
                self.hidden_states.append(hidden_states.detach().clone())
                self.encoder_hidden_states.append(kwargs["encoder_hidden_states"].detach().clone())
                if kwargs.get("return_all_hidden_states"):
                    return hidden_states, None
                return hidden_states

        class CaptureActionDecoder(torch.nn.Module):
            def forward(self, hidden_states, embodiment_id):
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    config.max_action_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_dit",
            add_pos_embed=False,
        )
        head = Gr00tN1d7ActionHead(config)
        head.model = CaptureModel()
        head.action_decoder = CaptureActionDecoder()

        head.forward(
            _make_backbone_output(config),
            _make_action_input(config),
            progress_backbone_output=_make_progress_backbone_output(config),
        )

        assert head.model.hidden_states[0].shape[1] == 1 + config.action_horizon
        assert head.model.hidden_states[1].shape[1] == 2
        assert head.model.encoder_hidden_states[0].shape[1] == 8
        assert head.model.encoder_hidden_states[1].shape[1] == 9

    def test_vlm_pooled_dit_progress_head_uses_pooled_progress_route(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.hidden_states = []
                self.encoder_hidden_states = []

            def forward(self, hidden_states, **kwargs):
                self.hidden_states.append(hidden_states.detach().clone())
                self.encoder_hidden_states.append(kwargs["encoder_hidden_states"].detach().clone())
                if kwargs.get("return_all_hidden_states"):
                    all_hidden_states = [
                        hidden_states + float(layer_idx) for layer_idx in range(self.num_layers + 1)
                    ]
                    return hidden_states, all_hidden_states
                return hidden_states

        class CaptureActionDecoder(torch.nn.Module):
            def forward(self, hidden_states, embodiment_id):
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    config.max_action_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_pooled_dit",
            add_pos_embed=False,
        )
        head = Gr00tN1d7ActionHead(config)
        head.model = CaptureModel(config.diffusion_model_cfg["num_layers"])
        head.action_decoder = CaptureActionDecoder()

        head.forward(_make_backbone_output(config), _make_action_input(config))

        assert head.model.hidden_states[0].shape[1] == 2
        assert head.model.hidden_states[1].shape[1] == 1 + config.action_horizon
        assert head.model.encoder_hidden_states[0].shape[1] == 8
        assert head.model.encoder_hidden_states[1].shape[1] == 8

    def test_state_multilayer_dit_progress_head_uses_state_progress_route(self):
        class CaptureModel(torch.nn.Module):
            def __init__(self, num_layers):
                super().__init__()
                self.num_layers = num_layers
                self.hidden_states = []
                self.encoder_hidden_states = []

            def forward(self, hidden_states, **kwargs):
                self.hidden_states.append(hidden_states.detach().clone())
                self.encoder_hidden_states.append(kwargs["encoder_hidden_states"].detach().clone())
                if kwargs.get("return_all_hidden_states"):
                    all_hidden_states = [
                        hidden_states + float(layer_idx) for layer_idx in range(self.num_layers + 1)
                    ]
                    return hidden_states, all_hidden_states
                return hidden_states

        class CaptureActionDecoder(torch.nn.Module):
            def forward(self, hidden_states, embodiment_id):
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    config.max_action_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        config = _small_config(
            enable_progress_head=True,
            progress_head_source="state_multilayer_dit",
            add_pos_embed=False,
        )
        head = Gr00tN1d7ActionHead(config)
        head.model = CaptureModel(config.diffusion_model_cfg["num_layers"])
        head.action_decoder = CaptureActionDecoder()

        head.forward(_make_backbone_output(config), _make_action_input(config))

        assert head.model.hidden_states[0].shape[1] == 2
        assert head.model.hidden_states[1].shape[1] == 1 + config.action_horizon
        assert head.model.encoder_hidden_states[0].shape[1] == 8
        assert head.model.encoder_hidden_states[1].shape[1] == 8

    @pytest.mark.parametrize(
        "source",
        [
            "vlm_pooled",
            "vlm_pooled_state",
            "vlm_concat_linear",
            "vlm_concat_projected_linear",
            "vlm_concat_attention_pool",
            "vlm_layer_pooled",
            "vlm_layer_concat_linear",
            "vlm_layer_concat_projected_linear",
            "vlm_layer_concat_attention_pool",
        ],
    )
    def test_vlm_pooled_progress_head_does_not_add_action_token(self, source):
        class CaptureModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.hidden_states = []

            def forward(self, hidden_states, **kwargs):
                self.hidden_states.append(hidden_states.detach().clone())
                if kwargs.get("return_all_hidden_states"):
                    return hidden_states, None
                return hidden_states

        class CaptureActionDecoder(torch.nn.Module):
            def forward(self, hidden_states, embodiment_id):
                return torch.zeros(
                    hidden_states.shape[0],
                    hidden_states.shape[1],
                    config.max_action_dim,
                    device=hidden_states.device,
                    dtype=hidden_states.dtype,
                )

        config = _small_config(
            enable_progress_head=True,
            progress_head_source=source,
            progress_vlm_layer=1,
            add_pos_embed=False,
        )
        head = Gr00tN1d7ActionHead(config)
        head.model = CaptureModel()
        head.action_decoder = CaptureActionDecoder()

        head.forward(
            _make_backbone_output(config, num_hidden_states=2),
            _make_action_input(config),
        )

        assert len(head.model.hidden_states) == 1
        assert head.model.hidden_states[0].shape[1] == 1 + config.action_horizon


class TestActionHeadGetAction:
    """Test inference denoising loop."""

    def test_get_action_output_shape(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert "action_pred" in out
        assert out["action_pred"].shape == (2, config.action_horizon, config.max_action_dim)

    def test_get_action_no_grad(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config)
        del action_input["action"]
        out = head.get_action(_make_backbone_output(config), action_input)
        assert not out["action_pred"].requires_grad

    def test_get_action_single_sample(self, action_head):
        head, config = action_head
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["action_pred"].shape[0] == 1

    def test_get_action_with_progress_head(self):
        config = _small_config(enable_progress_head=True)
        head = Gr00tN1d7ActionHead(config)
        head.eval()
        action_input = _make_action_input(config, batch_size=1)
        del action_input["action"]
        out = head.get_action(
            _make_backbone_output(config, batch_size=1),
            action_input,
        )
        assert out["progress_pred"].shape == (1,)
        assert torch.all((out["progress_pred"] >= 0.0) & (out["progress_pred"] <= 1.0))


class TestActionHeadEncodeFeatures:
    """Test feature encoding helper."""

    def test_encode_features_shapes(self, action_head):
        head, config = action_head
        result = head._encode_features(
            _make_backbone_output(config),
            _make_action_input(config),
        )
        assert result["backbone_features"].shape == (2, 8, config.backbone_embedding_dim)
        assert result["state_features"].shape == (2, 1, config.input_embedding_dim)


class TestActionHeadTrainableParams:
    """Test parameter freezing."""

    def test_all_trainable_by_default(self, action_head):
        head, _ = action_head
        head.set_trainable_parameters(True, True, True)
        assert all(p.requires_grad for p in head.parameters())

    def test_freeze_projector(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(False, True, True)
        for p in head.state_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_encoder.parameters():
            assert not p.requires_grad
        for p in head.action_decoder.parameters():
            assert not p.requires_grad

    def test_freeze_diffusion(self):
        config = _small_config()
        head = Gr00tN1d7ActionHead(config)
        head.set_trainable_parameters(True, False, True)
        for p in head.model.parameters():
            assert not p.requires_grad

    def test_progress_only_leaves_only_progress_params_trainable(self):
        config = _small_config(
            enable_progress_head=True,
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
            tune_progress_head=True,
        )
        head = Gr00tN1d7ActionHead(config)
        trainable = {name for name, p in head.named_parameters() if p.requires_grad}

        assert trainable
        assert all(
            name.startswith("progress_token") or name.startswith("progress_head")
            for name in trainable
        )

    def test_vlm_dit_progress_only_leaves_only_progress_params_trainable(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_dit",
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
            tune_progress_head=True,
        )
        head = Gr00tN1d7ActionHead(config)
        trainable = {name for name, p in head.named_parameters() if p.requires_grad}

        assert trainable
        assert all(
            name.startswith("progress_token")
            or name.startswith("progress_head")
            or name.startswith("progress_vlm_projector")
            for name in trainable
        )

    @pytest.mark.parametrize(
        "source",
        [
            "vlm_pooled",
            "vlm_pooled_state",
            "vlm_concat_linear",
            "vlm_concat_projected_linear",
            "vlm_concat_attention_pool",
            "vlm_layer_pooled",
            "vlm_layer_concat_linear",
            "vlm_layer_concat_projected_linear",
            "vlm_layer_concat_attention_pool",
        ],
    )
    def test_vlm_pooled_progress_only_leaves_only_progress_head_trainable(self, source):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source=source,
            progress_vlm_layer=1,
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
            tune_progress_head=True,
        )
        head = Gr00tN1d7ActionHead(config)
        trainable = {name for name, p in head.named_parameters() if p.requires_grad}

        assert trainable
        assert all(
            name.startswith("progress_head")
            or name.startswith("progress_vlm_token_norm")
            or name.startswith("progress_vlm_token_projector")
            or name.startswith("progress_vlm_token_attention")
            for name in trainable
        )

    def test_vlm_pooled_dit_progress_only_leaves_only_progress_params_trainable(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="vlm_pooled_dit",
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
            tune_progress_head=True,
        )
        head = Gr00tN1d7ActionHead(config)
        trainable = {name for name, p in head.named_parameters() if p.requires_grad}

        assert trainable
        assert all(
            name.startswith("progress_token") or name.startswith("progress_head")
            for name in trainable
        )

    def test_state_multilayer_dit_progress_only_leaves_only_progress_params_trainable(self):
        config = _small_config(
            enable_progress_head=True,
            progress_head_source="state_multilayer_dit",
            tune_projector=False,
            tune_diffusion_model=False,
            tune_vlln=False,
            tune_progress_head=True,
        )
        head = Gr00tN1d7ActionHead(config)
        trainable = {name for name, p in head.named_parameters() if p.requires_grad}

        assert trainable
        assert all(
            name.startswith("progress_token") or name.startswith("progress_head")
            for name in trainable
        )

    def test_freeze_progress_head(self):
        config = _small_config(enable_progress_head=True, tune_progress_head=False)
        head = Gr00tN1d7ActionHead(config)
        assert not head.progress_token.requires_grad
        assert not any(p.requires_grad for p in head.progress_head.parameters())
