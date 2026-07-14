from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def build_timestep_embedding_table(
    num_timesteps: int,
    embedding_dim: int,
    max_positions: int = 10_000,
) -> Tensor:
    if num_timesteps <= 0 or embedding_dim <= 1:
        raise ValueError("num_timesteps must be positive and embedding_dim > 1")
    timesteps = torch.arange(num_timesteps, dtype=torch.float32)
    half_dim = embedding_dim // 2
    scale = math.log(max_positions) / max(half_dim - 1, 1)
    frequencies = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -scale)
    angles = timesteps[:, None] * frequencies[None, :]
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    return F.pad(embedding, (0, embedding_dim - embedding.shape[1]))


class MultiHeadAttention(nn.Module):
    """Packed PyTorch attention with the original NDP tensor conventions."""

    def __init__(self, d_model: int, num_heads: int, sparse: bool = False) -> None:
        del sparse
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(
            d_model,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )

    def forward(
        self,
        value: Tensor,
        key: Tensor,
        query: Tensor,
        *,
        key_padding_mask: Tensor | None = None,
    ) -> Tensor:
        leading_shape = query.shape[:-2]
        query_length, embedding_dim = query.shape[-2:]
        key_length = key.shape[-2]
        value_length = value.shape[-2]
        query_flat = query.reshape(-1, query_length, embedding_dim)
        key_flat = key.reshape(-1, key_length, embedding_dim)
        value_flat = value.reshape(-1, value_length, embedding_dim)

        mask_flat = None
        all_masked = None
        if key_padding_mask is not None:
            mask = key_padding_mask.to(device=query.device, dtype=torch.bool)
            if mask.ndim != 2 or mask.shape[1] != key_length:
                raise ValueError("key_padding_mask must have shape [B, sequence]")
            repeat = query_flat.shape[0] // mask.shape[0]
            mask_flat = mask.repeat_interleave(repeat, dim=0)
            all_masked = mask_flat.all(dim=1)
            if all_masked.any():
                mask_flat = mask_flat.clone()
                mask_flat[all_masked, 0] = False

        output, _ = self.attention(
            query_flat,
            key_flat,
            value_flat,
            key_padding_mask=mask_flat,
            need_weights=False,
        )
        if all_masked is not None and all_masked.any():
            output[all_masked] = 0.0
        return output.reshape(*leading_shape, query_length, embedding_dim)


class BiDimensionalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads)
        self.mha_n = MultiHeadAttention(2 * hidden_dim, num_heads)

    def forward(
        self,
        state: Tensor,
        time_embedding: Tensor,
        mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        time_embedding = self.linear_t(time_embedding)[:, None, None, :]
        attention_input = torch.cat(
            [state + time_embedding, state + time_embedding], dim=-1
        )
        feature_attention = self.mha_d(
            attention_input, attention_input, attention_input
        )
        point_input = attention_input.transpose(1, 2)
        point_attention = self.mha_n(
            point_input,
            point_input,
            point_input,
            key_padding_mask=mask,
        ).transpose(1, 2)
        residual, skip = torch.chunk(feature_attention + point_attention, 2, dim=-1)
        residual = F.gelu(residual)
        skip = F.gelu(skip)
        if mask is not None:
            valid = (~mask.to(dtype=torch.bool))[:, :, None, None]
            residual = residual * valid
            skip = skip * valid
        return (state + residual) / math.sqrt(2.0), skip


class _NDPAttentionBase(nn.Module):
    backend_name: str

    def __init__(
        self,
        n_layers: int,
        hidden_dim: int,
        num_heads: int,
        num_timesteps: int = 500,
        init_zero: bool = True,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_timesteps = num_timesteps
        self.init_zero = init_zero
        self.input_linear = nn.Linear(2, hidden_dim)
        self.register_buffer(
            "timestep_embeddings",
            build_timestep_embedding_table(num_timesteps, hidden_dim),
            persistent=False,
        )
        self.layers = nn.ModuleList(
            [BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.proj_eps = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    @staticmethod
    def _tokens(x: Tensor, y: Tensor) -> Tensor:
        if x.ndim != 3 or y.ndim != 3:
            raise ValueError("NDP inputs must have shapes [B,N,D] and [B,N,1]")
        if x.shape[:2] != y.shape[:2] or y.shape[-1] != 1:
            raise ValueError("NDP x/y shapes are incompatible")
        expanded_y = y.unsqueeze(2).expand(-1, -1, x.shape[2], -1)
        return torch.cat([x.unsqueeze(-1), expanded_y], dim=-1)

    def _predict(
        self,
        x: Tensor,
        y: Tensor,
        time: Tensor,
        mask: Tensor | None,
    ) -> Tensor:
        batch_size, num_points = x.shape[:2]
        if mask is None:
            mask = torch.zeros(
                (batch_size, num_points), dtype=torch.bool, device=x.device
            )
        else:
            mask = mask.to(device=x.device, dtype=torch.bool)
            if mask.shape != (batch_size, num_points):
                raise ValueError("NDP mask must have shape [B,N]")
        time = torch.as_tensor(time, device=x.device).reshape(-1)
        if time.numel() == 1 and batch_size > 1:
            time = time.expand(batch_size)
        if time.numel() != batch_size:
            raise ValueError("NDP requires one timestep per function")
        time = time.long().clamp(0, self.num_timesteps - 1)

        state = F.gelu(self.input_linear(self._tokens(x, y)))
        time_embedding = self.timestep_embeddings[time]
        skip_sum = None
        for layer in self.layers:
            state, skip = layer(state, time_embedding, mask)
            skip_sum = skip if skip_sum is None else skip_sum + skip
        assert skip_sum is not None
        skip_sum = skip_sum.mean(dim=2) / math.sqrt(self.n_layers)
        prediction = self.output_linear(F.gelu(self.proj_eps(skip_sum)))
        return prediction.masked_fill(mask.unsqueeze(-1), 0.0)


class ConditionalNDPModel(_NDPAttentionBase):
    backend_name = "ndp_cond"

    def forward(
        self,
        x_target: Tensor,
        y_target: Tensor,
        t: Tensor,
        mask_target: Tensor | None = None,
        *,
        x_context: Tensor | None = None,
        y_context: Tensor | None = None,
        mask_context: Tensor | None = None,
    ) -> Tensor:
        if (x_context is None) != (y_context is None):
            raise ValueError("x_context and y_context must be supplied together")
        target_count = x_target.shape[1]
        if x_context is None or x_context.shape[1] == 0:
            return self._predict(x_target, y_target, t, mask_target)
        if x_context.shape[0] != x_target.shape[0]:
            raise ValueError("context and target batch sizes must match")
        target_mask = (
            torch.zeros(
                x_target.shape[:2], dtype=torch.bool, device=x_target.device
            )
            if mask_target is None
            else mask_target.to(device=x_target.device, dtype=torch.bool)
        )
        context_mask = (
            torch.zeros(
                x_context.shape[:2], dtype=torch.bool, device=x_target.device
            )
            if mask_context is None
            else mask_context.to(device=x_target.device, dtype=torch.bool)
        )
        prediction = self._predict(
            torch.cat([x_context, x_target], dim=1),
            torch.cat([y_context, y_target], dim=1),
            t,
            torch.cat([context_mask, target_mask], dim=1),
        )
        return prediction[:, -target_count:]


class JointNDPModel(_NDPAttentionBase):
    backend_name = "ndp_uncond"

    def forward(
        self,
        x_target: Tensor,
        y_target: Tensor,
        t: Tensor,
        mask_target: Tensor | None = None,
        *,
        x_context: Tensor | None = None,
        y_context: Tensor | None = None,
        mask_context: Tensor | None = None,
    ) -> Tensor:
        if any(value is not None for value in (x_context, y_context, mask_context)):
            raise ValueError("JointNDPModel does not accept a clean-context channel")
        return self._predict(x_target, y_target, t, mask_target)


def convert_original_attention_state_dict(
    state: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    """Convert original NDP q/k/v attention weights to packed PyTorch MHA."""
    normalized = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }
    if any("mha_d_self" in key or "mha_n_self" in key for key in normalized):
        raise ValueError(
            "Cross-attention NDP checkpoints are incompatible with the "
            "original-compatible ndp_cond/ndp_uncond architecture"
        )
    q_weight_keys = [key for key in normalized if key.endswith(".wq.weight")]
    if not q_weight_keys:
        return dict(normalized)

    converted: dict[str, Tensor] = {}
    projection_tokens = (".wq.", ".wk.", ".wv.", ".dense.")
    for key, value in normalized.items():
        if not any(token in key for token in projection_tokens):
            converted[key] = value

    for q_weight_key in q_weight_keys:
        prefix = q_weight_key[: -len("wq.weight")]
        q_bias_key = prefix + "wq.bias"
        k_weight_key = prefix + "wk.weight"
        k_bias_key = prefix + "wk.bias"
        v_weight_key = prefix + "wv.weight"
        v_bias_key = prefix + "wv.bias"
        dense_weight_key = prefix + "dense.weight"
        dense_bias_key = prefix + "dense.bias"
        required = (
            q_bias_key,
            k_weight_key,
            k_bias_key,
            v_weight_key,
            v_bias_key,
            dense_weight_key,
            dense_bias_key,
        )
        missing = [key for key in required if key not in normalized]
        if missing:
            raise ValueError(
                "Original NDP checkpoint has incomplete attention tensors: "
                + ", ".join(missing)
            )
        packed_prefix = prefix + "attention."
        converted[packed_prefix + "in_proj_weight"] = torch.cat(
            [normalized[q_weight_key], normalized[k_weight_key], normalized[v_weight_key]],
            dim=0,
        )
        converted[packed_prefix + "in_proj_bias"] = torch.cat(
            [normalized[q_bias_key], normalized[k_bias_key], normalized[v_bias_key]],
            dim=0,
        )
        converted[packed_prefix + "out_proj.weight"] = normalized[dense_weight_key]
        converted[packed_prefix + "out_proj.bias"] = normalized[dense_bias_key]
    return converted


__all__ = [
    "BiDimensionalAttentionBlock",
    "ConditionalNDPModel",
    "JointNDPModel",
    "MultiHeadAttention",
    "build_timestep_embedding_table",
    "convert_original_attention_state_dict",
]

