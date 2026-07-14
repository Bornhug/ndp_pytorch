from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from ..config import FlowNPConfig


def flow_position_encoding(positions: Tensor, dim_posenc: int) -> Tensor:
    """Official FlowNP sinusoidal encoding of coordinates and flow time."""
    if positions.ndim != 3:
        raise ValueError("positions must have shape [B, N, D]")
    if dim_posenc <= 0 or dim_posenc % 2 != 0:
        raise ValueError("dim_posenc must be a positive even integer")
    half = dim_posenc // 2
    frequencies = math.pi * torch.pow(
        torch.tensor(2.0, device=positions.device, dtype=positions.dtype),
        torch.arange(half, device=positions.device, dtype=positions.dtype) - 2.0,
    )
    angles = positions.unsqueeze(-1) * frequencies
    encoded = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    return encoded.flatten(start_dim=2)


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    depth: int,
) -> nn.Sequential:
    if depth < 2:
        raise ValueError("MLP depth must be at least two")
    modules: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
    ]
    for _ in range(depth - 2):
        modules.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)])
    modules.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*modules)


class FlowNPModel(nn.Module):
    """Permutation-equivariant transformer velocity field for regression tasks."""

    def __init__(self, input_dim: int, config: FlowNPConfig):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(config.output_dim)
        self.dim_posenc = int(config.dim_posenc)
        token_input_dim = (self.input_dim + 1) * self.dim_posenc + self.output_dim
        self.embedder = build_mlp(
            token_input_dim,
            config.hidden_dim,
            config.hidden_dim,
            config.embedding_depth,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers,
            enable_nested_tensor=False,
        )
        self.predictor = nn.Sequential(
            nn.Linear(config.hidden_dim, config.predictor_hidden_dim),
            nn.ReLU(),
            nn.Linear(config.predictor_hidden_dim, self.output_dim),
        )

    @staticmethod
    def _batchify(value: Tensor, dimensions: int) -> Tensor:
        if value.ndim == dimensions - 1:
            return value.unsqueeze(0)
        if value.ndim != dimensions:
            raise ValueError(
                f"Expected a {dimensions - 1}D or {dimensions}D tensor, "
                f"got shape {tuple(value.shape)}"
            )
        return value

    @staticmethod
    def _expand_time(
        time: Tensor | float,
        *,
        batch_size: int,
        num_points: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        value = torch.as_tensor(time, device=device, dtype=dtype)
        if value.ndim == 0:
            value = value.view(1, 1, 1)
        elif value.ndim == 1:
            if value.numel() == batch_size:
                value = value.view(batch_size, 1, 1)
            elif batch_size == 1 and value.numel() == num_points:
                value = value.view(1, num_points, 1)
            else:
                raise ValueError("1D time must have B or N values")
        elif value.ndim == 2:
            if value.shape == (batch_size, num_points):
                value = value.unsqueeze(-1)
            elif value.shape == (batch_size, 1):
                value = value.unsqueeze(-1)
            else:
                raise ValueError("2D time must have shape [B, N] or [B, 1]")
        elif value.ndim != 3:
            raise ValueError("time must be scalar or have shape [B], [B,N], or [B,N,1]")
        if value.shape[0] not in {1, batch_size}:
            raise ValueError("time batch dimension is incompatible with targets")
        if value.shape[1] not in {1, num_points}:
            raise ValueError("time point dimension is incompatible with targets")
        if value.shape[-1] != 1:
            raise ValueError("time must have a singleton final dimension")
        return value.expand(batch_size, num_points, 1)

    def forward(
        self,
        x_target: Tensor,
        y_target: Tensor,
        time: Tensor | float,
        mask_target: Tensor | None = None,
        *,
        x_context: Tensor | None = None,
        y_context: Tensor | None = None,
        mask_context: Tensor | None = None,
    ) -> Tensor:
        x_target = self._batchify(x_target, 3)
        y_target = self._batchify(y_target, 3)
        batch_size, num_target, input_dim = x_target.shape
        if input_dim != self.input_dim:
            raise ValueError(f"Expected input_dim={self.input_dim}, got {input_dim}")
        if y_target.shape != (batch_size, num_target, self.output_dim):
            raise ValueError("y_target has an incompatible shape")

        target_time = self._expand_time(
            time,
            batch_size=batch_size,
            num_points=num_target,
            device=x_target.device,
            dtype=x_target.dtype,
        )
        if mask_target is None:
            mask_target = torch.zeros(
                (batch_size, num_target),
                dtype=torch.bool,
                device=x_target.device,
            )
        else:
            mask_target = self._batchify(mask_target, 2).to(
                device=x_target.device, dtype=torch.bool
            )

        if x_context is None or y_context is None:
            x_context = x_target.new_empty((batch_size, 0, self.input_dim))
            y_context = y_target.new_empty((batch_size, 0, self.output_dim))
        else:
            x_context = self._batchify(x_context, 3).to(x_target.device)
            y_context = self._batchify(y_context, 3).to(y_target.device)
        num_context = x_context.shape[1]
        if x_context.shape != (batch_size, num_context, self.input_dim):
            raise ValueError("x_context has an incompatible shape")
        if y_context.shape != (batch_size, num_context, self.output_dim):
            raise ValueError("y_context has an incompatible shape")
        if mask_context is None:
            mask_context = torch.zeros(
                (batch_size, num_context),
                dtype=torch.bool,
                device=x_target.device,
            )
        else:
            mask_context = self._batchify(mask_context, 2).to(
                device=x_target.device, dtype=torch.bool
            )

        context_time = x_context.new_ones((batch_size, num_context, 1))
        context_position = flow_position_encoding(
            torch.cat([x_context, context_time], dim=-1), self.dim_posenc
        )
        target_position = flow_position_encoding(
            torch.cat([x_target, target_time], dim=-1), self.dim_posenc
        )
        context_tokens = torch.cat([context_position, y_context], dim=-1)
        target_tokens = torch.cat([target_position, y_target], dim=-1)
        tokens = torch.cat([context_tokens, target_tokens], dim=1)
        padding_mask = torch.cat([mask_context, mask_target], dim=1)

        encoded = self.encoder(self.embedder(tokens), src_key_padding_mask=padding_mask)
        target_encoded = encoded[:, num_context:]
        velocity = self.predictor(target_encoded)
        return velocity.masked_fill(mask_target.unsqueeze(-1), 0.0)
