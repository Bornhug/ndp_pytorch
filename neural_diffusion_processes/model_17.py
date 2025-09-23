# model.py
# Bi-Dimensional Attention with separate encoders (target/context) and a cross-decoder.
# - TargetEncoder / ContextEncoder: self-attn on D and on sequence axis (N or M).
# - CrossDecoder: cross-attn on BOTH axes (N and D), with robust have_ctx=False fallback.
# - Model API is unchanged from your original.

from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------- utils -----------------------------

def _b1(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 1:
        t = t[:, None]
    return t

def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal embedding -> [B, dim]."""
    t = _b1(t)                                  # [B,1]
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.linspace(0, 1, half, device=device) * math.log(10000.0))
    ang = t / freqs                             # [B,half]
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb                                  # [B,dim]

def process_inputs(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    x: [B,N,D], y: [B,N,1]  ->  tokens [B,N,D,2]
    """
    B, N, D = x.shape
    x4 = x.unsqueeze(-1)                         # [B,N,D,1]
    y4 = y.unsqueeze(2).expand(B, N, D, 1)       # [B,N,D,1]
    return torch.cat([x4, y4], dim=-1)           # [B,N,D,2]

def _to_bool(mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    return mask.bool() if mask.dtype != torch.bool else mask


# ------------------------- attention core -------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting query/key/value shaped either:
      - [B, L, E]            (A=1 implicit)
      - [B, A, L, E]         (A = groups; e.g., A=D for N-attn, A=N for D-attn)

    Mask may be:
      - key padding: [B, A, Lk]        (True = masked)
      - pairwise:    [B, A, Lq, Lk]    (True = masked)

    Returns the same rank as query.
    """
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = True):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _to4d(self, x: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if x.ndim == 3:         # [B,L,E] -> [B,1,L,E]
            return x.unsqueeze(1), True
        elif x.ndim == 4:       # [B,A,L,E]
            return x, False
        else:
            raise ValueError(f"Expected 3D/4D, got {x.shape}")

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q4, squeezed = self._to4d(q)  # [B,A,Lq,E]
        k4, _ = self._to4d(k)         # [B,A,Lk,E]
        v4, _ = self._to4d(v)         # [B,A,Lk,E]

        B, A, Lq, E = q4.shape
        H = self.num_heads
        Dh = self.head_dim

        def proj(x, layer):
            x = layer(x)
            x = x.view(B, A, -1, H, Dh).transpose(2, 3)  # [B,A,H,L,Dh]
            return x

        Q = proj(q4, self.q_proj)
        K = proj(k4, self.k_proj)
        V = proj(v4, self.v_proj)

        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Dh)  # [B,A,H,Lq,Lk]

        if mask is not None:
            m = mask
            if m.ndim == 3:      # [B,A,Lk]
                m = m[:, :, None, None, :]         # -> [B,A,1,1,Lk]
            elif m.ndim == 4:    # [B,A,Lq,Lk]
                m = m[:, :, None, :, :]            # -> [B,A,1,Lq,Lk]
            else:
                raise ValueError("mask must be [B,A,Lk] or [B,A,Lq,Lk]")

            m = m.bool()
            attn = attn.masked_fill(m, -1e9)       # avoid -inf to prevent NaNs

            all_masked = m.all(dim=-1, keepdim=True)  # [B,A,1,Lq,1] or [B,A,1,1,1]
            P = torch.softmax(attn, dim=-1)
            P = torch.where(all_masked, torch.zeros_like(P), P)
        else:
            P = torch.softmax(attn, dim=-1)

        out = torch.matmul(P, V)  # [B,A,H,Lq,Dh]
        out = out.transpose(2, 3).contiguous().view(B, A, Lq, E)  # [B,A,Lq,E]
        out = self.o_proj(out)
        return out.squeeze(1) if squeezed else out


# ----------------------------- encoders -----------------------------

class StreamEncoder(nn.Module):
    """
    Shared structure for both target/context encoders.
    Input:  s  [B,L,D,H], optional t_bias [B,H], mask [B,L] (True=pad)
    Output: yf [B,L,D,2H]  (fused features on both axes)
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        H2 = 2 * hidden_dim
        self.mha_d = MultiHeadAttention(H2, num_heads)  # self over D (A=L, L=D)
        self.mha_n = MultiHeadAttention(H2, num_heads)  # self over L (A=D, L=L)

    def forward(
        self,
        s: torch.Tensor,                         # [B,L,D,H]
        t_bias: Optional[torch.Tensor],          # [B,H] (None for context)
        seq_mask: Optional[torch.Tensor],        # [B,L] (True=pad)
    ) -> torch.Tensor:                           # [B,L,D,2H]
        if t_bias is not None:
            y = s + t_bias[:, None, None, :]     # [B,L,D,H]
        else:
            y = s
        y = torch.cat([y, y], dim=-1)            # [B,L,D,2H]

        # D-axis self-attention (groups=A=L, length=D)
        y_d = self.mha_d(y, y, y)                # [B,L,D,2H]

        # L-axis self-attention (groups=A=D, length=L)
        y_r = y.transpose(1, 2)                  # [B,D,L,2H]
        m = seq_mask.unsqueeze(1) if seq_mask is not None else None  # [B,1,L]
        y_n = self.mha_n(y_r, y_r, y_r, m)       # [B,D,L,2H]
        y_n = y_n.transpose(1, 2)                # [B,L,D,2H]

        yf = F.gelu(y_d + y_n)                   # [B,L,D,2H]
        return yf


# ----------------------------- cross-decoder -----------------------------

class CrossDecoder(nn.Module):
    """
    Cross attention over N and over D.
    Inputs:
      tgt_f    : [B,N,D,2H] (from target encoder)
      mask     : [B,N] or None
      ctx_f    : [B,M,D,2H] or None
      mask_ctx : [B,M] or None
    Output:
      y        : [B,N,D,2H]
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        H2 = 2 * hidden_dim
        self.mha_crossN = MultiHeadAttention(H2, num_heads)  # cross over N (per-D)
        self.mha_crossD = MultiHeadAttention(H2, num_heads)  # cross over D (per-N)

        # safety projection if any accidental width mismatch appears
        self._align_ctxD: Optional[nn.Linear] = None

    def forward(
        self,
        tgt_f: torch.Tensor,                       # [B,N,D,2H]
        mask: Optional[torch.Tensor],              # [B,N]
        ctx_f: Optional[torch.Tensor],             # [B,M,D,2H] or None
        mask_ctx: Optional[torch.Tensor],          # [B,M] or None
    ) -> torch.Tensor:                             # [B,N,D,2H]
        B, N, D, E = tgt_f.shape
        device = tgt_f.device
        have_ctx = (ctx_f is not None) and (ctx_f.shape[1] > 0)

        mask = _to_bool(mask)
        mask_ctx = _to_bool(mask_ctx)

        # ===== cross-attention over N (per-D slice) =====
        qN = tgt_f.transpose(1, 2)                 # [B,D,N,2H]
        if have_ctx:
            kvN = ctx_f.transpose(1, 2)            # [B,D,M,2H]
            if (mask is not None) or (mask_ctx is not None):
                mq = mask if mask is not None else torch.zeros(B, N, dtype=torch.bool, device=device)
                mk = mask_ctx if mask_ctx is not None else torch.zeros(B, kvN.shape[2], dtype=torch.bool, device=device)
                mask_crossN = (mq[:, None, :, None] | mk[:, None, None, :]).expand(B, D, N, kvN.shape[2])  # [B,D,N,M]
            else:
                mask_crossN = None
            y_crossN = self.mha_crossN(qN, kvN, kvN, mask_crossN)        # [B,D,N,2H]
        else:
            # no context: identity passthrough on this branch
            y_crossN = qN
        y_crossN = y_crossN.transpose(1, 2)         # [B,N,D,2H]

        # ===== cross-attention over D (per-N slice) =====
        qD = tgt_f                                  # [B,N,D,2H]
        if have_ctx:
            # ctx_f is [B,M,D,2H]. Pool over M -> [B,D,2H]
            if mask_ctx is not None:
                mkeep = (~mask_ctx).float()[:, :, None, None]    # [B,M,1,1]
                denom = mkeep.sum(dim=1).clamp_min(1.0)          # [B,1,1]
                ctx_pooled = (ctx_f * mkeep).sum(dim=1) / denom  # [B,D,2H]
            else:
                ctx_pooled = ctx_f.mean(dim=1)                   # [B,D,2H]

            # width safety for pooled features
            if ctx_pooled.ndim != 3:
                raise RuntimeError(f"ctx_pooled rank {ctx_pooled.ndim} != 3")
            if ctx_pooled.shape[-1] != E:
                if ctx_pooled.shape[-1] % E == 0:
                    extra = ctx_pooled.shape[-1] // E
                    ctx_pooled = ctx_pooled.view(B, D, extra, E).mean(dim=2)  # -> [B,D,E]
                else:
                    if self._align_ctxD is None:
                        self._align_ctxD = nn.Linear(ctx_pooled.shape[-1], E, bias=False).to(ctx_pooled.device)
                    ctx_pooled = self._align_ctxD(ctx_pooled)  # -> [B,D,E]

            # tile across N to make kvD: [B,N,D,2H]
            kvD = ctx_pooled.unsqueeze(1).expand(B, N, D, E).contiguous()

            # mask out rows where targets are padded
            mask_crossD = mask[:, :, None, None].expand(B, N, D, D) if (mask is not None) else None

            y_crossD = self.mha_crossD(qD, kvD, kvD, mask_crossD)        # [B,N,D,2H]
        else:
            # no context: identity passthrough
            y_crossD = qD

        # fuse both cross results
        y = y_crossN + y_crossD                                           # [B,N,D,2H]
        return y


# ------------------------ bi-dimensional block ------------------------

class BiDimensionalAttentionBlock(nn.Module):
    """
    One layer with separate encoders and a cross-decoder.
    Inputs:
      s        : [B, N, D, H]         (targets, pre-projected)
      t_emb    : [B, H]               (time embedding)
      mask     : [B, N] or None
      s_ctx    : [B, M, D, H] or None (context, pre-projected)
      mask_ctx : [B, M] or None
    Output:
      s_out    : [B, N, D, H]
      skip     : [B, N, D, H]
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.enc_tgt = StreamEncoder(hidden_dim, num_heads)
        self.enc_ctx = StreamEncoder(hidden_dim, num_heads)
        self.dec     = CrossDecoder(hidden_dim, num_heads)
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        s: torch.Tensor,
        t_emb: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        s_ctx: Optional[torch.Tensor] = None,
        mask_ctx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        t_bias = self.linear_t(t_emb)                         # [B,H]
        mask     = _to_bool(mask)
        mask_ctx = _to_bool(mask_ctx)

        # encode target stream (uses time bias)
        tgt_f = self.enc_tgt(s, t_bias, mask)                 # [B,N,D,2H]

        # encode context stream (no time bias)
        have_ctx = (s_ctx is not None) and (s_ctx.shape[1] > 0)
        if have_ctx:
            ctx_f = self.enc_ctx(s_ctx, None, mask_ctx)       # [B,M,D,2H]
        else:
            ctx_f = None

        # cross decode
        y = self.dec(tgt_f, mask, ctx_f, mask_ctx)            # [B,N,D,2H]

        # split to residual/skip, GELU, and residual add on s
        residual, skip = torch.chunk(y, 2, dim=-1)            # each [B,N,D,H]
        residual = F.gelu(residual)
        skip     = F.gelu(skip)
        s_out = (s + residual) / math.sqrt(2.0)               # [B,N,D,H]
        return s_out, skip


# ----------------------------- full model -----------------------------

class BiDimensionalAttentionModel(nn.Module):
    """
    Context-conditioned NDP with bi-dimensional blocks and cross-attention on
    both axes. API matches the original model.

    Inputs:
      x: [B,N,D], y: [B,N,1], t: [B] or [B,1], mask: [B,N] (True/1 = padded)
      Optional context: x_context [B,M,D], y_context [B,M,1], mask_context [B,M]
    Output: eps_hat [B,N,1]
    """
    def __init__(self, n_layers: int, hidden_dim: int, num_heads: int, init_zero: bool = True):
        super().__init__()
        self.hidden_dim   = hidden_dim
        self.input_linear = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList(
            [BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.proj_eps     = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear= nn.Linear(hidden_dim, 1)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)

    def forward(
        self,
        x: torch.Tensor,                      # [B,N,D]
        y: torch.Tensor,                      # [B,N,1]  (noisy y_t)
        t: torch.Tensor,                      # [B] or [B,1]
        mask: Optional[torch.Tensor] = None,  # [B,N] (True/1 = padded)
        x_context: Optional[torch.Tensor] = None,
        y_context: Optional[torch.Tensor] = None,
        mask_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:                        # -> [B,N,1]

        # preprocess target stream
        s_xy = process_inputs(x, y)                                # [B,N,D,2]
        s = F.gelu(self.input_linear(s_xy))                        # [B,N,D,H]

        # preprocess context stream (optional)
        s_ctx = None
        if (x_context is not None) and (y_context is not None):
            c_xy = process_inputs(x_context, y_context)            # [B,M,D,2]
            s_ctx = F.gelu(self.input_linear(c_xy))                # [B,M,D,H]

        # time embedding
        t_emb = timestep_embedding(t, self.hidden_dim)             # [B,H]

        # bi-dimensional stack with skip accumulation
        skip_sum = None
        for layer in self.layers:
            s, skip = layer(s, t_emb, mask=mask, s_ctx=s_ctx, mask_ctx=mask_context)  # [B,N,D,H]
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        # head: add skip, reduce over D, project to epsilon
        s = s + (skip_sum if skip_sum is not None else 0)          # [B,N,D,H]
        s = s.sum(dim=2)                                           # [B,N,H]  (Σ over D)
        s = F.gelu(self.proj_eps(s))                               # [B,N,H]
        eps_hat = self.output_linear(s)                            # [B,N,1]
        return eps_hat
