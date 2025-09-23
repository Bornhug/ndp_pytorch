# model.py
# Bi-Dimensional Attention with context: self-attn on both streams, then
# cross-attention on BOTH axes (N and D). No gates.

from __future__ import annotations
import math
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
    y4 = y.unsqueeze(2).expand(B, N, D, 1)  # [B,N,D,1]
    return torch.cat([x4, y4], dim=-1)          # [B,N,D,2]

def _to_bool(mask: torch.Tensor | None) -> torch.Tensor | None:
    if mask is None:
        return None
    return mask.bool() if mask.dtype != torch.bool else mask


# ------------------------- attention core -------------------------

class MultiHeadAttention(nn.Module):
    """
    Multi-head attention supporting inputs shaped either:
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
            mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q4, squeezed = self._to4d(q)  # [B,A,Lq,E]
        k4, _ = self._to4d(k)  # [B,A,Lk,E]
        v4, _ = self._to4d(v)  # [B,A,Lk,E]

        # (Optional) debug guards
        assert q4.shape[-1] == self.q_proj.in_features
        assert k4.shape[-1] == self.k_proj.in_features
        assert v4.shape[-1] == self.v_proj.in_features

        B, A, Lq, E = q4.shape
        Lk = k4.shape[2]
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
            if m.ndim == 3:  # [B,A,Lk]
                m = m[:, :, None, None, :]  # -> [B,A,1,1,Lk]
            elif m.ndim == 4:  # [B,A,Lq,Lk]
                m = m[:, :, None, :, :]  # -> [B,A,1,Lq,Lk]
            else:
                raise ValueError("mask must be [B,A,Lk] or [B,A,Lq,Lk]")

            m = m.bool()
            # Use large negative instead of -inf to avoid NaNs; softmax subtracts max internally.
            attn = attn.masked_fill(m, -1e9)

            # Identify rows where every key is masked
            all_masked = m.all(dim=-1, keepdim=True)  # [B,A,1,Lq,1]

            P = torch.softmax(attn, dim=-1)  # [B,A,H,Lq,Lk]
            # Zero out those rows so the attention contributes nothing
            P = torch.where(all_masked, torch.zeros_like(P), P)
        else:
            P = torch.softmax(attn, dim=-1)

        out = torch.matmul(P, V)  # [B,A,H,Lq,Dh]
        out = out.transpose(2, 3).contiguous().view(B, A, Lq, E)  # [B,A,Lq,E]
        out = self.o_proj(out)

        return out.squeeze(1) if squeezed else out


# -------------------- bi-dimensional attention block --------------------

class BiDimensionalAttentionBlock(nn.Module):
    """
    NDP 2-axis block with context.
    s      : [B, N, D, H]  (targets)
    s_ctx  : [B, M, D, H]  (context)  -- optional
    mask   : [B, N]        (True/1 = masked)
    mask_ctx: [B, M]       (True/1 = masked)

    Pipeline per stream:
      (s + t, s) -> concat -> MHA_D -> ReLU
                            -> MHA_N -> ReLU
    Then:
      - Cross-attention along N: queries from target, keys/values from context.
      - Cross-attention along D: queries from target, keys/values from a context
        summary pooled over M (so Lk = D).
      - Fuse both cross results (+) and do residual/skip.
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        H2 = 2 * hidden_dim

        self.mha_d      = MultiHeadAttention(H2, num_heads)  # self over D
        self.mha_n      = MultiHeadAttention(H2, num_heads)  # self over N/M
        self.mha_crossN = MultiHeadAttention(H2, num_heads)  # cross over N (targets↔context)
        self.mha_crossD = MultiHeadAttention(H2, num_heads)  # cross over D (targets↔context)

        self.linear_t = nn.Linear(hidden_dim, hidden_dim)

    def _stream(self, s: torch.Tensor, t: torch.Tensor | None,
                n_mask: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # s: [B,N_or_M,D,H], t_bias: [B,1,1,H]
        if t is None:
            y_temp = s
        else:
            y_temp = s + t                                     # [B,N/M,D,H]

        y = torch.cat([y_temp, y_temp], dim=-1)  # [B,N,D,2H]
        # D-axis self-attention (A=N/M, L=D)
        y_d = self.mha_d(y, y, y)                        # [B,N/M,D,2H]
        #y_d = F.gelu(y_d)
        # N/M-axis self-attention (A=D, L=N/M)
        y_r = y.transpose(1, 2)                          # [B,D,N/M,2H]
        m = n_mask.unsqueeze(1) if n_mask is not None else None  # [B,1,N/M]
        y_n = self.mha_n(y_r, y_r, y_r, m)               # [B,D,N/M,2H]
        y_n = y_n.transpose(1, 2)                   # [B,N/M,D,2H]
        # fuse stream (no gates): add
        y_f = y_d + y_n                         # [B,N/M,D,2H]
        #residual, skip = torch.chunk(y, 2, dim=-1) # residual, skip : [B, N, D, hidden_dim]
        residual, skip = torch.chunk(y_f, 2, dim=-1)
        y_f = F.gelu(residual)
        # residual : [B, N, D, hidden_dim]

        y_f = (y_f  + y_temp) / math.sqrt(2.0)
        y_f = torch.cat([y_f, y_f], dim=-1)  # [B,N,D,2H]
        return y_f, y_d, y_n

    def forward(
        self,
        s: torch.Tensor,
        t: torch.Tensor,
        mask: torch.Tensor | None = None,
        s_ctx: torch.Tensor | None = None,
        mask_ctx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        t_bias = self.linear_t(t)[:, None, None, :]      # [B,1,1,H]
        B, N, D, H = s.shape

        mask     = _to_bool(mask)
        mask_ctx = _to_bool(mask_ctx)

        # ----- target and context streams -----
        tgt_f, tgt_d, tgt_n = self._stream(s, t_bias, mask)              # each [B,N,D,2H]
        have_ctx = (s_ctx is not None) and (s_ctx.shape[1] > 0)
        if have_ctx:
            ctx_f, ctx_d, ctx_n = self._stream(s=s_ctx, t=None, n_mask=mask_ctx)  # each [B,M,D,2H]

        # ===== cross-attention over N (per-D slice) =====
        # q: [B,D,N,2H], kv: [B,D,M,2H]
        qN  = tgt_f.transpose(1, 2)                                      # [B,D,N,2H]
        if have_ctx:
            kvN = ctx_f.transpose(1, 2)                                  # [B,D,M,2H]
            if (mask is not None) or (mask_ctx is not None):
                mq = mask if mask is not None else torch.zeros(B, N, dtype=torch.bool, device=s.device)
                mk = mask_ctx if mask_ctx is not None else torch.zeros(B, kvN.shape[2], dtype=torch.bool, device=s.device)
                #mask_crossN = mq.unsqueeze(1).unsqueeze(-1) | mk.unsqueeze(1).unsqueeze(2).expand(B, D, N, kvN.shape[2])  # [B,D,N,M]
                mask_crossN = (mq[:, None, :, None] | mk[:, None, None, :]).expand(B, D, N, kvN.shape[2])  # [B,D,N,M]

            else:
                mask_crossN = None
            y_crossN = self.mha_crossN(qN, kvN, kvN, mask_crossN)        # [B,D,N,2H]
        else:
            # fallback: use target self path
            y_crossN = qN

        y_crossN = y_crossN.transpose(1, 2)                      # [B,N,D,2H]

        # ===== cross-attention over D (per-N slice) =====
        qD = tgt_f  # [B,N,D,2H]


        if have_ctx:
            # Ensure ctx_f is [B, M, D, 2H] BEFORE pooling.
            # If D is on dim=1 and M on dim=2, swap them.#TODO
            if ctx_f.shape[2] != D and ctx_f.shape[1] == D:
                ctx_f = ctx_f.transpose(1, 2).contiguous()  # -> [B,M,D,2H]
            # (If already [B,M,D,2H], do nothing.)

            # Pool over M -> [B,D,2H]
            if mask_ctx is not None:
                m = (~mask_ctx).float()[:, :, None, None]  # [B,M,1,1], 1=keep
                #denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B,1,1,1]
                denom = m.sum(dim=1).clamp_min(1.0)  # [B,1,1]
                ctx_pooled = (ctx_f * m).sum(dim=1) / denom  # [B,D,2H]
            else:
                ctx_pooled = ctx_f.mean(dim=1)  # [B,D,2H]

            # --- make absolutely sure the feature width is 2H and NEVER fold batch ---
            E = qD.shape[-1]  # 2H
            ctx_pooled = ctx_pooled.contiguous()  # [B,D,?]

            if ctx_pooled.ndim != 3:
                raise RuntimeError(f"ctx_pooled rank {ctx_pooled.ndim} != 3")

            if ctx_pooled.shape[-1] != E:
                # Common bug: last dim == B*E (e.g., 4096 when B=32 and E=128).
                if ctx_pooled.shape[-1] % E == 0:
                    # Interpret as an extra spurious axis folded into features, average it out.
                    extra = ctx_pooled.shape[-1] // E
                    ctx_pooled = ctx_pooled.view(B, D, extra, E).mean(dim=2)  # -> [B,D,E]
                else:
                    # Last-resort: project to the right width (created once, reused after)
                    if not hasattr(self, "_align_ctxD"):
                        self._align_ctxD = nn.Linear(ctx_pooled.shape[-1], E, bias=False).to(ctx_pooled.device)
                    ctx_pooled = self._align_ctxD(ctx_pooled)  # -> [B,D,E]

            # Tile across N (materialize, do NOT rely on lazy expand semantics later)
            kvD = ctx_pooled.unsqueeze(1).expand(B, N, D, E).contiguous()  # [B,N,D,2H]

            # Mask whole N rows where targets are padded
            mask_crossD = mask[:, :, None, None].expand(B, N, D, D) if (mask is not None) else None

            y_crossD = self.mha_crossD(qD, kvD, kvD, mask_crossD)  # [B,N,D,2H]
        else:
            y_crossD = qD

        # ----- fuse both cross results, split, residual -----
        y = y_crossN + y_crossD         # [B,N,D,2H]
        residual, skip = torch.chunk(y, 2, dim=-1)                        # [B,N,D,H] each
        residual = F.gelu(residual)
        skip     = F.gelu(skip)
        return (s + residual) / math.sqrt(2.0), skip


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
        self.hidden_dim = hidden_dim
        self.input_linear = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList(
            [BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.proj_eps = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)

    def forward(
        self,
        x: torch.Tensor,                      # [B,N,D]
        y: torch.Tensor,                      # [B,N,1]  (noisy y_t)
        t: torch.Tensor,                      # [B] or [B,1]
        mask: torch.Tensor | None = None,     # [B,N] (True/1 = padded)
        x_context: torch.Tensor | None = None,
        y_context: torch.Tensor | None = None,
        mask_context: torch.Tensor | None = None,
    ) -> torch.Tensor:                        # -> [B,N,1]
        # preprocess both streams
        s_xy = process_inputs(x, y)                                # [B,N,D,2]
        s = F.gelu(self.input_linear(s_xy))                        # [B,N,D,H]

        s_ctx = None
        if (x_context is not None) and (y_context is not None):
            c_xy = process_inputs(x_context, y_context)            # [B,M,D,2]
            s_ctx = F.gelu(self.input_linear(c_xy))                # [B,M,D,H]

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
