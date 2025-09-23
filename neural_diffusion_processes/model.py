import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math


def timestep_embedding(t: torch.Tensor, embedding_dim: int, max_positions: int = 10_000):
    """Sinusoidal embedding"""
    if t.ndim == 0:
        t = t.unsqueeze(0)  # make scalar into [1]

    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    return emb  # [B, H]


def scaled_dot_product_attention(q, k, v, mask=None):
    # q, k, v : [..., num_heads, seq_len, depth]
    matmul_qk = torch.einsum("...qd,...kd->...qk", q, k)
    depth = k.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(depth)

    if mask is not None:
        # mask is broadcastable to (..., 1, S_q, S_k)
        scaled_attention_logits = scaled_attention_logits + mask

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)
    output = torch.einsum("...qk,...kd->...qd", attention_weights, v)
    return output


class MultiHeadAttention(nn.Module):
    """
    Extended to support either:
      - self-attn mask vector for symmetric masking (original behavior), OR
      - kv_mask that masks only keys/values (for cross-attn with different lengths).
    """
    def __init__(self, d_model, num_heads, sparse=False):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0

        self.depth = d_model // num_heads
        self.attention = scaled_dot_product_attention

        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)

    def _build_pair_mask(self, mask, q_len, k_len):
        """
        Turn a vector mask (..., S) into a pair mask (..., 1, S_q, S_k)
        for the self-attn case where S_q == S_k.
        Values should be 0 for keep, 1 for mask.
        """
        if mask.dim() == 1:  # [N] -> [1,N]
            mask = mask.unsqueeze(0)
        if mask.dim() == 2:  # [B,N] -> [B,1,N] (will later broadcast)
            mask = mask.unsqueeze(1)
        # Now mask shape is (..., S). Build pairwise for S_q = S_k = S.
        mask_q = mask[..., :, None]                    # (..., S, 1)
        mask_k = mask[..., None, :]                    # (..., 1, S)
        pair = (mask_q + mask_k).clamp(max=1.0)        # (..., S, S) in {0,1}
        pair = pair.unsqueeze(-3)                      # (..., 1, S, S) for heads
        # Convert to additive logits mask
        return pair * (-1e9)

    def forward(self, v, k, q, mask=None, kv_mask=None):
        # q, k, v : [..., seq_len, d_model]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"
        q = rearrange(q, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        k = rearrange(k, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        v = rearrange(v, rearrange_arg, num_heads=self.num_heads, depth=self.depth)

        # Build additive mask for logits if provided
        add_mask = None
        if (mask is not None) and (kv_mask is not None):
            raise ValueError("Provide either mask (self) or kv_mask (cross), not both.")
        if mask is not None:
            # self-attention style mask where q_len == k_len
            q_len = q.size(-2)
            k_len = k.size(-2)
            if q_len != k_len:
                raise ValueError("Self-attn mask expects equal q_len and k_len.")
            add_mask = self._build_pair_mask(mask, q_len, k_len)
        elif kv_mask is not None:
            # kv_mask: (..., S_k) with 1 for masked positions
            if kv_mask.dim() == 1:
                kv_mask = kv_mask.unsqueeze(0)  # [1, S_k]
            # Expand to (..., 1, S_q, S_k)
            add_mask = kv_mask[..., None, :].unsqueeze(-3) * (-1e9)

        scaled_attention = self.attention(q, k, v, add_mask)
        scaled_attention = rearrange(
            scaled_attention,
            "... num_heads seq_len depth -> ... seq_len (num_heads depth)",
        )
        output = self.dense(scaled_attention)
        return output


class BiDimensionalAttentionBlock(nn.Module):
    """
    Bi-dimensional block with optional cross-attention from context on both axes.
    If s_ctx is None, reduces to the original self-attention-only block.
    """
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.linear_t = nn.Linear(hidden_dim, hidden_dim)

        # Self-attention modules (same as before)
        self.mha_d_self = MultiHeadAttention(2 * hidden_dim, num_heads)
        self.mha_n_self = MultiHeadAttention(2 * hidden_dim, num_heads)

        # Cross-attention modules (targets <- context)
        self.mha_d_cross = MultiHeadAttention(2 * hidden_dim, num_heads)
        self.mha_n_cross = MultiHeadAttention(2 * hidden_dim, num_heads)

        # Learnable gates for cross terms (init ~ 0 => start as "no cross")
        self.gamma_d = nn.Parameter(torch.tensor(0.0))
        self.gamma_n = nn.Parameter(torch.tensor(0.0))

    def forward(self, s_tgt, t, mask_tgt=None, s_ctx=None, mask_context=None):
        """
        s_tgt : [B, N_T, D, H]
        t     : [B, H]
        mask_tgt : [B, N_T] with 1=mask, 0=keep (optional)
        s_ctx : [B, N_C, D, H] or None
        mask_context : [B, N_C] with 1=mask, 0=keep (optional)
        """
        # Time conditioning
        t = self.linear_t(t)[:, None, None, :]  # [B,1,1,H]

        # Add time and duplicate features to 2H so we can split residual/skip
        y_t = s_tgt + t                         # [B,N_T,D,H]
        y_t = torch.cat([y_t, y_t], dim=-1)     # [B,N_T,D,2H]

        # ---------- Self-attention over D ----------
        y_self_d = self.mha_d_self(y_t, y_t, y_t)  # [B,N_T,D,2H]

        # ---------- Self-attention over N ----------
        y_t_r = y_t.transpose(1, 2)                # [B,D,N_T,2H]
        y_self_n = self.mha_n_self(y_t_r, y_t_r, y_t_r, mask=mask_tgt)
        y_self_n = y_self_n.transpose(1, 2)        # [B,N_T,D,2H]

        # --- replace the "Cross-attention terms" section in BiDimensionalAttentionBlock.forward ---

        # ---------- Cross-attention terms (if context is provided) ----------
        if (s_ctx is not None) and (s_ctx.size(1) > 0):
            y_c = s_ctx + t  # [B,N_C,D,H]
            #y_c = s_ctx
            y_c = torch.cat([y_c, y_c], dim=-1)  # [B,N_C,D,2H]

            # masked mean over N_C (robust when some/all rows are masked)
            if mask_context is not None and mask_context.numel() > 0:
                w = (1.0 - mask_context)[..., None, None]  # [B,N_C,1,1]
                denom = w.sum(dim=1, keepdim=True).clamp_min(1.0)  # avoid divide-by-0
                y_c_meanN = (y_c * w).sum(dim=1, keepdim=True) / denom  # [B,1,D,2H]
            else:
                y_c_meanN = y_c.mean(dim=1, keepdim=True)  # [B,1,D,2H]

            # D-axis cross-attn (targets query, context keys/values aggregated over N_C)
            y_cross_d = self.mha_d_cross(v=y_c_meanN, k=y_c_meanN, q=y_t, kv_mask=None)  # [B,N_T,D,2H]

            # N-axis cross-attn (targets query, context keys/values along N_C)
            y_c_r = y_c.transpose(1, 2)  # [B,D,N_C,2H]
            kv_mask_context = None
            if mask_context is not None and mask_context.numel() > 0:
                kv_mask_context = mask_context.unsqueeze(1).expand(-1, y_c_r.size(1), -1)  # [B,D,N_C]
            y_cross_n = self.mha_n_cross(v=y_c_r, k=y_c_r, q=y_t_r, kv_mask=kv_mask_context)  # [B,D,N_T,2H]
            y_cross_n = y_cross_n.transpose(1, 2)  # [B,N_T,D,2H]

            gate_d = torch.tanh(self.gamma_d)
            gate_n = torch.tanh(self.gamma_n)
            y = y_self_d + y_self_n + gate_d * y_cross_d + gate_n * y_cross_n
        else:
            # no context provided or N_C == 0 → no cross terms
            y = y_self_d + y_self_n


        # Split into residual/skip, apply nonlinearity, residual connect
        residual, skip = torch.chunk(y, 2, dim=-1)  # [B,N_T,D,H] x2
        residual = F.gelu(residual)
        skip = F.gelu(skip)

        return (s_tgt + residual) / math.sqrt(2.0), skip


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, sparse=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads, sparse=sparse)

    def forward(self, s, t):
        # s: [B, N, D, hidden_dim]
        t = self.linear_t(t)[:, None, :]
        y = s + t
        y = torch.cat([y, y], dim=-1)
        y_att_d = self.mha_d(y, y, y)
        residual, skip = torch.chunk(y_att_d, 2, dim=-1)
        residual = F.gelu(residual)
        skip = F.gelu(skip)
        return (s + residual) / math.sqrt(2.0), skip


class BiDimensionalAttentionModel(nn.Module):
    """
    Targets conditioned on optional context via cross-attention.
    forward(...):
        x_tgt, y_tgt, t, mask_tgt,
        x_context=None, y_context=None, mask_context=None
    """
    def __init__(self, n_layers, hidden_dim, num_heads, init_zero=True):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.init_zero = init_zero

        self.input_linear = nn.Linear(2, hidden_dim)
        self.layers = nn.ModuleList(
            [BiDimensionalAttentionBlock(hidden_dim, num_heads) for _ in range(n_layers)]
        )
        self.proj_eps = nn.Linear(hidden_dim, hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, 1)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def process_inputs(self, x, y):
        # x : [B, N, D]  or [N, D]
        # y : [B, N, 1]  or [N, 1]
        if x is None or y is None:
            return None

        if x.ndim == 2:  # [N,D]  →  add batch axis
            x = x.unsqueeze(0)
        if y.ndim == 2:  # [N,1]  →  add batch axis
            y = y.unsqueeze(0)  # [1,N,1]

        if x.ndim == 3:
            x = x.unsqueeze(-1)  # [B,N,D,1]
        if y.ndim == 3:
            y = y.unsqueeze(-1)  # [B,N,1,1]

        # tile y along D if needed
        if y.size(2) == 1 and x.size(2) > 1:
            y = y.repeat(1, 1, x.size(2), 1)  # [B,N,D,1]

        return torch.cat([x, y], dim=-1)  # [B,N,D,2]

    def forward(self, x_tgt, y_tgt, t, mask_tgt=None, x_context=None, y_context=None, mask_context=None):
        # preprocess targets
        s_t = self.process_inputs(x_tgt, y_tgt)         # [B,N_T,D,2]
        s_t = F.gelu(self.input_linear(s_t))            # [B,N_T,D,H]

        # optional context
        s_c = None
        if (x_context is not None) and (y_context is not None):
            s_c = self.process_inputs(x_context, y_context)     # [B,N_C,D,2]
            s_c = F.gelu(self.input_linear(s_c))        # [B,N_C,D,H]

        t_embedding = timestep_embedding(t, self.hidden_dim)  # [B,H]

        skip = None
        for layer in self.layers:
            s_t, skip_connection = layer(
                s_t, t_embedding, mask_tgt=mask_tgt, s_ctx=s_c, mask_context=mask_context
            )
            skip = skip_connection if skip is None else skip + skip_connection

        skip = reduce(skip, "b n d h -> b n h", "mean")       # [B,N_T,H]
        eps = skip / math.sqrt(self.n_layers)
        eps = F.gelu(self.proj_eps(eps))                      # [B,N_T,H]
        eps = self.output_linear(eps)                         # [B,N_T,1]
        return eps


class AttentionModel(nn.Module):
    def __init__(self, n_layers, hidden_dim, num_heads, output_dim, sparse=False, init_zero=True):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.output_dim = output_dim

        self.input_linear = nn.Linear(hidden_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [AttentionBlock(hidden_dim, num_heads, sparse=sparse) for _ in range(n_layers)]
        )
        self.mid = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def forward(self, x, y, t, mask=None):
        # x : [B, N, D_in]  y : [B, N, D_out]  t: [B, 1]
        x = torch.cat([x, y], dim=-1)              # [B, N, D_in + D_out]
        x = F.gelu(self.input_linear(x))           # [B, N, H]
        t_embedding = timestep_embedding(t, self.hidden_dim)  # [B, H]

        skip = None
        for layer in self.layers:
            x, skip_connection = layer(x, t_embedding)        # [B, N, H]
            skip = skip_connection if skip is None else skip + skip_connection

        eps = skip / math.sqrt(self.n_layers)      # [B, N, H]
        eps = F.gelu(self.mid(eps))                # [B, N, H]
        eps = self.output_linear(eps)              # [B, N, output_dim]
        return eps

