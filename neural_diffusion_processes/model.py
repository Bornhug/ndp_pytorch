import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce
import math


def timestep_embedding(t: torch.Tensor, embedding_dim: int, max_positions: int = 10_000):
    """Sinusoidal embedding"""
    if t.ndim == 0:
        t = t.unsqueeze(0)  # 👈 fix: make scalar into [1]

    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))

    # emb : [t, embedding_dim] OR [B, H]

    return emb


def scaled_dot_product_attention(q, k, v, mask=None):

    # q, k, v : [..., num_heads, seq_len, depth]

    matmul_qk = torch.einsum("...qd,...kd->...qk", q, k)

    # matmul_qk : [..., seq_len, seq_len]

    depth = k.size(-1)
    scaled_attention_logits = matmul_qk / math.sqrt(depth)

    # scaled_attention_logits : [..., seq_len, seq_len]

    if mask is not None:
        scaled_attention_logits += mask * -1e9

    attention_weights = F.softmax(scaled_attention_logits, dim=-1)

    # attention_weights : [..., seq_len, seq_len]

    output = torch.einsum("...qk,...kd->...qd", attention_weights, v)

    # output : [..., num_heads, seq_length, depth]

    return output


class MultiHeadAttention(nn.Module):
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

    def forward(self, v, k, q, mask=None):
        # q, k, v : [..., seq_len, d_model]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        # q, k, v : [..., seq_len, d_model]

        rearrange_arg = "... seq_len (num_heads depth) -> ... num_heads seq_len depth"
        q = rearrange(q, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        k = rearrange(k, rearrange_arg, num_heads=self.num_heads, depth=self.depth)
        v = rearrange(v, rearrange_arg, num_heads=self.num_heads, depth=self.depth)

        # q, k, v : [..., num_heads, seq_len, depth]

        if mask is not None:
            if mask.dim() == 1:  # [N] -> [1,N]
                mask = mask.unsqueeze(0)
            if mask.dim() == 2:  # [B,N] -> [B,1,N] (will later broadcast over D)
                mask = mask.unsqueeze(1)
            mask_seq_q = mask[..., :, None]   # (..., S, 1)
            mask_seq_v = mask[..., None, :]   # (..., 1, S)
            mask = mask_seq_q + mask_seq_v    # broadcast to (..., S, S)
            mask = torch.where(mask == 0.0, mask, torch.ones_like(mask))
            mask = mask[..., None, :, :]      # add a head axis → (..., 1, S, S)

        scaled_attention = self.attention(q, k, v, mask)

        # scaled_attention : [..., num_heads, seq_len, depth]

        scaled_attention = rearrange(
            scaled_attention,
            "... num_heads seq_len depth -> ... seq_len (num_heads depth)",
        )

        # scaled_attention : [..., seq_len, d_model]

        output = self.dense(scaled_attention)

        # output : [..., seq_len, d_model]

        return output


class BiDimensionalAttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads)
        self.mha_n = MultiHeadAttention(2 * hidden_dim, num_heads)

    def forward(self, s, t, mask=None):
        # s : (B, N, D, hidden_dim)
        # t : [B, hidden_dim]

        t = self.linear_t(t)[:, None, None, :]
        # t : [B, 1, 1, hidden_dim]

        y = s + t
        # y : [B, N, D, hidden_dim]

        #TODO: Maybe alternative way is to reduce the input dim of Q,K,V to d_model/ ?
        y = torch.cat([y, y], dim=-1)               # [B,N,D,2H]

        y_att_d = self.mha_d(y, y, y)
        # y_att_d : [B, N, D, hidden_dim]

        y_r = y.transpose(1, 2)
        # y_r : [B, D, N, hidden_dim]

        if mask is not None: # [B, N]
            mask = mask.unsqueeze(1) # [B, 1, N]

        y_att_n = self.mha_n(y_r, y_r, y_r, mask)
        # y_att_n : [B, D, N, 2 * hidden_dim]

        y_att_n = y_att_n.transpose(1, 2)
        # y_att_n : [B, N, D, 2 * hidden_dim]

        y = y_att_n + y_att_d
        # y : [B, N, D, 2 * hidden_dim]

        residual, skip = torch.chunk(y, 2, dim=-1)
        # residual, skip : [B, N, D, hidden_dim]

        residual = F.gelu(residual)
        # residual : [B, N, D, hidden_dim]

        skip = F.gelu(skip)
        # skip : [B, N, D, hidden_dim]

        return (s + residual) / math.sqrt(2.0), skip


class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, sparse=False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.linear_t = nn.Linear(hidden_dim, hidden_dim)
        self.mha_d = MultiHeadAttention(2 * hidden_dim, num_heads, sparse=sparse)

    def forward(self, s, t):
        # s: [B, N, D, hidden_dim]
        # t : [B, hidden_dim]

        t = self.linear_t(t)[:, None, :]
        # t: [B, 1, hidden_dim]

        y = s + t
        # y : [B, N, D, hidden_dim]

        y_att_d = self.mha_d(y, y, y)
        # y_att_d : [B, N, D, 2 * hidden_dim]

        residual, skip = torch.chunk(y_att_d, 2, dim=-1)
        # residual, skip : [B, N, D, hidden_dim]

        residual = F.gelu(residual)
        # residual: [B, N, D, hidden_dim]

        skip = F.gelu(skip)
        # skip: [B, N, D, hidden_dim]

        return (s + residual) / math.sqrt(2.0), skip


class BiDimensionalAttentionModel(nn.Module):
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

        if x.ndim == 2:  # [N,D]  →  add batch axis
            x = x.unsqueeze(0)
        if y.ndim == 2:  # [N,1]  →  add batch axis
            y = y.unsqueeze(0)  # now [1,N,1]

        if x.ndim == 3:
            x = x.unsqueeze(-1)  # [B,N,D,1]
        if y.ndim == 3:
            y = y.unsqueeze(-1)  # [B,N,1,1]

        # y has shape [B,N,1,1];  need to tile along D so it matches x
        if y.size(2) == 1 and x.size(2) > 1:
            y = y.repeat(1, 1, x.size(2), 1)  # [B,N,D,1]

        return torch.cat([x, y], dim=-1)  # [B,N,D,2]

    def forward(self, x, y, t, mask=None):
        x = self.process_inputs(x, y) # [B, N, D, 2]
        x = self.input_linear(x) # [B, N, D, H]
        x = F.gelu(x) # [B, N, D, H]

        t_embedding = timestep_embedding(t, self.hidden_dim) # [B, H]

        skip = None
        for layer in self.layers:
            x, skip_connection = layer(x, t_embedding, mask) # [B, N, D, H]
            skip = skip_connection if skip is None else skip + skip_connection

        skip = reduce(skip, "b n d h -> b n h", "mean") # [B, N ,H]
        eps = skip / math.sqrt(self.n_layers) # [B, N ,H]
        eps = F.gelu(self.proj_eps(eps))   # [B, N ,H]
        eps = self.output_linear(eps) # [B, N ,1]
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
        self.output_linear = nn.Linear(hidden_dim, output_dim)
        if init_zero:
            nn.init.zeros_(self.output_linear.weight)
            nn.init.zeros_(self.output_linear.bias)

    def forward(self, x, y, t, mask=None):
        # x : [B, N, D_in]  y : [B, N, D_out]  t: [B, 1]  mask: [B, N]

        x = torch.cat([x, y], dim=-1) # [B, N, D_in + D_out]
        x = self.input_linear(x) # [B, N, H] # TODO
        x = F.gelu(x) # [B, N, H]

        t_embedding = timestep_embedding(t, self.hidden_dim) # [B, H]

        skip = None
        for layer in self.layers:
            x, skip_connection = layer(x, t_embedding) # [B, N, H]
            skip = skip_connection if skip is None else skip + skip_connection

        eps = skip / math.sqrt(self.n_layers) # [B, N, H]
        eps = F.gelu(nn.Linear(self.hidden_dim, self.hidden_dim)(eps)) # [B, N, H]
        eps = self.output_linear(eps) # [B, N, output_dim]
        return eps


