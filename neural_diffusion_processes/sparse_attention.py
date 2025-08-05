# efficient_attention_torch.py
#
# PyTorch port of https://github.com/AminRezaei0x443/memory-efficient-attention
# Author: <you>
#
# Shape conventions
#   query : [..., Q, H, Dqk]
#   key   : [..., K, H, Dqk]
#   value : [..., K, H, Dv]
#
# Output : [..., Q, H, Dv]
#
# The code is 100 % torch-scriptable and works on CPU or CUDA.

from __future__ import annotations
import math
import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint


def _big_neg(dtype: torch.dtype) -> float:
    """Large negative number of the right type for masked logits."""
    return torch.finfo(dtype).min


# ---------------------------------------------------------------------------
#   ONE QUERY-CHUNK × ALL KEY-CHUNKS
# ---------------------------------------------------------------------------
def _query_chunk_attention(
    q_offset: int,
    query: Tensor,                        # [..., Qc, H, Dqk]
    key: Tensor,                          # [..., K, H, Dqk]
    value: Tensor,                        # [..., K, H, Dv]
    mask: Tensor | None,                  # [..., H, Qc, K]  or  None
    bias: Tensor | None,                  # same shape / broadcast rules
    *,
    key_chunk_size: int,
    mask_calc_fn=None,
    bias_calc_fn=None,
    weights_calc_fn=None,
    calc_fn_data=None,
) -> Tensor:                              # [..., Qc, H, Dv]
    """
    Computes attention for one **query chunk** against the full key/value set
    by iterating over KEY-chunks to keep memory at O(H · sqrt(N) · d).
    """

    _, num_heads, d_qk = key.shape[-3:]
    d_v = value.shape[-1]
    key_len = key.shape[-3]
    k_step = min(key_chunk_size, key_len)

    # Scale queries once up-front
    query = query / math.sqrt(d_qk)

    # Running (log-sum-exp) sketch:          S = Σ_j exp(s_ij) v_j
    #                                        Z = Σ_j exp(s_ij)
    S_acc = torch.zeros_like(
        query[..., :, :, :1]
    ).expand(*query.shape[:-1], d_v)        # [..., Qc, H, Dv]
    Z_acc = torch.zeros_like(query[..., :, :, 0])        # [..., Qc, H]
    M_acc = torch.full_like(Z_acc, _big_neg(query.dtype))  # max logits

    # -----------------------------------------------------------------------
    def _process_k_chunk(k_offset: int, k_chunk: Tensor, v_chunk: Tensor,
                         mask_chunk: Tensor | None, bias_chunk: Tensor | None):
        """
        One *key* chunk. Returns updated (S_acc, Z_acc, M_acc)
        following log-sum-exp merge rules.
        """
        # --- 1. raw scores --------------------------------------------------
        #   scores : [..., Qc, H, Kc]
        scores = torch.einsum("...qhd,...khd->...qhk", query, k_chunk)

        # --- 2. apply per-chunk hook functions -----------------------------
        if bias_calc_fn is not None:
            bias_chunk = bias_calc_fn(q_offset, k_offset,
                                      bias_chunk, scores, calc_fn_data)
        if bias_chunk is not None:
            scores = scores + bias_chunk.movedim(-2, -3)   # ...qhk

        if mask_calc_fn is not None:
            mask_chunk = mask_calc_fn(q_offset, k_offset,
                                      mask_chunk, scores, calc_fn_data)
        if mask_chunk is not None:
            scores = torch.where(mask_chunk.movedim(-2, -3),
                                 scores, _big_neg(scores.dtype))

        if weights_calc_fn is not None:
            scores = weights_calc_fn(q_offset, k_offset, scores, calc_fn_data)

        # --- 3. log-sum-exp merge with running sketch ----------------------
        M_chunk = scores.max(-1).values            # [..., Qc, H]
        exp_scores = torch.exp(scores - M_chunk.unsqueeze(-1))

        S_chunk = torch.einsum("...qhk,...khd->...qhd", exp_scores, v_chunk)
        Z_chunk = exp_scores.sum(-1)               # [..., Qc, H]

        # merge with accumulator   (see Appendix in the paper)
        M_new = torch.maximum(M_acc, M_chunk)
        alpha = torch.exp(M_acc - M_new)
        beta  = torch.exp(M_chunk - M_new)

        S_new = S_acc * alpha.unsqueeze(-1) + S_chunk * beta.unsqueeze(-1)
        Z_new = Z_acc * alpha + Z_chunk * beta

        return S_new, Z_new, M_new
    # -----------------------------------------------------------------------

    # Iterate over key-chunks
    for k_offset in range(0, key_len, k_step):
        k_slice = slice(k_offset, k_offset + k_step)
        k_chunk = key[..., k_slice, :, :]
        v_chunk = value[..., k_slice, :, :]

        mask_chunk = None
        bias_chunk = None
        if mask is not None:
            mask_chunk = mask[..., :, k_slice]
        if bias is not None:
            bias_chunk = bias[..., :, k_slice]

        # optional rematerialisation checkpoint for memory saving
        S_acc, Z_acc, M_acc = checkpoint(
            _process_k_chunk, k_offset, k_chunk, v_chunk,
            mask_chunk, bias_chunk
        )

    return S_acc / Z_acc.unsqueeze(-1)


# ---------------------------------------------------------------------------
#   FULL ATTENTION
# ---------------------------------------------------------------------------
def efficient_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    mask: Tensor | None = None,
    bias: Tensor | None = None,
    query_chunk_size: int = 1024,
    key_chunk_size: int = 4096,
    bias_calc_fn=None,
    mask_calc_fn=None,
    weights_calc_fn=None,
    calc_fn_data=None,
) -> Tensor:
    """
    Memory-efficient attention with O(√N) activations.

    Parameters are identical to the original JAX version.
    """
    q_len, num_heads, d_qk = query.shape[-3:]
    results: list[Tensor] = []

    q_step = min(query_chunk_size, q_len)

    q_offset = 0
    while q_offset < q_len:
        q_slice = slice(q_offset, q_offset + q_step)
        q_chunk = query[..., q_slice, :, :]             # [..., Qc, H, Dqk]

        mask_chunk = None
        bias_chunk = None
        if mask is not None:
            if mask.shape[-2] == 1:                     # broadcast row
                mask_chunk = mask
            else:
                mask_chunk = mask[..., q_slice, :]
        if bias is not None:
            if bias.shape[-2] == 1:
                bias_chunk = bias
            else:
                bias_chunk = bias[..., q_slice, :]

        # -- compute attention for this query chunk -------------------------
        out = _query_chunk_attention(
            q_offset,
            q_chunk,
            key,
            value,
            mask_chunk,
            bias_chunk,
            key_chunk_size=key_chunk_size,
            bias_calc_fn=bias_calc_fn,
            mask_calc_fn=mask_calc_fn,
            weights_calc_fn=weights_calc_fn,
            calc_fn_data=calc_fn_data,
        )
        results.append(out)
        q_offset += q_step

    return torch.cat(results, dim=-3)                   # concat along Q
