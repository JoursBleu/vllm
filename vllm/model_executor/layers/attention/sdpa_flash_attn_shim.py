"""
SDPA-based shim for flash_attn_varlen_func.
Fallback for platforms without flash_attn (e.g., RDNA3/gfx1100).
"""
import torch
import torch.nn.functional as F


def sdpa_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    softmax_scale=None,
    causal=False,
    return_attn_probs=False,
    **kwargs,
):
    """
    Drop-in replacement for flash_attn_varlen_func using torch SDPA.
    q/k/v: (total_tokens, num_heads, head_dim)
    cu_seqlens_q/k: (batch_size + 1,) cumulative sequence lengths
    """
    batch_size = cu_seqlens_q.shape[0] - 1
    outputs = []
    lse_list = []

    for i in range(batch_size):
        q_start, q_end = cu_seqlens_q[i].item(), cu_seqlens_q[i + 1].item()
        k_start, k_end = cu_seqlens_k[i].item(), cu_seqlens_k[i + 1].item()

        # (seq_len, num_heads, head_dim) -> (1, num_heads, seq_len, head_dim)
        qi = q[q_start:q_end].transpose(0, 1).unsqueeze(0)
        ki = k[k_start:k_end].transpose(0, 1).unsqueeze(0)
        vi = v[k_start:k_end].transpose(0, 1).unsqueeze(0)

        # Handle different head dims between q/k and v
        if vi.shape[-1] < qi.shape[-1]:
            vi = F.pad(vi, [0, qi.shape[-1] - vi.shape[-1]])

        oi = F.scaled_dot_product_attention(
            qi, ki, vi,
            scale=softmax_scale,
            is_causal=causal,
            enable_gqa=True,
        )

        # (1, num_heads, seq_len, head_dim) -> (seq_len, num_heads, head_dim)
        oi = oi.squeeze(0).transpose(0, 1)
        # Trim v padding if needed
        if v.shape[-1] < q.shape[-1]:
            oi = oi[..., :v.shape[-1]]

        outputs.append(oi)

    out = torch.cat(outputs, dim=0)

    if return_attn_probs:
        # Return dummy lse (num_heads, total_q_tokens)
        lse = torch.zeros(
            q.shape[1], q.shape[0], dtype=torch.float32, device=q.device
        )
        return out, lse, None
    return out
