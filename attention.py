import torch
import torch.nn.functional as F
def sdpa(q, k, v, q_lens=None, k_lens=None, dropout_p=0.0, softmax_scale=None, q_scale=None, causal=False, window_size=(-1, -1), deterministic=False, dtype=torch.bfloat16, version=None):
    assert window_size == (-1, -1) and (not causal)
    out = []
    original_dtype = q.dtype
    for b in range(q.shape[0]):
        nq = int(q_lens[b]) if q_lens is not None else q.shape[1]
        nk = int(k_lens[b]) if k_lens is not None else k.shape[1]
        vdtype = v.dtype if v.dtype in (torch.float16, torch.bfloat16) else dtype
        qb = q[b:b + 1, :nq].transpose(1, 2).to(vdtype)
        kb = k[b:b + 1, :nk].transpose(1, 2).to(vdtype)
        vb = v[b:b + 1, :nk].transpose(1, 2).to(vdtype)
        if q_scale is not None:
            qb = qb * q_scale
        z = F.scaled_dot_product_attention(qb, kb, vb, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale).transpose(1, 2)
        if nq < q.shape[1]:
            z = F.pad(z, (0, 0, 0, 0, 0, q.shape[1] - nq))
        out.append(z)
    return torch.cat(out).to(original_dtype)
