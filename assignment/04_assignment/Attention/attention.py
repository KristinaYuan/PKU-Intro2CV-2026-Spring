import math

import torch
import torch.nn as nn


def make_causal_mask(seq_len, device=None, dtype=torch.float32):
    """
    Build an additive causal mask for self-attention.

    The returned tensor has shape (1, 1, seq_len, seq_len). Entry (i, j) is 0
    when token i is allowed to attend to token j, and a large negative value
    when j is a future token that should be masked.
    """
    # ---------------- TODO ----------------
    # Return an upper-triangular additive mask. The diagonal should be 0.
    # Example for seq_len = 3:
    # [[0, -1e9, -1e9],
    #  [0,     0, -1e9],
    #  [0,     0,     0]]
    # --------------------------------------
    mask = torch.triu(
        torch.full((seq_len, seq_len), -1e9, device=device, dtype=dtype),
        diagonal=1,
    )
    return mask.view(1, 1, seq_len, seq_len)


def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Compute scaled dot-product attention.

    Args:
        Q: query tensor with shape (B, H, Tq, D)
        K: key tensor with shape (B, H, Tk, D)
        V: value tensor with shape (B, H, Tk, Dv)
        mask: optional additive mask broadcastable to (B, H, Tq, Tk).
              Valid positions should be 0 and masked positions should be a
              large negative value such as -1e9.

    Returns:
        out: attended values with shape (B, H, Tq, Dv)
        attn: attention weights with shape (B, H, Tq, Tk)
    """
    # ---------------- TODO ----------------
    # 1. Compute Q @ K^T / sqrt(D).
    # 2. Add mask if it is not None.
    # 3. Apply softmax over the key dimension.
    # 4. Compute attn @ V.
    # --------------------------------------
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        scores = scores + mask

    attn = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn, V)
    return out, attn


class MultiHeadSelfAttention(nn.Module):
    """
    A small multi-head self-attention layer.

    This module is intentionally minimal. It is enough for the captioning
    transformer used in this assignment, and it avoids relying on PyTorch's
    built-in MultiheadAttention so that the masking logic is visible.
    """

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, mask=None):
        """
        Args:
            x: input tensor with shape (B, T, C)
            mask: optional additive mask broadcastable to (B, H, T, T)

        Returns:
            out: tensor with shape (B, T, C)
            attn: attention weights with shape (B, H, T, T)
        """
        # ---------------- TODO ----------------
        # 1. Project x to Q, K, V.
        # 2. Split each projection into multiple heads.
        # 3. Call scaled_dot_product_attention.
        # 4. Merge the heads and apply out_proj.
        # --------------------------------------
        B, T, _ = x.shape

        def split_heads(tensor):
            return tensor.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        Q = split_heads(self.q_proj(x))
        K = split_heads(self.k_proj(x))
        V = split_heads(self.v_proj(x))

        out, attn = scaled_dot_product_attention(Q, K, V, mask)
        out = out.transpose(1, 2).contiguous().view(B, T, self.embed_dim)
        out = self.out_proj(out)
        return out, attn
