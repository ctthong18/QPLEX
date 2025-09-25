import torch
import torch.nn as nn

from typing import Callable, Optional

from .base_attn import AttentionBase, ScaledDotProductAttention
from ..common import Gated, ZeroCenteredRMSNorm


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_cls: type[AttentionBase] = ScaledDotProductAttention,
        attn_kwargs: Optional[dict] = None,
        gated: bool = False,
        gate_act_fn: str | Callable[[torch.Tensor], torch.Tensor] = "sigmoid",
        gate_operator_fn: str | Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = "*",
        num_k_heads: Optional[int] = None,
        num_v_heads: Optional[int] = None,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        qk_norm_cls: type[nn.Module] = ZeroCenteredRMSNorm,
        qk_norm_kwargs: Optional[dict] = None,
        bias: bool = False,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_k_heads = num_k_heads if num_k_heads is not None else num_heads
        self.num_v_heads = num_v_heads if num_v_heads is not None else num_heads
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        factory_kwargs = {"device": device, "dtype": dtype}

        self.head_dim = embed_dim // self.num_heads

        if embed_dim % num_heads != 0:
            raise ValueError(f"Embedding dimension ({embed_dim}) must be divisible by the number of heads ({num_heads}).")

        if self.num_heads == self.num_k_heads and self.num_heads == self.num_v_heads:
            pass    # normal attention
        elif self.num_heads > max(self.num_k_heads, self.num_v_heads):      # group query Attention
            if self.num_heads % self.num_k_heads != 0:
                raise ValueError(f"Number of attention heads must be multiple of number of key heads, get H={self.num_heads}, Hk={self.num_k_heads}")
            if self.num_heads % self.num_v_heads != 0:
                raise ValueError(f"Number of attention heads must be multiple of number of value heads, get H={self.num_heads}, Hk={self.num_v_heads}")
        else:   # to be update
            raise NotImplementedError("Only support Group Query Attention for now (Hk == Hv < Hq), get Hq={}, Hk={}, Hv={}".format(self.num_heads, self.num_k_heads, self.num_v_heads))

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, self.num_k_heads * self.head_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, self.num_v_heads * self.head_dim, bias=bias, **factory_kwargs)

        if gated:
            self.gate = Gated(embed_dim, embed_dim, bias = bias, gate_act_fn=gate_act_fn, gate_operator_fn=gate_operator_fn)
        else:
            self.gate = None

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        attn_kwargs = attn_kwargs if attn_kwargs is not None else {}
        self.attn = attn_cls(**attn_kwargs)

        self._reset_parameters()

        qk_norm_kwargs = qk_norm_kwargs if qk_norm_kwargs is not None else {}

        self.q_norm = qk_norm_cls(self.head_dim, **qk_norm_kwargs)
        self.k_norm = qk_norm_cls(self.head_dim, **qk_norm_kwargs)

    def _reset_parameters(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        self.apply(_init_weights)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
        # Q: (B, N, E)
        # K: (B, N, E)
        # V: (B, N, E)
        B, N, _ = Q.size()
        H = self.num_heads
        Hk = self.num_k_heads
        Hv = self.num_v_heads
        D = self.head_dim

        # E = H*D

        # Project to multi-head
        Q_proj: torch.Tensor = self.q_proj(Q)        # (B, N, H * D)
        K_proj: torch.Tensor = self.k_proj(K)        # (B, N, Hk * D)
        V_proj: torch.Tensor = self.v_proj(V)        # (B, N, Hv * D)

        Q_proj = Q_proj.view(B, N, H, D).transpose(1, 2)  # (B, H , N, D)
        K_proj = K_proj.view(B, N, Hk, D).transpose(1, 2) # (B, Hk, N, D)
        V_proj = V_proj.view(B, N, Hv, D).transpose(1, 2) # (B, Hv, N, D)

        Q_proj = self.q_norm(Q_proj)
        K_proj = self.k_norm(K_proj)

        # Attention
        attn_out, attn_weights = self.attn(Q_proj, K_proj, V_proj, causal=causal)  # (B, H, N, D)

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, N, H * D)  # (B, H, N, D) -> (B, N, H, D) -> (B, N, E)

        if self.gate is not None:
            attn_out = self.gate(Q, attn_out)       # attn_out = attn_out o act(Q @ W)  -> (B, N, E)

        out = self.out_proj(attn_out)        # (B, N, E)
        return out, attn_weights
