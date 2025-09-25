from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_attn import AttentionBase


def _elu_feature_map(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


class LinearAttention(AttentionBase):
    """
    Linear Attention (kernel-based, positive feature map) supporting causal and non-causal modes.

    Expected input shapes:
      Q, K, V : tensors of shape (B, H, N, D)
        B = batch, H = heads, N = seq_len, D = head_dim

    Output:
      out : (B, H, N, D_v)  where D_v == V.shape[-1]

    Parameters:
      feature_map: callable tensor->tensor. Must return non-negative values. Default: elu(x)+1.
      eps: small float for numerical stability
    """

    def __init__(
        self,
        feature_map: Callable[[torch.Tensor], torch.Tensor] = _elu_feature_map,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.feature_map = feature_map
        self.eps = eps

    def _forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal: bool = False
    ):
        # Feature mapping
        # Q: B, H, N, D
        # K: B, H, N, D
        # V: B, H, N, Dv

        Qf = self.feature_map(Q)  # (B,H,N,Dphi)
        Kf = self.feature_map(K)  # (B,H,N,Dphi)

        if not causal:
            # Non-causal (fast global aggregation)
            # phi(K).T (B, H, Dphi, N) @ V (B, Hv, N, Dv) -> (B, H, Dphi, Dv)
            KV = torch.matmul(  # phi(K).T @ V
                Kf.transpose(-2, -1),  # [B, H, D, N]
                V,  # [B, H, N, E]
            )  # (B, H, Dphi, Dv)

            # phi(K).T * I
            Kf_prefix = Kf.sum(dim=2)  # (B, H, Dphi)

            numerator = torch.matmul(  # phi(Q) @ (phi(K).T @ V)
                Qf,  # (B,H,N,Dphi)
                KV,  # (B,H,Dphi,Dv)
            )  # (B,H,N,Dv)

            # Denominator: z = Qf_i dot Kf_prefix -> (B,H,N,1)
            denominator = torch.matmul(  # phi(Q) @ (phi(K) @ I)
                Qf,  # (B,H,N,Dphi)
                Kf_prefix.unsqueeze(-1),  # (B,H,Dphi,1)
            )  # (B,H,N,1)
        else:
            # Causal attention: use prefix sums
            # We need per-position prefix sums of Kf and Kf * V
            # Kf: (B,H,N,Dphi), V: (B,H,N,Dv)
            # Build KV_each_pos = Kf[n].unsqueeze(-1) * V[n].unsqueeze(-2) => (B,H,N,Dphi,Dv)
            KV_each = Kf.unsqueeze(-1) * V.unsqueeze(-2)  # (B,H,N,Dphi,Dv)

            # prefix sums along sequence dim (n)
            KV_prefix = KV_each.cumsum(dim=2)  # (B,H,N,Dphi,Dv)
            Kf_prefix = Kf.cumsum(dim=2)  # (B,H,N,Dphi)

            # Numerator at position n: Qf[n] @ KV[n]
            # einsum: out_num[b,h,n,e] = sum_d Qf[b,h,n,d] * KV[b,h,n,d,e]
            numerator = torch.einsum(
                "b h n d, b h n d e -> b h n e", Qf, KV_prefix
            )  # (B,H,N,Dv)

            # Denominator: denom[b,h,n] = Qf[b,h,n] dot Kf_prefix[b,h,n]
            denominator = torch.einsum("b h n d, b h n d -> b h n", Qf, Kf_prefix).unsqueeze(
                -1
            )  # (B,H,N,1)

        out = numerator / (denominator + self.eps)  # (B,H,N,Dv) / (B,H,N,1) -> (B,H,N,Dv)
        return out, None    # no attention weights


class MHLA(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        feature_map: Callable[[torch.Tensor], torch.Tensor] = _elu_feature_map,
        eps: float = 1e-6,
        bias: bool = False,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        device: Optional[torch.device | str] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.feature_map = feature_map
        self.eps = eps
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        self.attn = LinearAttention(feature_map=feature_map, eps=eps)
        self.dropout_layer = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.0)
            nn.init.constant_(self.k_proj.bias, 0.0)
            nn.init.constant_(self.v_proj.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, causal=False):
        # Q: (B, N, E)
        # K: (B, N, Dk)
        # V: (B, N, Dv)
        B, N, _ = Q.size()
        H = self.num_heads
        D = self.embed_dim // H

        # Project to multi-head
        Q = self.q_proj(Q)
        K = self.k_proj(K).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        V = self.v_proj(V).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        Q = Q.view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        # Linear Attention
        out: torch.Tensor = self.attn(Q, K, V, causal=causal)  # (B, H, N, D)
        if isinstance(out, tuple):
            out = out[0]
        out = out.transpose(1, 2).contiguous().view(B, N, H * D)  # (B, N, E)

        out = self.out_proj(out)
        out = self.dropout_layer(out)
        return out

