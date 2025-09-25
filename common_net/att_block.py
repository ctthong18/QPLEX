from typing import Optional

import torch
import torch.nn as nn

from .attentions import MHAConfig
from .attentions import MultiHeadAttention as MHA
from .common import GatedMLP, ZeroCenteredRMSNorm
from .moe import MoE, MoEConfig


class AttnBlock(nn.Module):
    """Abstract base class for attention blocks."""

    pass


class MAB(AttnBlock):
    """Multihead Attention Block (MAB)

    This class implements a Multihead Attention Block that performs self-attention
    between query (Q), key (K), and value (V) tensors, followed by a feed-forward network.
    The architecture follows the standard transformer block design with normalization,
    attention, and feed-forward layers with residual connections.

    Parameters
    ----------
    embed_dim : int
        The embedding dimension for the input and output tensors.
    d_ff : int
        The hidden dimension size for the feed-forward network.
    mha_config : MHAConfig
        Configuration object for the multi-head attention module.
    inp_norm : type[nn.Module], default=ZeroCenteredRMSNorm
        The normalization layer type to use for input normalization.
    inp_norm_kwargs : dict, optional
        Additional keyword arguments to pass to the input normalization layer.
    moe_cls : type[MoE], optional
        The Mixture of Experts class to use instead of standard feed-forward network.
        If None, a GatedMLP will be used.
    moe_config : MoEConfig, optional
        Configuration object for the Mixture of Experts module if moe_cls is specified.

    Attributes
    ----------
    embed_dim : int
        The embedding dimension for the input and output tensors.
    inp_norm : nn.Module
        The input normalization layer.
    attn : MHA
        The multi-head attention module.
    ff : nn.Module
        The feed-forward network (either GatedMLP or a Mixture of Experts).

    Notes
    -----
    The forward pass applies layer normalization followed by self-attention,
    a residual connection, then a feed-forward network with another residual
    connection and final normalization.
    """

    """Multihead Attention Block (MAB)"""

    def __init__(
        self,
        embed_dim: int,
        d_ff: int,
        mha_config: MHAConfig,
        norm_cls: type[nn.Module] = ZeroCenteredRMSNorm,
        norm_kwargs: Optional[dict] = None,
        moe_cls: Optional[type[MoE]] = None,
        moe_config: Optional[MoEConfig] = None,
    ) -> None:
        super().__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        self.inp_norm = norm_cls(embed_dim, **norm_kwargs)

        self.attn = MHA(embed_dim=embed_dim, config=mha_config)

        self.ff_norm = norm_cls(embed_dim, **norm_kwargs)

        if moe_cls is not None:
            self.ff = moe_cls(embed_dim=embed_dim, d_ff=d_ff, config=moe_config)
        else:
            self.ff = GatedMLP(embed_dim, d_ff)

    def forward(
        self, X: torch.Tensor, causal: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        r"""Forward pass for the Multihead Attention Block.
        ```
        X -> Norm -> MHA -> + ----> Norm -> FF -> + ---> out
        |                   |   |                 |
        v                   |   v                 |
        -------------------->   ------------------>
        ```

        Args:
            X (torch.Tensor): _description_
            causal (bool, optional): _description_. Defaults to False.

        """

        # ---- Multi-Head Attention ----
        X_norm = self.inp_norm(X)  # (B, N, E)

        attn_out, attn_weights = self.attn(
            X_norm, X_norm, X_norm, causal=causal
        )  # (B, N, E)

        hidden_states = X + attn_out  # Residual

        # ---- Feed Forward ----
        ff_out = self.ff(self.ff_norm(hidden_states))  # (B, N, E)

        out = hidden_states + ff_out  # Residual (B, N, E)

        return out


class CrossMAB(AttnBlock):
    def __init__(
        self,
        embed_dim: int,
        d_ff: int,
        mha_config: MHAConfig,
        norm_cls: type[nn.Module] = ZeroCenteredRMSNorm,
        norm_kwargs: Optional[dict] = None,
        moe_cls: Optional[type[MoE]] = None,
        moe_config: Optional[MoEConfig] = None,
    ) -> None:
        super().__init__()
        norm_kwargs = norm_kwargs if norm_kwargs is not None else {}

        self.inp_norm = norm_cls(embed_dim, **norm_kwargs)

        self.kv_norm = norm_cls(embed_dim, **norm_kwargs)

        self.attn = MHA(embed_dim=embed_dim, config=mha_config)

        self.ff_norm = norm_cls(embed_dim, **norm_kwargs)

        if moe_cls is not None:
            self.ff = moe_cls(embed_dim=embed_dim, d_ff=d_ff, config=moe_config)
        else:
            self.ff = GatedMLP(embed_dim, d_ff)

    def forward(
        self, X: torch.Tensor, Y: torch.Tensor, causal: bool = False
    ) -> torch.Tensor:
        """
        X: (B, N, d)  query
        Y: (B, M, d)  key/value
        causal: whether to apply causal masking (not supported here)
        """

        # ---- Multi-Head Attention ----
        X_norm = self.inp_norm(X)  # (B, N, E)
        Y_norm = self.kv_norm(Y)  # (B, M, E)

        attn_out, attn_weights = self.attn(
            X_norm, Y_norm, Y_norm, causal=causal
        )  # (B, N, E)

        hidden_states = X + attn_out  # Residual

        # ---- Feed Forward ----
        ff_out = self.ff(self.ff_norm(hidden_states))  # (B, N, E)

        out = hidden_states + ff_out  # Residual (B, N, E)

        return out


class ISAB(AttnBlock):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_inducing: int,
        d_ff: int,
        dropout: float = 0.1,
        bias: bool = True,
        used_moe: bool = False,
        num_experts: int = 4,
        moe_top_k: int = 2,
    ):
        super().__init__()

        # Learnable inducing points (m, d)
        self.inducing_points = nn.Parameter(torch.randn(num_inducing, embed_dim))

        self.mab1 = CrossMAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            used_moe=used_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
        )

        self.mab2 = CrossMAB(
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            bias=bias,
            used_moe=used_moe,
            num_experts=num_experts,
            moe_top_k=moe_top_k,
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: (B, N, d)  input set
        mask: optional attention mask
        """

        B = x.shape[0]

        H, attn1_weights = self.mab1(
            self.inducing_points.unsqueeze(0).expand(B, -1, -1), x, mask
        )  # (B, m, d)

        X, attn2_weights = self.mab2(x, H)  # (B, N, d)

        return X


class ChainBlock(AttnBlock):
    def __init__(self, num_repeats: int, block_cls: type[AttnBlock], **block_kwargs):
        super().__init__()
        self.blocks = nn.ModuleList(
            [block_cls(**block_kwargs) for _ in range(num_repeats)]
        )

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, *args, **kwargs)

        return x
