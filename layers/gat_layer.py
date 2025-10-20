
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class GATLayer(nn.Module):
    """
    Graph Attention Network Layer for RNA secondary structure feature aggregation
    """

    def __init__(self, in_dim, out_dim, num_heads, activation=F.leaky_relu,
                 dropout=0.1, residual=True):
        """
        Args:
            in_dim: Input feature dimension
            out_dim: Output dimension per attention head
            num_heads: Number of attention heads
            activation: Activation function
            dropout: Dropout probability
            residual: Whether to use residual connection
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.residual = residual
        self.activation = activation

        # Linear transformation layer
        self.linear = nn.Linear(in_dim, in_dim)

        # Graph attention layer
        self.gat = GATConv(
            in_feats=in_dim,
            out_feats=out_dim,
            num_heads=num_heads,
            feat_drop=dropout,
            attn_drop=dropout,
            residual=False,  # Manual residual connection
            activation=None  # Manual activation handling
        )

        # Dimension matching for residual connection
        if residual and in_dim != out_dim * num_heads:
            self.residual = False

    def forward(self, g, h):
        """Forward pass"""
        # Linear transformation
        h = self.linear(h)
        h_in = h  # Save input for residual connection

        # Graph attention computation
        h = self.gat(g, h)

        # Concatenate multi-head outputs
        h = h.view(-1, self.num_heads * self.out_dim)

        # Apply activation
        if self.activation:
            h = self.activation(h)

        # Residual connection
        if self.residual:
            h = h_in + h

        return h

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_dim={self.in_dim}, '
                f'out_dim={self.out_dim}, heads={self.num_heads}, '
                f'residual={self.residual})')
