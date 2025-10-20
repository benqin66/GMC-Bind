
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from layers.MCNN_layer import MCNNLayer
from layers.gat_layer import GATLayer


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding for sequence position information
    """

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input features"""
        # x: (L, N, D) or (batch, seq_len, d_model)
        if x.dim() == 3:
            x = x.transpose(0, 1)  # Convert to (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0), :]
        return x.transpose(0, 1)  # Convert back to (batch, seq_len, d_model)


class CrossAttentionLayer(nn.Module):
    """
    Cross-Attention Layer for multimodal feature fusion
    """

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        """
        Forward pass for cross-attention mechanism
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            
        Returns:
            output: Cross-attention output with residual connection
        """
        # Cross-attention computation
        attn_output, _ = self.multihead_attn(
            query=query,
            key=key,
            value=value,
            need_weights=False
        )

        # Residual connection and layer normalization
        output = self.norm(query + self.dropout(attn_output))
        return output


class GMCBindNet(nn.Module):
    """
    GMC-Bind Main Network Architecture
    
    Multi-modal RNA-protein binding prediction framework integrating:
    - Multi-scale sequence features (MCNN)
    - Graph structural features (GAT)
    - Cross-attention fusion
    - Positional encoding
    """

    def __init__(self, net_params):
        super().__init__()
        # Basic parameters
        self.device = net_params['device']
        in_dim = net_params['in_dim']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        self.batch_norm = net_params['batch_norm']
        self.residual = net_params['residual']
        self.n_classes = net_params['n_classes']
        seq_len = net_params['seq_len']  # Sequence length
        self.seq_len = seq_len
        self.num_gat_layers = net_params.get('num_gat_layers', 2)

        # Sequence feature processing
        self.mcnn = MCNNLayer(
            in_channel=4,  # A/U/C/G nucleotides
            out_channel=hidden_dim // 2,
            kernel_sizes=[1, 3, 5, 7, 9, 11],  # Multi-scale convolution kernels
            activation=nn.ReLU(),
            batch_norm=True,
            residual=True
        )

        # Structural feature processing - multi-layer GAT stack
        self.gat = self._build_gat_layers(
            in_dim=in_dim,
            out_dim=hidden_dim // 2,
            num_layers=self.num_gat_layers
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=hidden_dim // 2,
            max_len=seq_len * 2
        )

        # Cross-attention layer
        self.cross_attn = CrossAttentionLayer(
            d_model=hidden_dim,
            nhead=4,  # Number of attention heads
            dropout=0.1
        )

        # Prediction module
        self.prediction = nn.Sequential(
            nn.Linear(seq_len * hidden_dim, 256),  # Adjusted to dynamic dimension
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.n_classes)
        )

    def _build_gat_layers(self, in_dim, out_dim, num_layers):
        """
        Build multi-layer GAT stack for hierarchical structural feature extraction
        
        Args:
            in_dim: Input dimension
            out_dim: Output dimension per layer
            num_layers: Number of GAT layers
            
        Returns:
            nn.ModuleList: Stack of GAT layers
        """
        layers = nn.ModuleList()

        # First layer: input to hidden
        layers.append(GATLayer(
            in_dim=in_dim,
            out_dim=out_dim,
            num_heads=4,
            activation=F.leaky_relu,
            dropout=0.1,
            residual=True
        ))

        # Middle layers: hidden to hidden
        for _ in range(1, num_layers - 1):
            layers.append(GATLayer(
                in_dim=out_dim * 4,  # Dimension after multi-head concatenation
                out_dim=out_dim,
                num_heads=4,
                activation=F.leaky_relu,
                dropout=0.1,
                residual=True
            ))

        # Final layer: hidden to output
        if num_layers > 1:
            layers.append(GATLayer(
                in_dim=out_dim * 4,  # Dimension after multi-head concatenation
                out_dim=out_dim,
                num_heads=4,
                activation=F.leaky_relu,
                dropout=0.1,
                residual=True
            ))

        return layers

    def forward(self, g):
        """
        Forward pass through the complete GMC-Bind architecture
        
        Args:
            g: DGL graph containing sequence and structural information
            
        Returns:
            pred: Binding site prediction probabilities
        """
        # Get node features (sequence information)
        seq_features = g.ndata['feat']  # (total_nodes, 4)

        # Reshape to sequence format
        batch_size = g.batch_size
        seq_features = seq_features.view(batch_size, self.seq_len, 4)

        # MCNN processing for sequence features
        # Convert dimensions for convolution
        seq_input = seq_features.permute(0, 2, 1)
        seq_emb = self.mcnn(seq_input)
        seq_emb = seq_emb.permute(0, 2, 1)

        # GAT processing for structural features
        struct_emb = self._process_graph(g)

        # Apply positional encoding
        seq_emb_pos = self.pos_encoder(seq_emb)
        struct_emb_pos = self.pos_encoder(struct_emb)

        # Cross-attention fusion
        # Sequence as query, structure as key and value
        fused_features = self.cross_attn(
            query=seq_emb_pos,
            key=struct_emb_pos,
            value=struct_emb_pos
        )  # (batch, seq_len, hidden_dim)

        # Flatten features for prediction
        flattened = fused_features.view(batch_size, -1)

        # Binding site prediction
        pred = self.prediction(flattened)
        return pred

    def _process_graph(self, g):
        """
        Process graph structural features through multi-layer GAT stack
        
        Args:
            g: Input graph with structural features
            
        Returns:
            h: Processed structural embeddings
        """
        # Initial node features
        h = g.ndata['feat']

        # GAT layer stacking - hierarchical processing
        for i, gat_layer in enumerate(self.gat):
            h = gat_layer(g, h)

            if self.batch_norm and i < len(self.gat) - 1:
                h = nn.BatchNorm1d(h.shape[1]).to(self.device)(h)

        # Reshape to sequence format
        batch_size = g.batch_size
        return h.view(batch_size, self.seq_len, -1)
