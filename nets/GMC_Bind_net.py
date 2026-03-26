import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dgl
from layers.MSWC_layer import MSWCLayer
from layers.gat_layer import GATLayer


class PositionalEncoding(nn.Module):
    """Sinusoidal Positional Encoding"""

    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model

        # Create position encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """Add positional encoding to input features"""
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x


class BidirectionalCrossAttention(nn.Module):
    """Bidirectional Cross-Attention Layer"""

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        # Block-diagonal projection matrices P^s and P^t
        self.W_qs = nn.Linear(d_model, d_model)  # Seq->Struct query projection
        self.W_ks = nn.Linear(d_model, d_model)  # Seq->Struct key projection
        self.W_vs = nn.Linear(d_model, d_model)  # Seq->Struct value projection

        self.W_qt = nn.Linear(d_model, d_model)  # Struct->Seq query projection
        self.W_kt = nn.Linear(d_model, d_model)  # Struct->Seq key projection
        self.W_vt = nn.Linear(d_model, d_model)  # Struct->Seq value projection

        # Fusion layer
        self.fusion = nn.Linear(2 * d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, seq_features, struct_features):
        """
        Forward pass
        seq_features: sequence features (batch, seq_len, d_model)
        struct_features: structure features (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = seq_features.shape

        # Seq->Struct direction
        Q_s = self.W_qs(seq_features)  # (batch, seq_len, d_model)
        K_s = self.W_ks(struct_features)  # (batch, seq_len, d_model)
        V_s = self.W_vs(struct_features)  # (batch, seq_len, d_model)

        # Multi-head attention
        Q_s = Q_s.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K_s = K_s.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V_s = V_s.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores_s = torch.matmul(Q_s, K_s.transpose(-2, -1)) / self.scale
        attn_s = F.softmax(scores_s, dim=-1)
        attn_s = self.dropout(attn_s)

        Z_s = torch.matmul(attn_s, V_s)  # (batch, nhead, seq_len, head_dim)
        Z_s = Z_s.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Struct->Seq direction
        Q_t = self.W_qt(struct_features)  # (batch, seq_len, d_model)
        K_t = self.W_kt(seq_features)  # (batch, seq_len, d_model)
        V_t = self.W_vt(seq_features)  # (batch, seq_len, d_model)

        # Multi-head attention
        Q_t = Q_t.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        K_t = K_t.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)
        V_t = V_t.view(batch_size, seq_len, self.nhead, self.head_dim).transpose(1, 2)

        scores_t = torch.matmul(Q_t, K_t.transpose(-2, -1)) / self.scale
        attn_t = F.softmax(scores_t, dim=-1)
        attn_t = self.dropout(attn_t)

        Z_t = torch.matmul(attn_t, V_t)  # (batch, nhead, seq_len, head_dim)
        Z_t = Z_t.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Fuse outputs from both directions
        Z_combined = torch.cat([Z_s, Z_t], dim=-1)  # (batch, seq_len, 2*d_model)
        Z_cross = self.fusion(Z_combined)  # (batch, seq_len, d_model)

        return Z_cross


class GMCBindNet(nn.Module):
    """Backbone network"""

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

        # Unified dimension parameters
        self.d_cross = hidden_dim  # Unified dimension for cross-attention

        # Sequence feature processing - multi-scale window convolution
        self.mswc = MSWCLayer(
            in_channel=4,  # Four nucleotides: A/U/C/G
            out_channel=hidden_dim // 2,
            kernel_sizes=[1, 3, 5, 7, 9, 11, 13, 15],  # Multi-scale convolution kernels
            activation=nn.ReLU(),
            batch_norm=True,
            residual=True
        )

        # Structure feature processing - stacked GAT layers
        self.gat = self._build_gat_layers(
            in_dim=in_dim,
            out_dim=hidden_dim // 2,
            num_layers=self.num_gat_layers
        )

        # Projection layers: project sequence and structure features to unified dimension
        self.seq_proj = nn.Linear(hidden_dim // 2, self.d_cross)
        self.struct_proj = nn.Linear(hidden_dim // 2, self.d_cross)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=self.d_cross,
            max_len=seq_len * 2
        )

        # Bidirectional cross-attention layer
        self.bidirectional_cross_attn = BidirectionalCrossAttention(
            d_model=self.d_cross,
            nhead=4,  # Number of attention heads
            dropout=0.1
        )

        # Prediction module
        self.prediction = nn.Sequential(
            nn.Linear(self.d_cross, 256),  # Per-position prediction, no flattening
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, self.n_classes)
        )

    def _build_gat_layers(self, in_dim, out_dim, num_layers):
        """
        Build stacked GAT layers
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

        # Intermediate layers: hidden to hidden
        for _ in range(1, num_layers - 1):
            layers.append(GATLayer(
                in_dim=out_dim * 4,  # Dimension after concatenating multi-head outputs
                out_dim=out_dim,
                num_heads=4,
                activation=F.leaky_relu,
                dropout=0.1,
                residual=True
            ))

        # Final layer: hidden to output
        if num_layers > 1:
            layers.append(GATLayer(
                in_dim=out_dim * 4,  # Dimension after concatenating multi-head outputs
                out_dim=out_dim,
                num_heads=4,
                activation=F.leaky_relu,
                dropout=0.1,
                residual=True
            ))

        return layers

    def forward(self, g):
        """Forward pass"""
        # Get node features (sequence information)
        seq_features = g.ndata['feat']  # (total_nodes, 4)

        # Reshape to sequence format
        batch_size = g.batch_size
        seq_features = seq_features.view(batch_size, self.seq_len, 4)

        # Multi-scale window convolution for sequence features
        seq_input = seq_features.permute(0, 2, 1)  # (batch, 4, seq_len) for MSWC
        seq_emb = self.mswc(seq_input)  # (batch, hidden_dim//2, seq_len)
        seq_emb = seq_emb.permute(0, 2, 1)  # (batch, seq_len, hidden_dim//2)

        # GAT processing for structure features
        struct_emb = self._process_graph(g)  # (batch, seq_len, hidden_dim//2)

        # Project to unified dimension
        seq_proj = self.seq_proj(seq_emb)  # (batch, seq_len, d_cross)
        struct_proj = self.struct_proj(struct_emb)  # (batch, seq_len, d_cross)

        # Add positional encoding
        seq_pe = self.pos_encoder(seq_proj)  # (batch, seq_len, d_cross)
        struct_pe = self.pos_encoder(struct_proj)  # (batch, seq_len, d_cross)

        # Bidirectional cross-attention fusion
        fused_features = self.bidirectional_cross_attn(
            seq_pe,  # Sequence features
            struct_pe  # Structure features
        )  # (batch, seq_len, d_cross)

        # Binding site prediction (per-position prediction)
        pred = self.prediction(fused_features)  # (batch, seq_len, n_classes)

        if self.n_classes == 1:
            pred = pred.squeeze(-1)  # (batch, seq_len)

        return pred

    def _process_graph(self, g):
        """Process graph structure features - stacked GAT layers"""
        # Initial node features
        h = g.ndata['feat']

        # Stacked GAT layers
        for i, gat_layer in enumerate(self.gat):
            h = gat_layer(g, h)

            if self.batch_norm and i < len(self.gat) - 1:
                h = nn.BatchNorm1d(h.shape[1]).to(self.device)(h)

        # Reshape to sequence format
        batch_size = g.batch_size
        return h.view(batch_size, self.seq_len, -1)
