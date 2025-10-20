import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPReadout(nn.Module):
    """
    MLP Readout Layer for binding site prediction
    
    Args:
        input_dim: Input dimension (feature dimension from cross-attention output)
        output_dim: Output dimension (binding site prediction dimension)
        hidden_dim: Hidden layer dimension (default: 256)
        dropout: Dropout probability (default: 0.2)
        num_layers: Number of MLP layers
    """

    def __init__(self, input_dim, output_dim, hidden_dim=256, dropout=0.2, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # First layer
        self.linear1 = nn.Linear(input_dim, hidden_dim)

        # Second layer
        self.linear2 = nn.Linear(hidden_dim, output_dim)

        # Activation and regularization
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()  # Final output uses sigmoid activation

        # Batch normalization for training stability
        self.bn = nn.BatchNorm1d(hidden_dim)

        # Residual connection support
        self.use_residual = (input_dim == hidden_dim)
        if self.use_residual:
            self.residual = nn.Identity()

    def forward(self, Z_cross):
        """
        Forward pass
        
        Args:
            Z_cross: Input features from cross-attention layer
            
        Returns:
            P_bind: Predicted binding probabilities
        """
        # Handle 3D input (preserving sequence position information)
        if Z_cross.dim() == 3:
            batch_size, seq_len, D_cross = Z_cross.shape
            # Transpose operation Z^T [batch_size, D_cross, seq_len]
            Z = Z_cross.permute(0, 2, 1)
            # Flatten to 2D [batch_size, D_cross * seq_len]
            Z = Z.reshape(batch_size, -1)
        else:
            Z = Z_cross

        # First layer
        h = self.linear1(Z)
        h = self.bn(h)  # Batch normalization
        h = self.activation(h)
        h = self.dropout(h)

        # Residual connection
        if self.use_residual:
            h = h + self.residual(Z)

        # Second layer
        logits = self.linear2(h)

        # Final output
        P_bind = self.sigmoid(logits)
        return P_bind

    def __repr__(self):
        return (f"{self.__class__.__name__}(input_dim={self.linear1.in_features}, "
                f"hidden_dim={self.linear1.out_features}, "
                f"output_dim={self.linear2.out_features}, "
                f"residual={self.use_residual})")
