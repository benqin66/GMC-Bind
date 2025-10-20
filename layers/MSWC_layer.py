import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import GraphConv
import random
import dgl
import os
import numpy as np

def seed_set(seed=None):
    """Set all random seeds for reproducibility"""
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True
    dgl.random.seed(seed)

seed_set(20001016)

class ListReadoutLayer(nn.Module):
    """List Readout Layer for sequence feature reshaping"""
    def __init__(self, in_channel, window_size=501):
        super().__init__()
        self.in_channel = in_channel
        self.window_size = window_size

    def forward(self, g, feat):
        output = torch.reshape(feat, (feat.shape[0]//self.window_size, self.window_size, feat.shape[1]))
        output =  torch.flatten(output, start_dim=1)
        return output


class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer combining 1D convolution and graph convolution
    """
    def __init__(self, in_channel, out_channel, kernel_size, activation, batch_norm,
                 dropout=False, residual=False, window_size=501):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.batch_norm = batch_norm
        self.residual = residual
        self.window_size = window_size

        if in_channel != out_channel:
            self.residual = False

        self.batchnorm_h = nn.BatchNorm1d(out_channel)
        self.activation = activation

        self.conv_layer = ConvReadoutLayer(in_channel, out_channel, kernel_size, padding=kernel_size[0] // 2, stride=1)
        self.graph_conv_layer = GraphConv(out_channel, out_channel)

    def forward(self, g, feature):
        h_in = feature  # Save input for residual connection

        h = self.conv_layer(g, feature)
        h = self.graph_conv_layer(g, h)

        if self.batch_norm:
            h = self.batchnorm_h(h)  # Apply batch normalization

        if self.activation:
            h = self.activation(h)

        if self.residual:
            h = h_in + h  # Apply residual connection

        return h

    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, residual={})'.format(self.__class__.__name__,
                                                                         self.in_channel,
                                                                         self.out_channel, self.residual)


class MSWCLayer(nn.Module):
    """
    Multi-Scale Window Convolution Layer for multi-scale sequence feature extraction
    """

    def __init__(self, in_channel, out_channel, kernel_sizes, activation=nn.ReLU(),
                 batch_norm=True, residual=False):
        """
        Args:
            in_channel: Input channels (fixed to 4 for A/U/C/G nucleotides)
            out_channel: Output channels per convolution path
            kernel_sizes: List of convolution kernel sizes, e.g., [1, 3, 5, ..., k]
            activation: Activation function (default: ReLU)
            batch_norm: Whether to use batch normalization
            residual: Whether to use residual connections
        """
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_sizes = kernel_sizes
        self.batch_norm = batch_norm
        self.residual = residual
        self.activation = activation

        # Create multiple parallel convolution paths
        self.conv_paths = nn.ModuleList()
        for ksize in kernel_sizes:
            conv = nn.Conv1d(in_channel, out_channel, kernel_size=ksize, padding=ksize // 2)
            if batch_norm:
                bn = nn.BatchNorm1d(out_channel)
                path = nn.Sequential(conv, bn)
            else:
                path = conv
            self.conv_paths.append(path)

        # Projection layer (1x1 convolution for dimensionality reduction)
        total_channels = len(kernel_sizes) * out_channel
        self.projection = nn.Conv1d(total_channels, out_channel, kernel_size=1)

        # Dimension matching for residual connection
        if residual and in_channel != out_channel:
            self.residual = False

    def forward(self, x):
        """Forward pass"""
        # Save input for residual connection
        h_in = x if self.residual else None

        # Parallel processing through all convolution paths
        conv_outputs = []
        for conv_path in self.conv_paths:
            # Convolution operation
            h = conv_path(x)
            # Activation function
            if self.activation:
                h = self.activation(h)
            conv_outputs.append(h)

        # Concatenate all path outputs (along channel dimension)
        h_concat = torch.cat(conv_outputs, dim=1)

        # Projection for dimensionality reduction
        h_out = self.projection(h_concat)

        # Activation function
        if self.activation:
            h_out = self.activation(h_out)

        # Residual connection
        if self.residual:
            h_out = h_in + h_out

        return h_out

    def __repr__(self):
        return (f'{self.__class__.__name__}(in_channels={self.in_channel}, '
                f'out_channels={self.out_channel}, kernel_sizes={self.kernel_sizes}, '
                f'residual={self.residual})')


class MAXPoolLayer(nn.Module):
    """MAXPool layer for 2D max pooling operations"""
    def __init__(self, kernel_size, stride, padding=0, **kwargs):
        super().__init__()
        self.ksize = kernel_size
        self.stride = stride
        self.padding = padding

        self.pooling = nn.MaxPool2d(kernel_size, stride, padding=padding)

    def forward(self, inputs):
        output = self.pooling(inputs)
        return output
