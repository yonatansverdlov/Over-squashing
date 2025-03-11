import torch.nn as nn
from torch_geometric.data import Data
from easydict import EasyDict
from utils import get_layer
from fsw.fsw_layer import FSW_readout
from torch_geometric.nn import global_mean_pool
import torch

class GraphModel(nn.Module):
    """
    A customizable graph neural network model designed for various graph tasks.
    Includes support for tasks requiring tree-specific embeddings, residual connections,
    layer normalization, and single/multi-graph datasets.

    Args:
        args (EasyDict): Configuration dictionary with model parameters.
    """
    def __init__(self, args: EasyDict):
        super().__init__()
        dtype = getattr(torch, args.dtype)
        # Model configuration
        self.use_layer_norm = args.use_layer_norm
        self.use_residual = args.use_residual
        self.num_layers = args.num_layers
        self.h_dim = args.dim
        self.out_dim = args.out_dim
        self.task_type = args.task_type
        self.gnn_type = args.gnn_type

        # Dataset-specific configuration
        self.need_encode_value = self.task_type in {'Tree','one_radius'}
        # Embedding initialization          
        self.embed_label = nn.Linear(args.in_dim, self.h_dim, dtype=dtype,bias=False)
        self.embed_value = (
            nn.Embedding(args.in_dim, self.h_dim, dtype=dtype)
            if self.need_encode_value else None
        )

        # Model layers and normalization
        self.layers = nn.ModuleList([
            get_layer(in_dim=self.h_dim, out_dim=self.h_dim, args=args)
            for _ in range(self.num_layers)
        ])
        self.layer_norms = (
            nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_layers)])
            if self.use_layer_norm else None
        )
        
        # Output layer setup
        self.out_layer = nn.Linear(self.h_dim, self.out_dim)
        # Initialize model parameters
        self.init_model()

    def init_model(self):
        """
        Initialize model parameters using Xavier (Glorot) uniform initialization.
        Ensures stable gradient flow at the start of training.
        """
        nn.init.xavier_uniform_(self.out_layer.weight)
        nn.init.normal_(self.embed_label.weight, mean=0, std=1)
        # embedding_dim = self.h_dim
        # nn.init.uniform_(self.embed_label.weight, -1.0 / embedding_dim**0.5, 1.0 / embedding_dim**0.5)
            
    def forward(self, data: Data):
        """
        Forward pass through the graph model.

        Args:
            data (Data): A Torch Geometric Data object containing node features, edges, and other attributes.

        Returns:
            torch.Tensor: Predictions for nodes (or graphs, depending on task).
        """
        x = self.compute_node_embedding(data)
        return self.out_layer(x)[data.root_mask]  # Filter predictions for root nodes

    def compute_node_embedding(self, data: Data):
        """
        Compute node embeddings using the embedding layers, graph layers, activations,
        residual connections, and optional layer normalization.

        Args:
            data (Data): A Torch Geometric Data object containing x, edge_index, and optionally edge_attr.

        Returns:
            torch.Tensor: Node embeddings of shape (num_nodes, h_dim).
        """
        x, edge_index = data.x, data.edge_index
        x.requires_grad_()
        # Node feature embedding
        if self.need_encode_value:
            x = self.embed_label(x[:, 0]) + self.embed_value(x[:, 1])
        else:
            x = self.embed_label(x)

        # Graph layers
        for i, layer in enumerate(self.layers):
            new_x = x
            new_x = layer(new_x, edge_index)

            # Residual connections
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x

            # Layer normalization
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        return x
    