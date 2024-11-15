import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_layer
from easydict import EasyDict
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool

from torch_geometric.nn.conv import MessagePassing

class GraphModel(nn.Module):
    """
    A customizable graph neural network model with optional tree-specific embeddings, residual connections, 
    and layer normalization. This model is designed for graph tasks in the Torch Geometric framework.
    
    Args:
    - args (EasyDict): A configuration dictionary containing model parameters.
    - dtype (torch.dtype): Data type for embedding layers. Default is torch.float32.
    """
    def __init__(self, args: EasyDict, dtype=torch.float32):
        super().__init__()
        self.use_layer_norm = args.use_layer_norm
        self.use_activation = args.use_activation
        self.use_residual = args.use_residual
        self.num_layers = args.depth
        self.h_dim = args.dim
        self.task_type = args.task_type
        self.single_graph_datasets = {'Cora', 'Actor', 'Corn', 'Texas', 'Wisc', 'Squir', 'Cham', 'Cite', 'Pubm','MUTAG','PROTEIN'}
        self.is_real = self.task_type in self.single_graph_datasets
        self.is_tree = self.task_type == 'Tree'
        self.is_mutag = self.task_type in ['MUTAG']
        
        # Embedding initialization for label and value embeddings based on task type
        self.embed_label = (nn.Linear(args.in_dim, self.h_dim) if self.is_real
                            else nn.Embedding(args.in_dim, self.h_dim, dtype=dtype))
        self.embed_value = (nn.Embedding(args.in_dim, self.h_dim, dtype=dtype) if self.is_tree else None)

        if self.is_mutag:
            self.embed_egde = nn.Linear(4, self.h_dim)

        # Initialize model layers and optional layer normalizations
        self.layers = nn.ModuleList([get_layer(args,self.h_dim, self.h_dim) for _ in range(self.num_layers)])
        self.layer_norms = (nn.ModuleList([nn.LayerNorm(self.h_dim) for _ in range(self.num_layers)])
                            if self.use_layer_norm else None)
        
        # Output layer to map to the final output dimension
        self.out_layer = nn.Linear(self.h_dim, args.out_dim)
        self.init_model()

    def init_model(self):
        """
        Initializes model parameters using Xavier (Glorot) uniform initialization for the output layer.
        This ensures stable gradients in the output layer at the start of training.
        """
        if not self.is_mutag:
            nn.init.xavier_uniform_(self.out_layer.weight)
        else:
            nn.init.xavier_uniform_(self.embed_egde.weight)

    def forward(self, data: Data):
        """
        Forward pass through the graph model.
        
        Args:
        - data (Data): A Torch Geometric `Data` object containing graph node features and edges.
        
        Returns:
        - torch.Tensor: Model predictions for nodes indicated by the root mask in `data`.
        """
        x = self.compute_node_embedding(data)
        if self.is_mutag or self.task_type == 'PROTEIN':
           return global_mean_pool(x,data.batch)
        else:
          return self.out_layer(x)[data.root_mask]

    def compute_node_embedding(self, data: Data):
        """
        Computes node embeddings by applying embedding layers, graph layers, activations, residual connections,
        and layer normalization as specified by the configuration.
        
        Args:
        - data (Data): A Torch Geometric `Data` object containing node features (`x`) and edge indices (`edge_index`).
        
        Returns:
        - torch.Tensor: Node embeddings of shape `(num_nodes, h_dim)`, where `h_dim` is the hidden dimension.
        """
        x, edge_index,edge_attr = data.x, data.edge_index, None

        # Tree-specific embedding combining label and value embeddings; otherwise, standard label embedding
        if self.is_tree:
            x = self.embed_label(x[:, 0]) + self.embed_value(x[:, 1])
        else:
            x = self.embed_label(x)

        for i, layer in enumerate(self.layers):
            new_x = x
            if self.is_mutag:
                edge_attr = self.embed_egde(data.edge_attr)
            new_x = layer(new_x, edge_index, edge_attr)
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)

        return x

