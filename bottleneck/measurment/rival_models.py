import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel, StopAtValAccCallback
from utils import get_args, create_model_dir, compute_energy, return_datasets
from models.graph_model import GraphModel
import argparse
from easydict import EasyDict
import random
import torch
from torch_geometric.data import Data
import numpy as np
import os
# Set random seed for reproducibility
import torch
import random
import numpy as np
from TMD.tmd import TMD
# Ensure PyTorch uses deterministic algorithms
import torch.nn as nn
import torch
from torch_geometric.nn.models import GIN, GAT, GCN
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv

class GraphLevelGAT(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Graph-level GCN model.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output (e.g., number of graph-level classes).
            num_layers (int): Number of GCN layers.
        """
        super(GraphLevelGAT, self).__init__()
        
        # Initialize GCN layers
        self.model = GAT(in_channels=input_dim,hidden_channels=hidden_dim,num_layers=num_layers)
        self.model = self.model.double()
        
    def forward(self, data):
        """
        Forward pass.

        Args:
            data (torch_geometric.data.Data): Graph data object with attributes `x`, `edge_index`, and `batch`
        Returns:
            torch.Tensor: Graph-level predictions.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.model(x = x, edge_index = edge_index, batch = batch)
        x = global_mean_pool(x, batch)
        return x

class GraphLevelGIN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Graph-level GCN model.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output (e.g., number of graph-level classes).
            num_layers (int): Number of GCN layers.
        """
        super(GraphLevelGIN, self).__init__()
        
        # Initialize GCN layers
        self.model = GIN(in_channels=input_dim,hidden_channels=hidden_dim,num_layers=num_layers)
        self.model = self.model.double()
        
    def forward(self, data):
        """
        Forward pass.

        Args:
            data (torch_geometric.data.Data): Graph data object with attributes `x`, `edge_index`, and `batch`
        Returns:
            torch.Tensor: Graph-level predictions.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.model(x = x, edge_index = edge_index, batch = batch)
        x = global_mean_pool(x, batch)
        # print(x.size())
        return x

class GraphLevelGCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        Graph-level GCN model.

        Args:
            input_dim (int): Dimension of input node features.
            hidden_dim (int): Dimension of hidden layers.
            output_dim (int): Dimension of the output (e.g., number of graph-level classes).
            num_layers (int): Number of GCN layers.
        """
        super(GraphLevelGCN, self).__init__()
        
        # Initialize GCN layers
        self.model = GCN(in_channels=input_dim,hidden_channels=hidden_dim,num_layers=num_layers)
        self.model = self.model.double()
        
    def forward(self, data):
        """
        Forward pass.

        Args:
            data (torch_geometric.data.Data): Graph data object with attributes `x`, `edge_index`, and `batch`
        Returns:
            torch.Tensor: Graph-level predictions.
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.model(x = x, edge_index = edge_index, batch = batch)
        x = global_mean_pool(x, batch)
        # print(x.size())
        return x