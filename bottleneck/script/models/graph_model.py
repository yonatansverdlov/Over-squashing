import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import get_layer

class GraphModel(torch.nn.Module):
    def __init__(self, args, dtype=torch.float32):
        super(GraphModel, self).__init__()
        self.layer_type = args.layer_type
        self.gnn_type = args.gnn_type
        self.use_layer_norm = args.use_layer_norm
        self.use_activation = args.use_activation
        self.use_residual = args.use_residual
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_layers = args.depth
        self.in_dim = args.in_dim
        self.out_dim = args.out_dim
        self.h_dim = args.dim
        self.task_type = args.task_type
        self.layer0_keys = nn.Embedding(num_embeddings=self.in_dim + 1, embedding_dim=self.h_dim, dtype=dtype,)
        self.layer0_values = nn.Embedding(num_embeddings=self.in_dim + 1, embedding_dim=self.h_dim, dtype=dtype)
        self.layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.single_graph = ['Cora','Actor','Corn','Texas','Wisc','Squir','Cham','Cite','Pubm']
        args.single_graph = self.single_graph
        self.global_task = self.task_type in self.single_graph
        self.is_tree = str(self.task_type) == 'Tree'
        if self.global_task:
            self.layer0_keys = nn.Linear(self.in_dim,self.h_dim)
        for i in range(self.num_layers):
                self.layers.append(get_layer(
                    in_dim=self.h_dim,
                    out_dim =self.h_dim,
                    args=args))
        if self.use_layer_norm:
            for i in range(self.num_layers):
                self.layer_norms.append(nn.LayerNorm(self.h_dim))

        self.out_layer =  nn.Linear(in_features=self.h_dim, out_features=self.out_dim, bias=False)
        self.init_model()

    def init_model(self):
        torch.nn.init.xavier_uniform_(self.out_layer.weight)

    def forward(self, data):
        x, edge_index, batch, roots = data.x, data.edge_index, data.batch, data.root_mask
        if self.is_tree:
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self.layer0_keys(x_key)
            x_val_embed = self.layer0_values(x_val)
            x = x_key_embed + x_val_embed
        else:
            x = self.layer0_keys(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            new_x = x
            new_x = layer(new_x, edge_index)
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
        preds = self.out_layer(x)[roots]
        return preds
    
    def compute_node_embedding(self,data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        # TODO - change it.
        if self.is_tree:
            x_key, x_val = x[:, 0], x[:, 1]
            x_key_embed = self.layer0_keys(x_key)
            x_val_embed = self.layer0_values(x_val)
            x = x_key_embed + x_val_embed
        else:
            x = self.layer0_keys(x)

        for i in range(self.num_layers):
            layer = self.layers[i]
            new_x = x
            new_x = layer(new_x, edge_index)
            if self.use_activation:
                new_x = F.relu(new_x)
            if self.use_residual:
                x = x + new_x
            else:
                x = new_x
            if self.use_layer_norm:
                x = self.layer_norms[i](x)
        return x



