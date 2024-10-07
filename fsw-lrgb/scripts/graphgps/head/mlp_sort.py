import torch.nn as nn
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym import cfg
from torch_geometric.graphgym.register import register_head
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graphgps.pooling.sort_pool import SortGlobalConv



@register_head('mlp_sort')
class MLPSortHead(nn.Module):
    """
    MLP prediction head for graph prediction tasks.

    Args:
        dim_in (int): Input dimension.
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
        L (int): Number of hidden layers.
    """

    def __init__(self, dim_in, dim_out):
        super().__init__()
        assert cfg.model.graph_pooling == 'sort_global'
        self.global_sort = SortGlobalConv(dim_in, dim_in)

        dropout = cfg.gnn.dropout
        L = cfg.gnn.layers_post_mp

        layers = []
        for _ in range(L-1):
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(dim_in, dim_in, bias=True))
            layers.append(register.act_dict[cfg.gnn.act]())

        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dim_in, dim_out, bias=True))
        self.mlp = nn.Sequential(*layers)

    def _scale_and_shift(self, x):
        return x

    def _apply_index(self, batch):
        return batch.graph_feature, batch.y

    def forward(self, batch, blank_vec):
        x = self.global_sort(batch.x, blank_vec, batch.batch)
        pred = self.mlp(x)
        pred = self._scale_and_shift(pred)

        return pred, batch.y

