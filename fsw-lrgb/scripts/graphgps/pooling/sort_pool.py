import torch
from torch_geometric.utils import to_dense_batch
import os
import sys
from torch_geometric.graphgym.register import register_pooling
from torch_geometric.graphgym.config import cfg
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from graphgps.layer.sort_conv_layer import SortConv





@register_pooling('sort_global')
class SortGlobalConv(SortConv):
    def __init__(self, dim_in, dim_out):
        """
        :param dim_in: dimension of input node feature vectors
        :param dim_out: dimension of global graph output vector
        :param max_nodes: max number of vertices in graph across all dataset
        """
        super().__init__(dim_in, dim_out, max_nodes=cfg.dataset.max_nodes)

    def convert_to_dense(self, inputs, index, indices, num_indices):
       return to_dense_batch(inputs[indices], index[indices],
                                    max_num_nodes=self.max_nodes, batch_size=num_indices)

    def return_collapsed_output(self, sorted):
      # collapse
      if self.collapse_method == 'vector':
        out = self.lin_collapse(sorted.permute(0,2,1)).squeeze(-1) # [Num_nodes("batch"), num_neighbors("max_nodes"), dim_out] --> [Num_nodes("batch"), dim_out]
      elif self.collapse_method == 'matrix':
        out = sorted * self.lin_collapse
        out = torch.sum(out, dim=1).squeeze(1)
      else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")
    
      return out
    
    def replace_indices_(self, index):
      with torch.no_grad():
        replacements = torch.stack([self.first_node_index, torch.argsort(self.first_node_index)]).T
        mask = (index == replacements[:, :1])
        index = (1 - mask.sum(dim=0)) * index + (mask * replacements[:,1:]).sum(dim=0)
      return index

    def update(self, aggr_out, index, batch):
        # return the aggregated out
        return aggr_out
    

    def forward(self, x, blank_vec, batch=None):
        # create batch numbering for single graph if needed
        if batch is None:
            batch = torch.zeros(x.shape[0]).long().to(x.device)
          
        self.num_indices = batch.max().item() + 1

        # num nodes per graph in batch
        num_nodes = (torch.unique(batch, return_counts=True)[1]).to(x.device)
        # index of first node per graph
        first_node_index = num_nodes.cumsum(dim=0)
        self.first_node_index = torch.cat([torch.tensor([0], device=x.device), first_node_index], dim=0)[:-1]
        self.max_index = first_node_index[-1].item()
        # setting edge index so all nodes are connected to the first node per graph
        src = torch.arange(x.shape[0], device=x.device)
        dst = torch.cat([torch.ones(num_nodes[i], device=x.device)*self.first_node_index[i] for i in range(0, len(num_nodes))], dim=0)
        edge_index = torch.stack([src, dst], dim=0).long().to(x.device)

        x = self.propagate(edge_index, x=x, blank_vec=blank_vec, batch=batch)
        return x
    
