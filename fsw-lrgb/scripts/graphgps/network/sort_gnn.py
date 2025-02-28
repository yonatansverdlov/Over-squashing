import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from graphgps.layer.combine import LinearCombination, LTSum, Concat, ConcatProject
from torch_geometric.nn.norm import LayerNorm, BatchNorm
combine_dict = {'LinearCombination': LinearCombination, 'LTSum': LTSum, 'Concat': Concat, 'ConcatProject': ConcatProject}
norm_dict = {'layer': LayerNorm, 'batch': BatchNorm, 'none': torch.nn.Identity}
from graphgps.layer.sort_conv_layer import SortConv
import torch.nn as nn
import torch.nn.functional as F

def setup_learnable_blank_vector(in_dim, embed_dim, num_layers):
    blank_vectors = nn.ParameterList([nn.Parameter(torch.zeros(in_dim))])
    for _ in range(num_layers):
        blank_vectors.append(nn.Parameter(torch.zeros(embed_dim)))

    return blank_vectors


@register_network('sort_gnn')
class SortMPNN(torch.nn.Module):
    '''
    MPNN that uses sort as non-linearity.
    The MPNN optionally maintains an additional blank tree vector which augments node neighborhoods.
    '''
    def __init__(self, dim_in, dim_out):
        super().__init__()
        embed_dim = cfg.gnn.dim_inner
        num_layers = cfg.gnn.layers_mp
        combine = combine_dict[cfg.gnn.combine]
        update_w_orig = cfg.gnn.update_w_orig
        redisual = cfg.gnn.residual
        self.dropout = cfg.gnn.dropout
        norm = cfg.gnn.norm
        blank_vector_method = cfg.gnn.blank_vector_method
        self.update_w_orig = update_w_orig
        self.skip_connections = redisual
        

        if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')
        
        self.node_encoder = FeatureEncoder(dim_in)
        dim_in = self.node_encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."
        
        assert blank_vector_method in ['iterative_update', 'zero', 'learnable'], \
                                      'blank_vector_method must be one of [iterative_update, zero, learnable]'
        self.blank_vector_method = blank_vector_method
        if self.blank_vector_method == 'learnable':
          self.blank_vectors = setup_learnable_blank_vector(dim_in, embed_dim, num_layers)

        
        first_combine = LTSum if (combine == LinearCombination and dim_in!=embed_dim)  else combine
        orig_dim = dim_in if update_w_orig else None
        conv1 = SortConv(dim_in, dim_in, orig_dim=orig_dim, combine=first_combine)
        layers = [conv1]
        for _ in range(cfg.gnn.layers_mp-1):
            layers.append(SortConv(dim_in,
                                     dim_in, orig_dim=orig_dim))
        self.convs = torch.nn.Sequential(*layers)
        self.norms = nn.ModuleList()
        for i in range(num_layers):
          self.norms.append(norm_dict[norm](embed_dim))
        
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)



    def forward(self, batch):
        self.node_encoder(batch)
        x ,edge_index = batch.x, batch.edge_index
        
        orig_x = x if self.update_w_orig else None
        if self.blank_vector_method == 'learnable':
          blank_vec = self.blank_vectors[0].to(x.device)
        else:
          blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)
        orig_blank_vec = blank_vec if self.update_w_orig else None

        for i, conv in enumerate(self.convs):
            if self.blank_vector_method == 'learnable':
              blank_vec = self.blank_vectors[i].to(x.device)
            elif self.blank_vector_method == 'zero':
              blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)
            
            if self.skip_connections:
               res_x , res_blank_vec = x, blank_vec
            edge_attrs = None

            x, blank_vec = conv(x, edge_index, blank_vec, orig_x=orig_x, orig_blank_vec=orig_blank_vec, edge_attrs=edge_attrs)
            
                        
            # apply dropout to both x and blank_vec
            x = F.dropout(self.norms[i](x), p=self.dropout, training=self.training)
            blank_vec = F.dropout(blank_vec, p=self.dropout, training=self.training)

            if self.skip_connections and i > 0:
              x = x + res_x
              blank_vec = blank_vec + res_blank_vec
        
        batch.x = x
        if cfg.gnn.head == 'mlp_graph':
          out = self.post_mp(batch)
        else:
          if self.blank_vector_method == 'learnable':
            blank_vec = self.blank_vectors[-1].to(x.device)
          elif self.blank_vector_method == 'zero':
            blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)
          out = self.post_mp(batch, blank_vec)
        return out