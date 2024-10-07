import torch
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.models.gnn import FeatureEncoder, GNNPreMP
from torch_geometric.graphgym.register import register_network
from torch_geometric.nn.norm import LayerNorm, BatchNorm
from graphgps.layer.fsw_conv import FSW_conv
import torch.nn as nn
import torch.nn.functional as F


@register_network('sw_gnn')
class SWGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        dim_inner = cfg.gnn.dim_inner
        sw_embed_dim = cfg.gnn.sw_embed_dim
        learnable_embedding = cfg.gnn.learnable_embedding
        concat_self = cfg.gnn.concat_self
        bias = cfg.gnn.bias
        conv_mlp_layers = cfg.gnn.conv_mlp_layers
        conv_mlp_hidden_dim = cfg.gnn.conv_mlp_hidden_dim
        conv_mlp_activation_final = register.act_dict[cfg.gnn.conv_mlp_activation_final]()
        conv_mlp_activation_hidden = register.act_dict[cfg.gnn.conv_mlp_activation_hidden]()
        mlp_init = cfg.gnn.mlp_init
        batchnorm_final = cfg.gnn.batchnorm_final
        batchnorm_hidden = cfg.gnn.batchnorm_hidden
        dropout_final = cfg.gnn.dropout_final
        dropout_hidden = cfg.gnn.dropout_hidden
        edge_weighting = cfg.gnn.edge_weighting
        self_loop_weight = cfg.gnn.self_loop_weight
        homog_degree_encoding = cfg.gnn.homog_degree_encoding
        vertex_degree_pad_thresh = cfg.gnn.vertex_degree_pad_thresh
        
        self.skip_connections = cfg.gnn.residual
        

        self.node_encoder = FeatureEncoder(dim_in)
        dim_in = self.node_encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp = GNNPreMP(
                dim_in, cfg.gnn.dim_inner, cfg.gnn.layers_pre_mp)
            dim_in = cfg.gnn.dim_inner

        assert cfg.gnn.dim_inner == dim_in, \
            "The inner and hidden dims must match."
        
        conv1 = FSW_conv(in_channels=dim_in, out_channels=dim_inner, embed_dim=sw_embed_dim, 
                       learnable_embedding=learnable_embedding, concat_self=concat_self, bias=bias, 
                       mlp_layers=conv_mlp_layers, mlp_hidden_dim=conv_mlp_hidden_dim, 
                       mlp_activation_final=conv_mlp_activation_final, mlp_activation_hidden=conv_mlp_activation_hidden,
                       mlp_init=mlp_init, batchNorm_final=batchnorm_final, batchNorm_hidden=batchnorm_hidden,
                        dropout_final=dropout_final, dropout_hidden=dropout_hidden, self_loop_weight=self_loop_weight,
                        edge_weighting=edge_weighting,edgefeat_dim = 0,vertex_degree_pad_thresh = vertex_degree_pad_thresh,homog_degree_encoding =homog_degree_encoding )
        layers = [conv1]
        for _ in range(cfg.gnn.layers_mp-1):
            layers.append(FSW_conv(in_channels=dim_inner, out_channels=dim_inner, embed_dim=sw_embed_dim,
                                 learnable_embedding=learnable_embedding, concat_self=concat_self, bias=bias, 
                                 mlp_layers=conv_mlp_layers, mlp_hidden_dim=conv_mlp_hidden_dim, 
                                 mlp_activation_final=conv_mlp_activation_final, mlp_activation_hidden=conv_mlp_activation_hidden,
                                 mlp_init=mlp_init, batchNorm_final=batchnorm_final, batchNorm_hidden=batchnorm_hidden,
                                 dropout_final=dropout_final, dropout_hidden=dropout_hidden, self_loop_weight=self_loop_weight,
                                 edge_weighting=edge_weighting,edgefeat_dim = 0,vertex_degree_pad_thresh = vertex_degree_pad_thresh,homog_degree_encoding =homog_degree_encoding ))
            
        self.convs = torch.nn.Sequential(*layers)
        
        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_in=cfg.gnn.dim_inner, dim_out=dim_out)
        # self.learnable_param = nn.Parameter(torch.empty(27, 64))
        # Initialize the parameter with Xavier initialization (uniform)
        # nn.init.xavier_uniform_(self.learnable_param)



    def forward(self, batch):
        self.node_encoder(batch)
        x ,edge_index,edge_attr  = batch.x, batch.edge_index,batch.edge_attr
        # Make one-hot.
        # edge_attr = torch.nn.functional.one_hot(edge_attr,9).view(-1,27).float()
        
        # edge_attr = edge_attr @ self.learnable_param
        
        for i, conv in enumerate(self.convs):
            
            if self.skip_connections:
               res_x = x
            
            x = conv(x, edge_index)
            
                        
            if self.skip_connections and i > 0:
              x = x + res_x
        
        batch.x = x
        if cfg.gnn.head == 'mlp_graph':
          out = self.post_mp(batch)
        # else:
        #   if self.blank_vector_method == 'learnable':
        #     blank_vec = self.blank_vectors[-1].to(x.device)
        #   elif self.blank_vector_method == 'zero':
        #     blank_vec = torch.zeros(x.shape[1], requires_grad=False).to(x.device)
        #   out = self.post_mp(batch, blank_vec)
        return out