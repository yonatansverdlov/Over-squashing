import numpy as np

import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_layer
import sys
import os
import importlib.util

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

mydir = os.path.dirname(os.path.abspath(__file__))
fsw_embedding_path = os.path.join(mydir, 'fsw_embedding.py')

# The following is equivalent to:
#from fsw_embedding import FSW_embedding, minimize_mutual_coherence, ag, sp

# Load the module from the file path
spec = importlib.util.spec_from_file_location("fsw_embedding", fsw_embedding_path)
fsw_embedding = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fsw_embedding)

# Access the items directly from the imported module
FSW_embedding = fsw_embedding.FSW_embedding
minimize_mutual_coherence = fsw_embedding.minimize_mutual_coherence
ag = fsw_embedding.ag
sp = fsw_embedding.sp

# Now you can use FSW_embedding, minimize_mutual_coherence, ag, and sp directly

# Release notes:
#
# 2024-09-03
# - Added support for edge features.
#   To use, set edgefeat_dim > 0 at init, and pass argument X_edge in forward(),
#   where X_edge is a tensor of size (num of edges, edgefeat_dim)
# - Elegant handling of empty neighborhoods => No need to use self_loop_weight > 0 anymore.
# - Added built-in support to make the whole model homogeneous w.r.t. the vertex- and edge-features.
#   To make the model homogeneous: (1) use homog_degree_encoding = True, (2) use bias=False, and
#   (3) use only homogeneous activations, namely ReLU or Leaky ReLU.
#   Use this when the function to be learned is known to be homogeneous.
# - If the function to be learned is known to be invariant to vertex degrees, set 
#   encode_vertex_degrees=False.


def return_act(act: str):
    if act == 'relu':
        return torch.nn.ReLU()
    if act == 'leaky':
        return torch.nn.LeakyReLU(0.1)
    if act == 'tanh':
        return torch.nn.Tanh()
    if act == 'silu':
        return torch.nn.SiLU()
    if act == 'gelu':
        return torch.nn.GELU()
    else:
        raise NotImplemented

@register_layer('fsw_conv')
class SW_conv(MessagePassing):
    # in_channels:    dimension of input vertex features
    #
    # out_channels:   dimension of output vertex features
    #
    # edgefeat_dim:   input edge-feature dimension
    #
    # embed_dim:      output dimension of the SW-embedding of neighboring vertex features.
    #                 if <mlp_layers> == 0 and <concat_self> == False, this argument is forced to equal out_channels.
    #                 Default: 2*max(<in_channels>, <out_channels>)  (chosen heuristically)
    #
    # learnable_embedding: tells whether the SW-embedding parameters (slices, frequency) are learnable or fixed.
    #                      default: True
    #
    # encode_vertex_degrees: tells whether to encode the in-degree of each vertex as part of its neighborhood embedding.
    #                        better keep this on unless the learning task involved is known to be degree-invariant.
    #                        default: True
    #
    # vertex_degree_encoding_function: tells what function to apply to vertex degrees before encoding. options:
    #                                  'identity': f(x)=x
    #                                  'sqrt': f(x) = sqrt(1+x)-1
    #                                  'log': f(x) = log(1+x)
    #                                  default: 'identity'
    #
    # homog_degree_encoding: tells whether the neighborhood-size encoding should be homogeneous.
    #                        better keep this off unless it is desired that the model should be homogeneous.
    #                        NOTE: To make the whole model homogeneous, make sure bias=False and all activations are
    #                              homogeneous (e.g. leaky ReLU).
    #                        default: False
    #
    # concat_self: when set to True, the embedding of each vertex's neighborhood is concatenated to its own
    #              feature vector before it is passed to the MLP. If <mlp_layers> = 0, then dimensionality reduction
    #              is applied after the concatenation to yield a vector of dimension <out_channels>.
    #              note that when set to False, the network is not fully expressive.
    #              Default: True
    #                   
    # self_loop_weight: NOTE: There is no need to use this anymore, since the new FSW embedding can elegantly handle empty sets.
    #                         Better use the default self_loop_weight = 0
    #                   when set to a positive number, a self loop with this weight is added to each vertex.
    #                   this is necessary in order to avoid empty neighborhoods (since the SW embedding is
    #                   not defiend for empty sets).
    #                   if set to zero and an empty neighborhood is encountered, this will lead to a runtime error.
    #                   Default: 0
    #
    # edge_weighting:   weighting scheme for the graph edges
    #                   options: 'unit': all edges get unit weight; 'gcn': weighting according to the GCN scheme; see [a]
    #                   Default: 'unit'
    #                   [a] https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.GCNConv.html
    #
    # bias:           if set to true, the MLP uses a bias vector.
    #                 to make the model scale equivariant (i.e. positively homogeneous) with respect to the vertex features, set 
    #                 all of <bias>, <batchNorm_final>, <batchNorm_hidden> to False, and use a scale-equivariant activation
    #                 (e.g. ReLU, or the Leaky ReLU used by deafault).
    #                 Default: True
    #
    # mlp_layers:     number of MLP layers to apply to the embedding that aggregates neighbor vertex features.
    #                 0 - do not apply an MLP; instead pass the embedded neighborhood directly as the output vertex features
    #                 Default: 1
    #
    # mlp_hidden_dim: dimension of the hidden layers in the MLP. only takes effect if mlp_layers > 1.
    #                 Default: max(in_channels, out_channels)  (chosen heuristically)
    #
    # mlp_activation_final, mlp_activation_hidden:
    #                 activation function to be used at the output of the final and hidden MLP layers.
    #                 if set to None, does not apply an activation
    #                 Defaults: Leaky ReLU with a negative slope of 0.2
    #
    # mlp_init:       initialization scheme for the MLP's linear layers. 
    #                 options: None / 'xavier_uniform' / 'xavier_normal' / 'kaiming_normal' / 'kaiming_uniform'
    #                 takes effect only if mlp_layers > 0. biases are always initialized to zero.
    #                 None: default PyTorch nn.linear initialization (similar to Kaiming Uniform / He)
    #                 'xavier_uniform' / 'xavier_normal' : Xavier (a.k.a. Glorot) initialization with uniform / normal entries
    #                 'kaiming_uniform' / 'kaiming_normal' : Kaiming (a.k.a. He) initialization with uniform / normal entries
    #                 Default: None
    #
    # batchNorm_final, batchNorm_hidden:
    #                 Tell whether to apply batch normalization to the outputs of the final and hidden layers.
    #                 The normalization is applied after the linear layer and before the activation.
    #                 Defaults: False
    #
    # dropout_final, dropout_hidden:
    #                 Dropout probabilities 0 <= p < 1 to be used at the final and hidden layers of the MLP.
    #                 The order of each layer is:  Linear transformation -> Batch normalization -> Activation -> Dropout
    #                 Defaults: 0
    def __init__(self,
                 in_channels, out_channels, 
                 args,
                 edgefeat_dim=0,
                 embed_dim=None, 
                 encode_vertex_degrees=True, vertex_degree_encoding_function='identity', homog_degree_encoding=False, 
                 concat_self = True,
                 bias=True,
                 mlp_hidden_dim=None, 
                 batchNorm_final = True, batchNorm_hidden = True,
                 edge_weighting = 'unit',
                 device=None, dtype=torch.float32):
        
        super().__init__(aggr=None)
        mlp_layers = args.mlp_layers
        mlp_activation_final = return_act(args.mlp_activation_final)
        mlp_activation_hidden = return_act(args.mlp_activation_hidden)
        dropout_final = args.dropout_final
        dropout_hidden = args.dropout_hidden
        learnable_embedding = args.learnable_embedding
        self.self_loop_weight = args.self_loop_weight
        self.embed_factor = 2
        mlp_init = args.mlp_init

        assert edge_weighting in {'unit', 'gcn'}, 'invalid value passed in argument <edge_weighting>'
        assert vertex_degree_encoding_function in {'identity', 'sqrt', 'log'}, 'invalid value passed in argument <vertex_degree_encoding_function>'

        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(in_channels, out_channels)

        if (mlp_layers == 0) and (concat_self == False):
            embed_dim = out_channels
        elif embed_dim == None:
            embed_dim = self.embed_factor * max(in_channels, out_channels)

        # If we're using an MLP and bias==True, then the MLP will add a bias anyway.
        embedding_bias = (bias and mlp_layers == 0)

        # Method to encode the vertex degrees by the FSW embedding
        embedding_total_mass_encoding_method = 'homog' if homog_degree_encoding else 'plain'
        
        self.edgefeat_dim = edgefeat_dim
        self.concat_self = concat_self
        self.edge_weighting = edge_weighting

        # if mlp_layers=0, mlp_input_dim is used also to determine the size of the dimensionality-reduction matrix
        if concat_self:
            mlp_input_dim = in_channels + embed_dim
        else:
            mlp_input_dim = embed_dim

        # construct MLP
        if mlp_layers == 0:
            self.mlp = None

            if concat_self:
                with torch.no_grad():
                    dim_reduct = torch.randn(size=(out_channels, mlp_input_dim), device=device, dtype=dtype, requires_grad=False)
                    dim_reduct = minimize_mutual_coherence(dim_reduct, report=False)
                
                self.dim_reduct = torch.nn.Parameter(dim_reduct, requires_grad=learnable_embedding)
        else:
            mlp_modules = []
            
            for i in range(mlp_layers):
                in_curr = mlp_input_dim if i == 0 else mlp_hidden_dim
                out_curr = out_channels if i == mlp_layers-1 else mlp_hidden_dim
                act_curr = mlp_activation_final if i == mlp_layers-1 else mlp_activation_hidden
                bn_curr = batchNorm_final if i == mlp_layers-1 else batchNorm_hidden
                dropout_curr = dropout_final if i == mlp_layers-1 else dropout_hidden

                layer_new = torch.nn.Linear(in_curr, out_curr, bias=bias, device=device, dtype=dtype)                

                # Apply initialization
                if mlp_init is None:
                    # do nothing
                    pass
                elif mlp_init == 'xavier_uniform':
                    torch.nn.init.xavier_uniform_(layer_new.weight)
                elif mlp_init == 'xavier_normal':
                    torch.nn.init.xavier_normal_(layer_new.weight)
                elif mlp_init == 'kaiming_uniform':
                    torch.nn.init.kaiming_uniform_(layer_new.weight)
                elif mlp_init == 'kaiming_normal':
                    torch.nn.init.kaiming_normal_(layer_new.weight)
                else:
                    raise RuntimeError('Invalid value passed at argument mlp_init')

                if (mlp_init is not None) and bias:
                    torch.nn.init.zeros_(layer_new.bias)

                mlp_modules.append( layer_new )

                if bn_curr:
                    mlp_modules.append( torch.nn.BatchNorm1d(num_features=out_curr, device=device, dtype=dtype) ) 

                if act_curr is not None:
                    mlp_modules.append( act_curr )

                if dropout_curr > 0:
                    mlp_modules.append( torch.nn.Dropout(p=dropout_curr) )
                
            self.mlp = torch.nn.Sequential(*mlp_modules); 
        
        self.size_coeff = torch.nn.Parameter( torch.ones(1, device=device, dtype=dtype) / np.sqrt(embed_dim), requires_grad=learnable_embedding)

        self.fsw_embed = FSW_embedding(d_in=in_channels, d_out=embed_dim, d_edge=edgefeat_dim,
                                       learnable_slices=learnable_embedding, learnable_freqs=learnable_embedding,
                                       encode_total_mass = encode_vertex_degrees, 
                                       total_mass_encoding_function = vertex_degree_encoding_function,
                                       total_mass_encoding_method = embedding_total_mass_encoding_method, 
                                       minimize_slice_coherence=True, freqs_init='spread',
                                       enable_bias=embedding_bias,
                                       device=device, dtype=dtype)

        device = device if device is not None else self.fsw_embed.get_device()
        dtype = dtype if dtype is not None else self.fsw_embed.get_dtype()
        self.to(device=device, dtype=dtype)



    def forward(self, vertex_features, edge_index, edge_features=None):
        # vertex_features has shape [num_vertices, in_channels]
        # edge_index has shape [2, num_edges]
        # edge_features should be left None if edgefeat_dim == 0, or be of shape (num_edges, edgefeat_dim)
        # with scalar features (i.e. edgefeat_dim == 1), edge_features can either be of shape (num_edges, 1) or (num_edges,)

        # Verify input
        assert vertex_features.dtype == self.fsw_embed.get_dtype(), 'vertex_features has incorrect dtype (expected %s, got %s)' % (self.fsw_embed.get_dtype(), vertex_features.dtype)
        assert vertex_features.device == self.fsw_embed.get_device(), 'vertex_features has incorrect device (expected %s, got %s)' % (self.fsw_embed.get_device(), vertex_features.device)
        assert edge_index.device == self.fsw_embed.get_device(), 'edge_index has incorrect device (expected %s, got %s)' % (self.fsw_embed.get_device(), edge_index.device)
        
        n = vertex_features.size(0)

        # This adds self-loops the old-fashioned way
        #edge_index, _ = add_self_loops(edge_index, num_nodes=n)

        # Calculate vertex degrees the old-fashioned way
        # row, col = edge_index
        # vertex_degrees = degree(col, n, dtype=vertex_features.dtype).unsqueeze(-1)

        # Convert edge_index to sparse adjacency matrix
        adj, X_edge, in_degrees = SW_conv.edge_index_to_adj(edge_index, edge_features=edge_features, num_vertices=n, edgefeat_dim=self.edgefeat_dim, dtype=vertex_features.dtype, edge_weighting=self.edge_weighting, self_loop_weight=self.self_loop_weight)
                    
        # Aggregate neighboring vertex features
        emb = self.fsw_embed(X=vertex_features, W=adj, X_edge=X_edge, graph_mode=True, serialize_num_slices=None)

        if self.concat_self:
            emb = torch.cat((emb, vertex_features), dim=-1)

        # Apply MLP or dimensionality reduction to neighborhood embeddings
        if self.mlp is not None:
            out = self.mlp(emb)
        elif self.concat_self:
            out = torch.matmul(emb, self.dim_reduct.transpose(0,1))
        else:
            out = emb

        return out


    def aggregate(self, inputs, index):
        return

    def message(self, x_j):
        return

    def update(self, aggr_out):
        return


    def edge_index_to_adj(edge_index, edge_features, num_vertices, edgefeat_dim, dtype, self_loop_weight=0, edge_weighting='unit'):
        num_edges = edge_index.shape[1]

        inds = edge_index.flip(0)
        vals = torch.ones(num_edges, device=edge_index.device, dtype=dtype)

        if self_loop_weight > 0:
            inds2 = torch.arange(num_vertices, device=edge_index.device).reshape([1, num_vertices]).repeat(2,1)
            vals2 = self_loop_weight*torch.ones(num_vertices, device=edge_index.device, dtype=dtype)

            inds = torch.cat( (inds, inds2), dim=1 )
            vals = torch.cat( (vals, vals2), dim=0 )

        adj = torch.sparse_coo_tensor(indices=inds, values=vals, size=(num_vertices,num_vertices))          
        adj = adj.coalesce()

        slice_info_W = sp.get_slice_info(adj, -1)
        in_degrees = ag.sum_sparseToDense.apply(adj, -1, slice_info_W)

        if edge_weighting == 'unit':
            pass # do nothing

        elif edge_weighting == 'gcn':
            in_degrees_sqrt = torch.sqrt(in_degrees)
            adj = ag.div_sparse_dense.apply(adj, in_degrees_sqrt, slice_info_W)
            adj = ag.div_sparse_dense.apply(adj, in_degrees_sqrt.transpose(-1,-2), slice_info_W)

            # Note: This weighting scheme only considers in-degrees.
            #       Once can use a directed-graph variant by replacing the line above with
            #out_degrees = ag.sum_sparseToDense.apply(adj, -2, slice_info_W)
            #adj = ag.div_sparse_dense.apply(adj, torch.sqrt(out_degrees), slice_info_W)
        else:
            raise RuntimeError('Invalid weighting method passed in argument <edge_weighting>')
            
        # Handle edge features
        if edgefeat_dim > 0:
            assert edge_features is not None, 'Edge features must be provided since edgefeat_dim > 0'
            assert edge_features.dim() in (1,2), 'edge_features should have the shape (num_edges, edegfeat_dim) (or optionally (num_edges,) in the case edgefeat_dim=1)'
            assert (edgefeat_dim == 1) or (edge_features.dim() == 2), 'edge_features must have the shape (num_edges, edgefeat_dim)'
            if edgefeat_dim == 1:
                assert tuple(edge_features.shape) in { (num_edges,), (num_edges, edgefeat_dim)}, 'edge_features should have the shape (num_edges, edegfeat_dim) (or optionally (num_edges,) in the case edgefeat_dim=1)'
            else:
                assert tuple(edge_features.shape) == (num_edges, edgefeat_dim), 'edge_features must have the shape (num_edges, edgefeat_dim)'
            X_edge_shape = adj.shape if edge_features.dim()==1 else tuple(adj.shape)+(edgefeat_dim,)
            assert adj.is_coalesced()
            X_edge = sp.sparse_coo_tensor_coalesced(indices=adj.indices(), values=edge_features, size=X_edge_shape)

        else:
            assert edge_features is None, 'Edge features should not be provided since edgefeat_dim = 0'
            
            X_edge = None


        return adj, X_edge, in_degrees



#@register_pooling('fsw_readout')
class FSW_readout(SW_conv):
    # What is the @register_pooling decorator?
    # What is the input format? Shapes?
    # - Is 'batch' the batch index?
    # - So the number of batches is the maximal batch index?
    # - What if some indices in the range are absent? 
    #   Are they considered empty graphs that are part of the batch? Do we return a global feature for them as well?
    # - Currently I am not updating the edge features. Do competing methods do it?
    
    def forward(self, vertex_features, batch=None):
        # create batch numbering for single graph if needed
        if batch is None:
            batch = torch.zeros(vertex_features.shape[0]).long().to(vertex_features.device)
          
        self.batch_size = batch.max().item() + 1
        self.num_vertices = vertex_features.shape[0]
        # setting edge index so all nodes are connected to the first node per graph
        src = torch.arange(self.num_vertices, device=vertex_features.device)
        #dst = torch.cat([torch.ones(num_nodes[i], device=vertex_features.device)*self.first_node_index[i] for i in range(0, len(num_nodes))], dim=0)
        dst = batch
        edge_index = torch.stack([src, dst], dim=0).long().to(vertex_features.device)
        vals = torch.ones_like(edge_index[0,:], dtype=vertex_features.dtype)
        
        #adj = torch.sparse_coo_tensor(edge_index.flip(0), vals, (self.batch_size, self.num_vertices), is_coalesced=False).coalesce()
        adj = torch.sparse_coo_tensor(edge_index.flip(0), vals, (self.batch_size, self.num_vertices), is_coalesced=True)
        
        slice_info_W = sp.get_slice_info(adj, -1)
        in_degrees = ag.sum_sparseToDense.apply(adj, -1, slice_info_W)
        ################################################################################################

        
        # Aggregate neighboring vertex features
        emb = self.sw_embed(X=vertex_features, W=adj, graph_mode=True, serialize_num_slices=None)
        

        # Add neighborhood sizes multiplied by the norms of the neighborhood embeddings, and optionally also the self feature of each vertex
        emb_cat_list = (vertex_features,) if self.concat_self else ()
        emb_cat_list = emb_cat_list + (emb, (self.size_coeff*in_degrees*emb.norm(dim=-1,keepdim=True)))
        emb = torch.cat(emb_cat_list, dim=-1)
        
        # Apply MLP or dimensionality reduction to neighborhood embeddings
        if self.mlp is not None:
            out = self.mlp(emb)
        elif self.concat_self:
            out = torch.matmul(emb, self.dim_reduct.transpose(0,1))
        else:
            out = emb

        return out
    
