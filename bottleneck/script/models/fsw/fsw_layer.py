import numpy as np

import torch
import torch_geometric as pyg
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.graphgym import cfg
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_layer, register_pooling
import sys
import os
import importlib.util

import type_enforced # Note: A runtime error in this line implies that that some function below is given an input arguemnt of the wrong type
from typing import Dict, Any
import inspect

#TODO: Check if this path update is necessary
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


dtype_mapping = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.int32": torch.int32,
    # Add more mappings as needed
}


def return_act(act: str):
    if act == 'relu':
        return torch.nn.ReLU()
    if act == 'lrelu':
        return torch.nn.LeakyReLU(0.1)
    if act == 'tanh':
        return torch.nn.Tanh()
    if act == 'silu':
        return torch.nn.SiLU()
    if act == 'gelu':
        return torch.nn.GELU()
    else:
        raise NotImplemented

# Now you can use FSW_embedding, minimize_mutual_coherence, ag, and sp directly

# Release notes:
#
# 2024-09-03
# - Added support for edge features.
#   To use, set edgefeat_dim > 0 at init, and pass argument edge_features in forward(),
#   where edge_features is a tensor of size (num of edges, edgefeat_dim)
# - Elegant handling of empty neighborhoods => No need to use self_loop_weight > 0 anymore.
# - Added built-in support to make the whole model homogeneous w.r.t. the vertex- and edge-features.
#   To make the model homogeneous: (1) use homog_degree_encoding = True, (2) use bias=False, and
#   (3) use only homogeneous activations, namely ReLU or Leaky ReLU.
#   Use this when the function to be learned is known to be homogeneous.
# - If the function to be learned is known to be invariant to vertex degrees, set 
#   encode_vertex_degrees=False.

@type_enforced.Enforcer(enabled=True)
class FSW_conv(MessagePassing):
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
    #                        note that 'vertex degree' here refers to the sum of weights of incoming edges to that vertex.
    #                        default: True
    #
    # vertex_degree_encoding_function: tells what function to apply to vertex degrees before encoding. options:
    #                                  'identity': f(x)=x
    #                                  'sqrt': f(x) = sqrt(1+x)-1
    #                                  'log': f(x) = log(1+x)
    #                                  default: 'identity'
    #
    # vertex_degree_encoding_scale: factor to multiply the vertex degree encodings.
    #                               is learnable when vertex_degree_encoding_scale=True.
    #                               default: 1.0
    #
    # learnable_vertex_degree_encoding_scale: tells whether the vertex degree encoding scale is learnable.
    #                                         default: False
    #
    # homog_degree_encoding: tells whether the neighborhood-size encoding should be homogeneous.
    #                        better keep this off unless it is desired that the model should be homogeneous.
    #                        NOTE: To make the whole model homogeneous, make sure bias=False and all activations are
    #                              homogeneous (e.g. leaky ReLU).
    #                        default: False
    #
    # vertex_degree_pad_thresh: vertices with degrees smaller than this threshold (after adding self loops)
    #                           are padded with an incoming neighbor whose vertex feature is zero and a corresponding edge
    #                           weight ( vertex_degree_pad_thresh - <this vertex's degree> ).
    #                           can be any positive number. better leave this value at the default 1.0.
    #                           default: 1.0
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
    #                 If mlp_layers > 0, the normalization is applied after the linear layer and before the activation.
    #                 If mlp_layers = 0, the normalization is applied at the very last step, i.e. after embedding and, when applicable,
    #                 self-concatenation and dimensionality reduction.
    #                 Defaults: False
    #
    # dropout_final, dropout_hidden:
    #                 Dropout probabilities 0 <= p < 1 to be used at the final and hidden layers of the MLP.
    #                 The order of each layer is:  Linear transformation -> Batch normalization -> Activation -> Dropout
    #                 Defaults: 0
    #
    # config:         dictionary with settings that are to override the input arguments.
    #                 for example, __init__(in_channels = 5, ..., config={'in_channels':10})
    #                 will use in_channels = 10.
    #                 default: None
    def __init__(self,
                 in_channels, out_channels, edgefeat_dim=0,
                 embed_dim=None, learnable_embedding=True,
                 encode_vertex_degrees=True, vertex_degree_encoding_function='identity', 
                 vertex_degree_encoding_scale=1.0, learnable_vertex_degree_encoding_scale=False, homog_degree_encoding=False, 
                 vertex_degree_pad_thresh = 1.0,
                 concat_self = True,
                 bias=True,
                 mlp_layers=1, mlp_hidden_dim=None,
                 mlp_activation_final = torch.nn.LeakyReLU(negative_slope=0.2), 
                 mlp_activation_hidden = torch.nn.LeakyReLU(negative_slope=0.2), 
                 mlp_init = None,
                 batchNorm_final = False, batchNorm_hidden = False,
                 dropout_final = 0, dropout_hidden = 0,
                 self_loop_weight = 0.0, edge_weighting = 'unit',
                 device=None, dtype=torch.float32,
                 config : dict | None = None):
        
        super().__init__(aggr=None)

        config = config if config is not None else {}

        # Get the function's input argument names
        # This is in case the function is not a method:
        #frame = inspect.currentframe()
        #function_name = frame.f_code.co_name
        #func = globals()[function_name]
        func = getattr(self.__class__, '__init__', None)
        arg_names = { param.name for param in inspect.signature(func).parameters.values() }
        arg_names = arg_names.difference({'config','self'})

        # Override input arguments by arguments given in config
        config = {key: value for key, value in config.items() if key in arg_names}
             
        for argname in arg_names:
            if argname not in config:
                config[argname] = locals()[argname]

        self.init_helper(**config)
    

    # The input arguments here should be the same as in __init__, except for 'config' which is omitted here.
    def init_helper(self,
                    in_channels, out_channels, edgefeat_dim,
                    embed_dim, learnable_embedding,
                    encode_vertex_degrees, vertex_degree_encoding_function, 
                    vertex_degree_encoding_scale, learnable_vertex_degree_encoding_scale, homog_degree_encoding, 
                    vertex_degree_pad_thresh,
                    concat_self,
                    bias,
                    mlp_layers, mlp_hidden_dim,
                    mlp_activation_final, 
                    mlp_activation_hidden, 
                    mlp_init,
                    batchNorm_final, batchNorm_hidden,
                    dropout_final, dropout_hidden,
                    self_loop_weight, edge_weighting,
                    device, dtype):
        assert edge_weighting in {'unit', 'gcn'}, 'invalid value passed in argument <edge_weighting>'
        assert vertex_degree_encoding_function in {'identity', 'sqrt', 'log'}, 'invalid value passed in argument <vertex_degree_encoding_function>'
        dtype = dtype_mapping[dtype]
        if mlp_hidden_dim is None:
            mlp_hidden_dim = max(in_channels, out_channels)

        if (mlp_layers == 0) and (concat_self == False):
            embed_dim = out_channels
        elif embed_dim == None:
            embed_dim = 2 * max(in_channels, out_channels)

        # If we're using an MLP and bias==True, then the MLP will add a bias anyway.
        embedding_bias = (bias and mlp_layers == 0)

        # Method to encode the vertex degrees by the FSW embedding
        embedding_total_mass_encoding_method = 'homog' if homog_degree_encoding else 'plain'
        
        self.edgefeat_dim = edgefeat_dim
        self.concat_self = concat_self
        self.edge_weighting = edge_weighting
        self.self_loop_weight = self_loop_weight

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

            # Batch-normalization to apply to the output when an MLP is not used
            self.bn_final = torch.nn.BatchNorm1d(num_features=out_channels, device=device, dtype=dtype) if batchNorm_final else None

        else:
            self.bn_final = None
            mlp_modules = []
            mlp_activation_final = return_act(mlp_activation_final)
            mlp_activation_hidden = return_act(mlp_activation_hidden)
            for i in range(mlp_layers):
                in_curr = mlp_input_dim if i == 0 else mlp_hidden_dim
                out_curr = out_channels if i == mlp_layers-1 else mlp_hidden_dim
                act_curr =  mlp_activation_final if i == mlp_layers-1 else mlp_activation_hidden
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
                                       learnable_slices=learnable_embedding, learnable_freqs=learnable_embedding, learnable_total_mass_encoding_scale = learnable_vertex_degree_encoding_scale,
                                       encode_total_mass = encode_vertex_degrees, 
                                       total_mass_encoding_function = vertex_degree_encoding_function,
                                       total_mass_encoding_scale = vertex_degree_encoding_scale,
                                       total_mass_encoding_method = embedding_total_mass_encoding_method, 
                                       total_mass_pad_thresh = vertex_degree_pad_thresh,
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
        adj, X_edge, in_degrees = FSW_conv.edge_index_to_adj(edge_index, edge_features=edge_features, num_vertices=n, edgefeat_dim=self.edgefeat_dim, dtype=vertex_features.dtype, edge_weighting=self.edge_weighting, self_loop_weight=self.self_loop_weight)
                    
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

        if self.bn_final is not None:
            out = self.bn_final(out)

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

            if self_loop_weight > 0:
                s = list(edge_features.shape)
                s[0] = num_vertices
                edge_features_pad = torch.zeros(s, device=edge_index.device, dtype=dtype)
                edge_features = torch.cat( (edge_features, edge_features_pad), dim=0 )

                s = list(adj.shape)

            X_edge = torch.sparse_coo_tensor(indices=inds, values=edge_features, size=X_edge_shape)
            X_edge = X_edge.coalesce()

        else:
            assert edge_features is None, 'Edge features should not be provided since edgefeat_dim = 0'
            
            X_edge = None


        return adj, X_edge, in_degrees

class FSW_readout(FSW_conv):
    def forward(self, vertex_features, graph_index=None, batch_size=None):
        # vertex_features: tensor of size [num_vertices, vertex_feature_dimension], where num_vertices is the total number of vertices in the whole batch.
        # 
        # graph_index: tensor of size [num_vertices,]. for each vertex, tells which graph in the batch it belongs to.
        #              if None, assumes the batch consists of a single graph.
        #
        # batch_size:  number of graphs in the batch. 
        #              if None, this number is determined automatically as max(graph_index)+1.
        #
        # NOTE: Graphs whose index is in the range 0,...,batch_size-1 that are not assigned any vertices
        #       are treated as empty graphs that are still part of the input, and an output embedding is generated for them a well.
        #
        # Output shape: [out_channels, batch_size]

        # TODO: This check should better be done in a custom __init__ method of this class.
        assert self.edgefeat_dim == 0, 'edgefeat_dim should equal zero in a global readout layer'

        num_vertices = vertex_features.shape[0]

        if graph_index is None:
            assert batch_size is None, 'batch_size must be None when graph_index is None'
        else:
            assert tuple(graph_index.shape) == (num_vertices,), 'graph_index should be of shape (num_vertices,), where vertex_features is of shape (num_features, vertex_feature_dimension)'
            assert is_monotone_increasing(graph_index), 'for efficiency, graph_index should be monotone non-decreasing'

        # create batch numbering for single graph if needed
        if graph_index is None:
            graph_index = torch.zeros(vertex_features.shape[0], device=vertex_features.device, dtype=torch.int64)

        # automatically determine batch_size if not provided  
        batch_size = graph_index.max().item() + 1 if batch_size is None else batch_size

        assert (graph_index < batch_size).all(), 'all entries of graph_index must be in the range 0,...,batch_size-1'
        assert (graph_index >= 0).all(), 'all entries of graph_index must be in the range 0,...,batch_size-1'

        # Check device and dtype of input for compatibility
        assert vertex_features.device == self.fsw_embed.get_device(), 'invalid device given in vertex_features (expected \'%s\', got \'%s\'' % (str(self.fsw_embed.get_device()), str(vertex_features.device))
        assert graph_index.device == self.fsw_embed.get_device(), 'invalid device given in graph_index (expected \'%s\', got \'%s\'' % (str(self.fsw_embed.get_device()), str(graph_index.device))

        assert vertex_features.dtype == self.fsw_embed.get_dtype(), 'invalid dtype given in vertex_features (expected \'%s\', got \'%s\'' % (str(self.fsw_embed.get_dtype()), str(vertex_features.dtype))
        assert graph_index.dtype == torch.int64, 'invalid dtype given in graph_index (expected \'%s\', got \'%s\'' % (str(torch.int64), str(graph_index.dtype))

        # creating edges from all vertices to the corresponding global node of their graph
        src = torch.arange(num_vertices, device=vertex_features.device)
        dst = graph_index

        # Create sparse adjacency matrix.
        # Note that we use [dst, src] in adj_indices since the adjacency matrix is given in the format: adj[i,j] = weight of the edge from j to i
        adj_indices = torch.stack([dst, src], dim=0).long()
        adj_vals = torch.ones_like(adj_indices[0,:], dtype=vertex_features.dtype)
        
        adj = sp.sparse_coo_tensor_coalesced(adj_indices, adj_vals, (batch_size, num_vertices))
               
        # Aggregate global graph features
        emb = self.fsw_embed(X=vertex_features, W=adj, graph_mode=True, serialize_num_slices=None)
        
        # Apply MLP or dimensionality reduction to neighborhood embeddings
        if self.mlp is not None:
            out = self.mlp(emb)
        elif self.concat_self:
            out = torch.matmul(emb, self.dim_reduct.transpose(0,1))
        else:
            out = emb

        return out
    

def is_monotone_increasing(tensor):
    # Compute the difference between consecutive elements
    diffs = tensor[1:] - tensor[:-1]
    
    # Check if all differences are greater than or equal to 0
    return torch.all(diffs >= 0)
