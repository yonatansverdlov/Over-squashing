import torch.nn as nn
import torch
from torch_geometric.nn import MessagePassing
import torch
from torch_geometric.nn.norm import LayerNorm, BatchNorm
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import index_sort, to_dense_batch

class LinearCombination(nn.Module):
  '''
  Takes (x,y)-> [a_1*x+y|...|a_{num_repeats}*x+y]
  '''
  def __init__(self, in_dim, num_repeats):
    """
    :param: in_dim - dimension of each input vector
    :param: num_repeats - number of scalars for which to compute ax+y and concatenate.
    """
    super().__init__()
    self.num_scalars = num_repeats
    self.embedding_dim = in_dim*num_repeats
    self.scalars = nn.ParameterList([torch.nn.init.uniform_(torch.nn.Parameter(torch.Tensor([1])), a=-1, b=1 ) for _ in range(num_repeats)])

  def forward(self, x,y):
    out = torch.cat([a*x+y for a in self.scalars], dim=-1)
    return out

  @staticmethod
  def num_repeats_from_dims(in_dim, embed_dim):
    if embed_dim % in_dim != 0:
      raise NotImplementedError('embed_dim must be a multiple of in_dim in LinearCombination')
    return int(embed_dim/in_dim)
  
class LTSum(nn.Module):
  '''
  Takes (x,y)-> [A*x+y|...|A_{num_scalars}*x+y]
  '''
  def __init__(self, in_dim_1, in_dim_2, num_repeats):
    """
    :param: in_dim_1 - dimension of first input vector
    :param: in_dim_2 - dimension of second input vector
    :param: num_repeats - number of matrices for which to compute Ax+y and concatenate.
    """
    super().__init__()
    self.in_dim_2 = in_dim_2
    self.num_repeats = num_repeats
    self.embedding_dim = in_dim_2*num_repeats
    self.linears = nn.ParameterList([torch.nn.Linear(in_dim_1, in_dim_2, bias=False) for _ in range(num_repeats)])
    for lin in self.linears:
      # torch.nn.init.normal_(lin.weight)
      torch.nn.init.normal_(lin.weight, 0 ,0.05)

  def forward(self, x,y):
    if len(y.shape) == 0:
      y = torch.zeros()
    out = torch.cat([lin(x)+y for lin in self.linears], dim=-1)
    return out

  @staticmethod
  def num_repeats_from_dims(in_dim_2, embed_dim):
    if embed_dim % in_dim_2 != 0:
      raise NotImplementedError('embed_dim must be a multiple of in_dim_2 in LTSum')
    return int(embed_dim/in_dim_2)
  
class Concat(nn.Module):
  '''
  Takes (x,y)-> [x|y]
  '''
  def __init__(self, in_dim_1, in_dim_2):
    super().__init__()
    self.embedding_dim = (in_dim_1+in_dim_2)

  def forward(self, x,y):
    out = torch.cat([x,y], dim=-1)
    return out
  
class ConcatProject(nn.Module):
  '''
  Takes (x,y)-> [<w_1,[x|y]>|...|<w_{num_repeats},[x|y]>]
  '''
  def __init__(self, in_dim_1, in_dim_2, num_repeats):
    super().__init__()
    """
    :param: in_dim_1 - dimension of first input vector
    :param: in_dim_2 - dimension of second input vector
    :param: num_repeats - number of inner products to concat.
    """
    super().__init__()
    self.num_repeats = num_repeats
    self.embedding_dim = num_repeats
    self.lin = torch.nn.Linear(in_dim_1+in_dim_2, num_repeats, bias=False)
    torch.nn.init.normal_(self.lin.weight)
    # torch.nn.init.normal_(self.lin.weight, 0, 0.01)

  def forward(self, x,y):
    cat = torch.cat([x,y], dim=-1)
    out = self.lin(cat)
    return out

combine_dict = {'LinearCombination': LinearCombination, 'LTSum': LTSum, 'Concat': Concat, 'ConcatProject': ConcatProject}
norm_dict = {'layer': LayerNorm, 'batch': BatchNorm, 'none': torch.nn.Identity}

class SortConv(MessagePassing):
    def __init__(self, in_dim, out_dim, orig_dim, combine,collapse_method, max_nodes:int):
        """
        :param in_dim: dimension of input node feature vectors
        :param out_dim: dimension of global graph output vector
        :param max_nodes: max number of neighbors across all dataset
        """
        super().__init__()
        bias = True
        if orig_dim is None:
          orig_dim = in_dim
        self.orig_dim = orig_dim
        self.out_dim = out_dim

        self.lin_project = torch.nn.Linear(in_dim, out_dim, bias=bias)

        # Setting up for chosen collapse method
        self.collapse_method = collapse_method

        if collapse_method == 'vector':
          self.lin_collapse = torch.nn.Linear(max_nodes, 1, bias=bias)
        elif collapse_method == 'matrix':
          self.lin_collapse = torch.nn.Parameter(torch.zeros((max_nodes,out_dim)))
        else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")

        # Setting up chosen update method
        if combine == LinearCombination:
          if in_dim != out_dim:
            raise NotImplementedError('Cannot combine with LinearCombination when in_dim!=out_dim')
          self.combine = combine(in_dim, 1)
        elif combine == LTSum:
          self.combine = combine(in_dim, out_dim,1)
        elif combine == Concat:
          self.combine = combine(in_dim, out_dim)
        elif combine == ConcatProject:
          self.combine = combine(in_dim, out_dim, out_dim)
        elif combine is None:
          self.combine = torch.nn.Identity()
        else:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')

        self.max_nodes = max_nodes
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_normal_(self.lin_project.weight)
        self.lin_project.weight = nn.Parameter(self.lin_project.weight / torch.norm(self.lin_project.weight, dim=1, keepdim=True))
        if self.collapse_method == 'vector':
          torch.nn.init.xavier_normal_(self.lin_collapse.weight)
        elif self.collapse_method == 'matrix':
          torch.nn.init.normal_(self.lin_collapse)

    def message(self, x_j):
      # project from in_dim --> out_dim
      return self.lin_project(x_j)

    def get_num_indices(self, index):
      with torch.no_grad():
        num_indices = int(torch.max(index).item()) +1
        return num_indices

    def return_collapsed_output(self, sorted):
      # collapse
      if self.collapse_method == 'vector':
        out = self.lin_collapse(sorted.permute(0,2,1)).squeeze(-1) # [Num_nodes("batch"), num_neighbors("max_nodes"), out_dim] --> [Num_nodes("batch"), out_dim]
      elif self.collapse_method == 'matrix':
        out = sorted * self.lin_collapse
        out = torch.sum(out, dim=1).squeeze(1)
      else:
          raise NotImplementedError("collapse method must be on of [vector|matrix]")

      self.blank_vec = out[-1:,:]
      return out[:-1,:]

    
    def convert_to_dense(self, inputs, index, indices, num_indices):
      return to_dense_batch(inputs[indices], index[indices],
                                    max_num_nodes=self.max_nodes, batch_size=num_indices+1)

    def aggregate(self, inputs, index, blank_vec):
      # in index, we replace each index with its position in the sorted list
      # create dense neighborhoods augmented with blank vectors
      num_indices = self.num_indices
      _, indices = index_sort(index)
      # converting to dense batch, adding blank vector to end
      result, mask = self.convert_to_dense(inputs, index, indices, num_indices)
      result = result.to(torch.float32)
      blank_vec = self.lin_project(blank_vec)
      result += (~mask.unsqueeze(-1)).repeat(1,1,blank_vec.shape[0])*blank_vec # filling with blank tree vector

      # sort each column independently
      sorted, _ = torch.sort(result, dim=-2)

      # collapse
      return self.return_collapsed_output(sorted)

    def update(self, aggr_out, orig_x, blank_vec, orig_blank_vec):
        # multiply central node by eta and add to aggregated neighbors
        if len(blank_vec.shape) < 2:
          blank_vec = blank_vec.unsqueeze(0)
        if len(orig_blank_vec.shape) < 2:
          orig_blank_vec = orig_blank_vec.unsqueeze(0)

        self.blank_vec = self.combine(orig_blank_vec,self.blank_vec)
        return self.combine(orig_x, aggr_out)

    def forward(self, x, edge_index, blank_vec, orig_x=None, orig_blank_vec=None, edge_attrs=None):
        if orig_x is None:
          orig_x = x
        if orig_blank_vec is None:
          orig_blank_vec = blank_vec
        self.num_indices = x.shape[0]
        x = self.message(x_j=x)
        inputs = x[edge_index[1,:],:]
        index = edge_index[0,:]
        aggr_out = self.aggregate(inputs=inputs, blank_vec=blank_vec,index=index)
        x = self.update(aggr_out=aggr_out, orig_x=orig_x,blank_vec=blank_vec,orig_blank_vec=orig_blank_vec)
        return x, self.blank_vec

class SortGlobalConv(SortConv):
    def __init__(self, dim_in, dim_out,max_nodes,orig_dim, collapse_method, combine):
        """
        :param dim_in: dimension of input node feature vectors
        :param dim_out: dimension of global graph output vector
        :param max_nodes: max number of vertices in graph across all dataset
        """
        super().__init__(dim_in, dim_out, max_nodes=max_nodes,orig_dim=orig_dim,
                          collapse_method=collapse_method, combine=combine)

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
        x = self.message(x_j=x)
        inputs = x[edge_index[1,:],:]
        aggr_out = self.aggregate(inputs=inputs, blank_vec=blank_vec,index=batch)
        return aggr_out


def setup_learnable_blank_vector(in_dim, embed_dim, num_layers):
    blank_vectors = nn.ParameterList([nn.Parameter(torch.zeros(in_dim))])
    for _ in range(num_layers):
        blank_vectors.append(nn.Parameter(torch.zeros(embed_dim)))

    return blank_vectors


class SortMPNN(torch.nn.Module):
    '''
    MPNN that uses sort as non-linearity.
    The MPNN optionally maintains an additional blank tree vector which augments node neighborhoods.
    '''
    def __init__(self, dim_in, dim_out,num_layers,max_nodes):
        super().__init__()
        embed_dim = 128
        self.max_nodes = max_nodes
        self.need_contenous_features = True
        num_layers = num_layers
        combine = ConcatProject # concatproject
        update_w_orig = False 
        redisual = True
        self.dropout = 0.0
        norm = 'batch'
        blank_vector_method = 'learnable'
        collapse_method = 'matrix' # matrix
        self.update_w_orig = update_w_orig
        self.skip_connections = redisual
        self.embed_label = nn.Linear(dim_in, embed_dim)
        dim_in = embed_dim

        if combine not in [LinearCombination, LTSum, Concat, ConcatProject]:
          raise NotImplementedError('combine must be one of [LinearCombination|LTSum|Concat|ConcatProject]')
        
        assert blank_vector_method in ['iterative_update', 'zero', 'learnable'], \
                                      'blank_vector_method must be one of [iterative_update, zero, learnable]'
        self.blank_vector_method = blank_vector_method
        if self.blank_vector_method == 'learnable':
          self.blank_vectors = setup_learnable_blank_vector(dim_in, embed_dim, num_layers)

        
        first_combine = LTSum if (combine == LinearCombination and dim_in!=embed_dim)  else combine
        orig_dim = dim_in if update_w_orig else None
        conv1 = SortConv(dim_in, dim_in, orig_dim=orig_dim, combine=first_combine,collapse_method=collapse_method, 
                         max_nodes=max_nodes)
        layers = [conv1]
        for _ in range(num_layers-1):
            layers.append(SortConv(dim_in,
                                     dim_in, orig_dim=orig_dim,combine=combine, collapse_method=collapse_method, max_nodes = max_nodes))
        self.convs = torch.nn.Sequential(*layers)
        self.norms = nn.ModuleList()
        for i in range(num_layers):
          self.norms.append(norm_dict[norm](embed_dim))
        self.is_real = False
        self.embedding = nn.Embedding(embedding_dim=embed_dim,num_embeddings=5)
        self.readout = nn.Linear(embed_dim,dim_out)
        torch.nn.init.xavier_uniform_(self.readout.weight)
        self.readout = SortGlobalConv(dim_in=embed_dim,dim_out=dim_out,max_nodes=max_nodes,
                                      combine=combine, collapse_method=collapse_method,orig_dim=orig_dim)

    def forward(self, batch):
        # self.node_encoder(batch)
        x ,edge_index = batch.x, batch.edge_index
        x = self.embed_label(x)
        
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
        out = self.readout(x, blank_vec,batch.batch)
        return out



