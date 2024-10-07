import torch.nn as nn
import torch




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






COMBINE_DICT = {'LinearCombination': LinearCombination,
                'LTSum': LTSum,
                'Concat': Concat,
                'ConcatProject': ConcatProject}