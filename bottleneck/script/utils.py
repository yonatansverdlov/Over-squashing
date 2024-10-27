import pathlib
from enum import Enum, auto
import torch
import yaml
from easydict import EasyDict
from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, SAGEConv
import torch_geometric
from models.sw_layer import SW_conv
import shutil, os
from datasets.graphs_generation import TreeDataset, CliqueRing, RingDataset

class GNN_TYPE(Enum):
    """
    The GNN types.
    """
    GCN = 0
    GGNN = 1
    GIN = 2
    GAT = 3
    SW = 4
    SAGE = 5

    def __str__(self):
        if self is GNN_TYPE.GIN:
            return 'GIN'
        if self is GNN_TYPE.SW:
            return 'SW'
        if self is GNN_TYPE.GAT:
            return 'GAT'
        if self is GNN_TYPE.GGNN:
            return 'GGNN'
        if self is GNN_TYPE.GCN:
            return 'GCN'
        if self is GNN_TYPE.SAGE:
            return 'SAGE'

def get_layer(args: EasyDict, in_dim: int, out_dim: int):
    """
    Getting the layer
    :param args: The args.
    :param in_dim: The input dim.
    :param out_dim: The output dim.
    :return:
    """
    type = args.gnn_type
    if type is GNN_TYPE.GCN:
        return GCNConv(
            in_channels=in_dim,
            out_channels=out_dim)
    elif type is GNN_TYPE.GGNN:
        return GatedGraphConv(out_channels=out_dim, num_layers=1)
    elif str(type) == 'GIN':
        return GINConv(nn.Sequential(nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
                                     nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()))
    elif str(type) == 'GAT':
        return GATConv(in_dim, out_dim // 4, heads=4)
    elif str(type) == 'SW':
        return SW_conv(in_channels=in_dim, out_channels=out_dim,args = args)
    elif str(type) == 'SAGE':
        return SAGEConv(in_channels=in_dim,out_channels=out_dim)



def get_args(depth: int, gnn_type: GNN_TYPE, task_type: str):
    """
    :param depth:
    :param gnn_type:
    :param num_layers:
    :param task_type:
    :return:
    """
    clean_args = EasyDict()
    path_to_args = f"{str(pathlib.Path(__file__).parent)}/configs/task_config.yaml"
    with open(path_to_args) as f:
        type_config = EasyDict(yaml.safe_load(f))
    for key, item in type_config['General'].items():
        setattr(clean_args, key, item)
    for key, item in type_config['Task_specific'][str(gnn_type)][task_type].items():
        setattr(clean_args, key, item)
    clean_args.depth = depth
    clean_args.gnn_type = gnn_type
    clean_args.task_type = task_type
    if gnn_type is GNN_TYPE.SW:
        clean_args.layer_type = SW_conv
    elif gnn_type is GNN_TYPE.GGNN:
        clean_args.layer_type = GatedGraphConv
    elif gnn_type is GNN_TYPE.GIN:
        clean_args.layer_type = GINConv
    elif gnn_type is GNN_TYPE.GAT:
        clean_args.layer_type = GATConv
    elif gnn_type is GNN_TYPE.GCN:
        clean_args.layer_type = GCNConv
    elif gnn_type is GNN_TYPE.SAGE:
        clean_args.layer_type = SAGEConv
    return clean_args, type_config['Task_specific'][str(gnn_type)][task_type]

def compute_dirichlet_energy(data, embedding):
    # Get node features and edge indices
    x = embedding
    edge_index = data.edge_index

    # Compute norms of each node feature vector
    norms = torch.norm(x, dim=1, keepdim=True)

    # Normalize node features
    x_normalized = x / norms

    # Compute the Dirichlet energy
    energy = 0
    for edge in edge_index.t():
        i, j = edge
        dot_product = torch.dot(x_normalized[i], x_normalized[j])
        energy += 1 - dot_product

    degree = torch.zeros(data.num_nodes, dtype=torch.float)
    for edge in data.edge_index.t():
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    sum_of_degrees = degree.sum()
    energy /= (sum_of_degrees * 2)

    return energy.item()

def compute_energy(data,model):
    model.eval()
    data = data.cuda()
    model = model.cuda()
    with torch.no_grad():
         embedding = model.compute_node_embedding(data)
    return compute_dirichlet_energy(data=data, embedding=embedding)

def create_model_dir(args, task_specific):
    model_name = 'Model'
    for key, item in task_specific.items():
        model_name += f'_{key}_{item}'
    path_to_project = str(pathlib.Path(__file__).parent.parent)
    model_dir = f'{path_to_project}/data/models/{str(args.task_type)}/{str(args.gnn_type)}/Radius_{args.depth}/{model_name}'
    print(model_dir)
    src = str(pathlib.Path(__file__).parent)
    if not os.path.exists(model_dir) and False:
        shutil.copytree(src, model_dir)
    return model_dir, path_to_project

def return_datasets(args):
    task = args.task_type
    if task == 'Tree':
        X_train, X_test, X_val = TreeDataset(args=args).generate_data(args.train_fraction)
    if task == 'Ring':
        X_train, X_test, X_val = RingDataset(args=args, add_crosses=False
                                             ).generate_data(args.train_fraction)
    if task == 'CrossRing':
        X_train, X_test, X_val = RingDataset(args=args, add_crosses=True
                                             ).generate_data(args.train_fraction)
    if task == 'CliqueRing':
        X_train, X_test, X_val = CliqueRing(args = args).generate_data(
            args.train_fraction)
    if task == 'Actor':
        sets = torch_geometric.datasets.Actor('../data')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 932
        args.out_dim = 5
    if task == 'Squir':
        sets = torch_geometric.datasets.WikipediaNetwork('../data',name='Squirrel')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 2089
        args.out_dim = 5
    if task == 'Cham':
        sets = torch_geometric.datasets.WikipediaNetwork('../data',name='chameleon')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 2325
        args.out_dim = 5 
    if task == 'Texas':
        sets = torch_geometric.datasets.WebKB('../data',name='Texas')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 1703
        args.out_dim = 5
    if task == 'Corn':
        sets = torch_geometric.datasets.WebKB('../data',name='Cornell')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 1703
        args.out_dim = 5
    if task == 'Wisc':
        sets = torch_geometric.datasets.WebKB('../data',name='Wisconsin')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 1703
        args.out_dim = 5
    if task == 'Cora':
        sets = torch_geometric.datasets.Planetoid('../data',split = 'geom-gcn',name='Cora')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 1433
        args.out_dim = 7
    if task == 'Cite':
        sets = torch_geometric.datasets.Planetoid('../data',split = 'geom-gcn',name='CiteSeer')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 3703 
        args.out_dim = 6
    if task == 'Pubm':
        sets = torch_geometric.datasets.Planetoid('../data',split = 'geom-gcn',name='PubMed')
        X_train, X_test, X_val = sets, sets, sets
        args.in_dim = 500 
        args.out_dim = 3
    
    return X_train, X_test, X_val