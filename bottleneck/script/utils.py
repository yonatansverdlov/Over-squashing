import torch_geometric
import pathlib
import torch
import yaml
from easydict import EasyDict
from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, SAGEConv
from torch_geometric.data import Dataset
from torch.utils.data import Dataset, Subset

from models.fsw.fsw_layer import FSW_conv
from data_generate.graphs_generation import TreeDataset, CliqueRing, RingDataset


def split_dataset_for_10_fold(dataset: Dataset, fold_id: int):
    """
    Split dataset into training and testing subsets for 10-fold cross-validation.

    Args:
        dataset (Dataset): The full dataset to be split.
        fold_id (int): The fold index to use as the test set (0 to 9).

    Returns:
        tuple: Training and testing subsets of the dataset.
    """
    assert 0 <= fold_id < 10, "fold_id should be between 0 and 9."

    dataset_size = len(dataset)
    fold_size = dataset_size // 10

    # Indices for test and train sets
    test_start_idx = fold_id * fold_size
    test_end_idx = test_start_idx + fold_size
    test_indices = list(range(test_start_idx, test_end_idx))
    train_indices = list(range(0, test_start_idx)) + list(range(test_end_idx, dataset_size))

    return Subset(dataset, train_indices), Subset(dataset, test_indices)


def get_layer(args: EasyDict, in_dim: int, out_dim: int):
    """
    Get a GNN layer based on the specified type in args.

    Args:
        args (EasyDict): Configuration dictionary with `gnn_type`.
        in_dim (int): Input dimension of the layer.
        out_dim (int): Output dimension of the layer.

    Returns:
        nn.Module: Initialized GNN layer.
    """
    gnn_layers = {
        'GCN': lambda: GCNConv(in_channels=in_dim, out_channels=out_dim),
        'GGNN': lambda: GatedGraphConv(out_channels=out_dim, num_layers=1),
        'GIN': lambda: GINConv(nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(),
            nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU()
        )),
        'GAT': lambda: GATConv(in_dim, out_dim // 4, heads=4),
        'SW': lambda: FSW_conv(in_channels=in_dim, out_channels=out_dim, config=dict(args)),
        'SAGE': lambda: SAGEConv(in_channels=in_dim, out_channels=out_dim)
    }
    return gnn_layers[args.gnn_type]()


def get_args(depth: int, gnn_type: str, task_type: str):
    """
    Load and update arguments from a YAML configuration file.

    Args:
        depth (int): Depth of the model.
        gnn_type (str): Type of GNN layer.
        task_type (str): Task type for dataset generation.

    Returns:
        tuple: Configuration arguments and task-specific settings.
    """
    clean_args = EasyDict(depth=depth, gnn_type=gnn_type, task_type=task_type)
    config_path = pathlib.Path(__file__).parent / "configs/task_config.yaml"
    
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))

    # Update with general and task-specific configurations
    clean_args.update(config['General'])
    clean_args.update(config['Task_specific'][gnn_type][task_type])

    # Modify batch size and gradients for Tree task
    if task_type == 'Tree':
        tree_settings = {
            2: (1024, 1, 2048), 3: (1024, 1, 2048), 4: (1024, 1, 2048),
            5: (1024, 1, 2048), 6: (1024, 1, 2048), 7: (512, 2, 2048), 8: (256, 4, 1024)
        }
        clean_args.batch_size, clean_args.accum_grad, clean_args.val_batch_size = tree_settings.get(depth, (1024, 1, 2048))

    return clean_args, config['Task_specific'][gnn_type][task_type]


def compute_dirichlet_energy(data, embedding):
    """
    Compute Dirichlet energy of normalized embeddings based on graph structure.

    Args:
        data (Data): Graph data with edge indices.
        embedding (Tensor): Node embeddings.

    Returns:
        float: Dirichlet energy value.
    """
    edge_index = data.edge_index
    norms = torch.norm(embedding, dim=1, keepdim=True)
    x_normalized = embedding / norms

    energy = sum(1 - torch.dot(x_normalized[i], x_normalized[j])
                 for i, j in edge_index.t()) / (2 * edge_index.size(1))
    return energy.item()


def compute_energy(data, model):
    """
    Compute Dirichlet energy for node embeddings of a model.

    Args:
        data (Data): Graph data.
        model (nn.Module): Graph neural network model.

    Returns:
        float: Dirichlet energy of the node embeddings.
    """
    model.eval()
    data, model = data.cuda(), model.cuda()
    with torch.no_grad():
        embedding = model.compute_node_embedding(data)
    return compute_dirichlet_energy(data, embedding)


def create_model_dir(args, task_specific, seed: int):
    """
    Create a directory for model checkpoints and logs.

    Args:
        args (EasyDict): Configuration arguments.
        task_specific (dict): Task-specific settings.
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: Model directory path and project base path.
    """
    model_name = '_'.join([f"{key}_{val}" for key, val in task_specific.items()])
    path_to_project = pathlib.Path(__file__).parent.parent
    model_dir = path_to_project / f"data/models/{args.task_type}/{args.gnn_type}/Radius_{args.depth}/{seed}/{model_name}"
    return str(model_dir), str(path_to_project)


def return_datasets(args):
    """
    Load datasets based on task type and update configuration dimensions.

    Args:
        args (EasyDict): Configuration arguments.

    Returns:
        tuple: Training, testing, and validation datasets.
    """
    data_path = pathlib.Path('../data/raw')
    transductive = ['Actor', 'Squi', 'Cham', 'Texas', 'Corn', 'Wisc', 'Cora', 'Cite', 'Pubm']
    TUDatasets = ['MUTAG']

    task_datasets = {
        'Tree': lambda: TreeDataset(args=args).generate_data(args.train_fraction),
        'Ring': lambda: RingDataset(args=args, add_crosses=False).generate_data(),
        'CrossRing': lambda: RingDataset(args=args, add_crosses=True).generate_data(),
        'CliqueRing': lambda: CliqueRing(args=args).generate_data(),
        'Actor': lambda: torch_geometric.datasets.Actor(data_path),
        'Squi': lambda: torch_geometric.datasets.WikipediaNetwork(data_path, name='squirrel'),
        'Cham': lambda: torch_geometric.datasets.WikipediaNetwork(data_path, name='chameleon'),
        'Texas': lambda: torch_geometric.datasets.WebKB(data_path, name='Texas'),
        'Corn': lambda: torch_geometric.datasets.WebKB(data_path, name='Cornell'),
        'Wisc': lambda: torch_geometric.datasets.WebKB(data_path, name='Wisconsin'),
        'Cora': lambda: torch_geometric.datasets.Planetoid(data_path, split='geom-gcn', name='Cora'),
        'Cite': lambda: torch_geometric.datasets.Planetoid(data_path, split='geom-gcn', name='CiteSeer'),
        'Pubm': lambda: torch_geometric.datasets.Planetoid(data_path, split='geom-gcn', name='PubMed'),
        'MUTAG': lambda: torch_geometric.datasets.TUDataset(data_path, name='MUTAG')
    }

    dataset = task_datasets[args.task_type]()

    if args.task_type in transductive:
        args.in_dim = dataset.num_node_features
        args.out_dim = dataset.num_classes
        return dataset, dataset, dataset
    elif args.task_type in TUDatasets:
        args.in_dim = dataset.num_node_features
        args.out_dim = dataset.num_classes
        train_dataset, test_dataset = split_dataset_for_10_fold(dataset=dataset, fold_id=args.split_id)
        return train_dataset, test_dataset, test_dataset
    else:
        return dataset
