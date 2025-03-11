import torch_geometric
import pathlib
import torch
import yaml
from torch_geometric.loader import DataLoader
from easydict import EasyDict
from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.data import Dataset
from torch.utils.data import Dataset, Subset
import numpy as np
from models.fsw.fsw_layer import FSW_conv
from data_generate.graphs_generation import TreeDataset, CliquePath, RingDataset, TwoConnectedCycles, PathGraph, KIndependentPaths, OneRadiusProblemStarGraph, TwoRadiusProblemStarGraph
import time
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
import random                                                                                                                                                                
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from lightning.pytorch import seed_everything

class MetricAggregationCallback(Callback):
    def __init__(self, eval_every=5):
        super().__init__()
        self.eval_every = eval_every
        self.all_val_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch
        val_acc = trainer.callback_metrics.get("val_acc", None)
        if val_acc is not None:
            if epoch not in self.all_val_metrics:
                self.all_val_metrics[epoch] = []
            self.all_val_metrics[epoch].append(val_acc.item() * 100)

    def get_best_epoch(self):
        if not self.all_val_metrics:
            return None, None, None  # Handle case where no metrics exist

        avg_metrics = {epoch: np.mean(accs) for epoch, accs in self.all_val_metrics.items()}
        best_epoch = max(avg_metrics, key=avg_metrics.get)
        
        best_mean = np.mean(self.all_val_metrics[best_epoch])
        best_std = np.std(self.all_val_metrics[best_epoch])

        return best_mean, best_std

class TimingCallback(Callback):
    def __init__(self, validation_interval=5):
        self.total_training_time = 0
        self.total_validation_time = 0
        self.total_test_time = 0
        self.epochs_tracked = 0
        self.validation_interval = validation_interval
        self.validation_epochs = 0  # Count only epochs where validation happens

    def on_train_epoch_start(self, trainer, pl_module):
        """Called at the start of each training epoch."""
        self.epoch_training_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        """Called at the end of each training epoch."""
        epoch_training_time = time.time() - self.epoch_training_start_time
        self.total_training_time += epoch_training_time
        self.epochs_tracked += 1

    def on_validation_start(self, trainer, pl_module):
        """Called at the start of validation (only triggered at intervals)."""
        if trainer.current_epoch % self.validation_interval == 0:
            self.epoch_validation_start_time = time.time()

    def on_validation_end(self, trainer, pl_module):
        """Called at the end of validation (only triggered at intervals)."""
        if trainer.current_epoch % self.validation_interval == 0:
            epoch_validation_time = time.time() - self.epoch_validation_start_time
            self.total_validation_time += epoch_validation_time
            self.validation_epochs += 1

    def on_test_start(self, trainer, pl_module):
        """Called at the start of testing."""
        self.test_start_time = time.time()

    def on_test_end(self, trainer, pl_module):
        """Called at the end of testing."""
        self.total_test_time = time.time() - self.test_start_time

        avg_training_time = self.total_training_time / self.epochs_tracked
        avg_validation_time = (
            self.total_validation_time / self.validation_epochs
            if self.validation_epochs > 0
            else 0
        )
        print("\nTraining and Testing Summary:")
        print(f"Average training time per epoch: {avg_training_time:.2f} seconds")
        print(f"Average validation time per validation run: {avg_validation_time:.2f} seconds")
        print(f"Average testing time per testing run: {self.total_test_time:.2f} seconds")

class StopAtValAccCallback(Callback):
    """
    Callback for early stopping when validation accuracy reaches a target value.
    
    Args:
        target_acc (float): Accuracy threshold for stopping training early.
    """
    def __init__(self, target_acc=0.99):
        super().__init__()
        self.target_acc = target_acc

    def on_validation_epoch_end(self, trainer, _):
        """
        Checks validation accuracy at the end of each epoch, stopping training if the target is met.
        
        Args:
            trainer (Trainer): PyTorch Lightning trainer instance managing training.
        """
        val_acc = trainer.callback_metrics.get('val_acc')
        if val_acc is not None and val_acc >= self.target_acc:
            trainer.should_stop = True
            print(f"Stopping training as `val_acc` reached {val_acc * 100:.2f}%")
        else:
            print(f"Current validation accuracy: {val_acc * 100:.2f}%")

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
        'GAT': lambda: GATConv(in_dim, out_dim // 1, heads=1),
        'SW': lambda: FSW_conv(in_channels=in_dim, out_channels=out_dim, config=dict(args)),
        'SAGE': lambda: SAGEConv(in_channels=in_dim, out_channels=out_dim),
        'Transformer': lambda: TransformerConv(in_channels=in_dim, out_channels=out_dim),
    }
    return gnn_layers[args.gnn_type]()

def get_args(num_layers, depth: int, gnn_type: str, task_type: str,n:int):
    """
    Load and update arguments from a YAML configuration file.

    Args:
        depth (int): Depth of the model.
        gnn_type (str): Type of GNN layer.
        task_type (str): Task type for dataset generation.

    Returns:
        tuple: Configuration arguments and task-specific settings.
    """
    clean_args = EasyDict(depth=depth, gnn_type=gnn_type, task_type=task_type,num_layers = num_layers,n=n)
    config_path = pathlib.Path(__file__).parent / "configs/task_config.yaml"
    
    with open(config_path) as f:
        config = EasyDict(yaml.safe_load(f))

    # Update with general and task-specific configurations
    clean_args.update(config['Common'])
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


def create_model_dir(args, task_specific):
    """
    Create a directory for model checkpoints and logs.

    Args:
        args (EasyDict): Configuration arguments.
        task_specific (dict): Task-specific settings.

    Returns:
        tuple: Model directory path and project base path.
    """
    model_name = '_'.join([f"{key}_{val}" for key, val in task_specific.items()])
    path_to_project = pathlib.Path(__file__).parent.parent
    model_dir = path_to_project / f"data/models/{args.task_type}/{args.gnn_type}/Radius_{args.depth}/{model_name}"
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
    TUDatasets = ['MUTAG','Protein','PTC','NCI']

    task_datasets = {
        'Tree': lambda: TreeDataset(args=args).generate_data(args.train_fraction),
        'Ring': lambda: RingDataset(args=args, add_crosses=False).generate_data(),
        'CrossRing': lambda: RingDataset(args=args, add_crosses=True).generate_data(),
        'CliquePath': lambda: CliquePath(args=args).generate_data(),
        'TwoCycles': lambda: TwoConnectedCycles(args=args).generate_data(),
        'Path': lambda: PathGraph(args=args).generate_data(),
        'KPaths': lambda: KIndependentPaths(args=args).generate_data(),
        'one_radius': lambda: OneRadiusProblemStarGraph(args=args).generate_data(),
        'two_radius': lambda: TwoRadiusProblemStarGraph(args=args).generate_data(),
    }

    dataset = task_datasets[args.task_type]()
    return dataset
    
def oversmoothing_metric(X, adj):
    """
    Computes the normalized oversmoothing metric for a GNN layer.

    Parameters:
    - X: Tensor of shape (num_nodes, feature_dim), node embeddings.
    - adj: Sparse adjacency matrix (num_nodes, num_nodes).

    Returns:
    - normalized_metric: Normalized oversmoothing metric in [0, 1].
    """
    # Normalize the embeddings (cosine similarity requires normalization)
    X_norm = X / torch.norm(X, dim=1, keepdim=True)

    # Compute pairwise cosine similarities for neighbors
    cosine_sim = torch.mm(X_norm, X_norm.T)  # (num_nodes, num_nodes)
    
    # Element-wise 1 - cosine similarity
    dissimilarity = 1 - cosine_sim

    # Mask with the adjacency matrix to only consider neighbors
    neighbor_dissimilarity = dissimilarity * adj.to_dense()

    # Compute the unnormalized metric
    num_nodes = X.size(0)
    metric = neighbor_dissimilarity.sum() / num_nodes

    # Normalize by the maximum possible value (mu_max)
    mu_max = 2 * adj.sum() / num_nodes  # Maximum dissimilarity
    normalized_metric = metric / mu_max

    return normalized_metric

def average_oversmoothing_metric(model, data_loader):
    """
    Computes the average oversmoothing metric for graphs in a DataLoader.

    Parameters:
    - model: GNN model with a `compute_node_embedding` method.
    - data_loader: DataLoader where each element is a Data object (PyTorch Geometric).

    Returns:
    - average_metric: Average normalized oversmoothing metric across all graphs in the DataLoader.
    """
    oversmoothing_metrics = []
    
    device = model.device
    
    for data in data_loader:
        # Extract node features and edge indices for the current graph
        data = data.to(device)
        x = data.x  # Node features
        edge_index = data.edge_index  # Edge indices
        # Compute node embeddings using the model
        node_embeddings = model.compute_node_embedding(data)

        # Create adjacency matrix for the current graph
        num_nodes = x.size(0)
        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        adj[edge_index[0], edge_index[1]] = 1

        # Compute the oversmoothing metric for the current graph
        metric = oversmoothing_metric(node_embeddings, adj)
        oversmoothing_metrics.append(metric.item())

    # Compute the average oversmoothing metric across all graphs
    average_metric = sum(oversmoothing_metrics) / len(oversmoothing_metrics)

    return average_metric

def worker_init_fn(seed: int):
    np.random.seed(seed)
    random.seed(seed)
    
def create_trainer(args, task_specific,  metric_callback, need_time=True):
    model_dir, _ = create_model_dir(args, task_specific)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-f{val_acc:.5f}',
        save_top_k=10,
        monitor='val_acc',
        save_last=True,
        mode='max')
    stop_callback = StopAtValAccCallback()   
    time_callback = TimingCallback() if need_time else None
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback, time_callback, metric_callback] if callback]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=True,
        check_val_every_n_epoch=args.eval_every,
        callbacks=callbacks_list,
        default_root_dir=f'{model_dir}/lightning_logs'
    )
    return trainer, checkpoint_callback

def fix_seed(seed):
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        seed_everything(seed, workers=True)

def return_dataloader(args):
    X_train, X_test, X_val = return_datasets(args=args)
    # Prepare data loaders
    train_loader = DataLoader(
        X_train, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers,
        persistent_workers=True, worker_init_fn=lambda _: worker_init_fn(args.seed)
    )
    val_loader = DataLoader(
        X_val, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.loader_workers
    )
    test_loader = DataLoader(
        X_test, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.loader_workers
    )
    return train_loader, val_loader, test_loader, X_val 

def compute_os_energy(model, Data):
    model = model.eval()  # Evaluation mode
    # Compute number of nodes per graph assuming contiguous blocks in Data.x
    num_nodes = Data.x.size(0) // Data.num_graphs  
    # Ensure Data.x requires gradients
    Data.x = Data.x.clone().detach().requires_grad_()

    def model_target(x):
        Data_clone = Data.clone()  # Clone full Data object
        Data_clone.x = x  # Use modified input
        Data_clone.x.retain_grad()  # Retain gradients (optional debugging)
        torch.set_grad_enabled(True)
        # Assume that model.compute_node_embedding returns a tensor whose entries
        # corresponding to each graph are selected by Data_clone.train_mask
        return model.compute_node_embedding(Data_clone)[Data_clone.train_mask]

    # Compute the full Jacobian.
    # Expected jacobian shape: (B, F_out, B*num_nodes, F_in) where B = Data.num_graphs (here 5)
    jacobian = torch.autograd.functional.jacobian(model_target, Data.x)

    B = Data.num_graphs  # Batch size (expected to be 5)
    # Reshape jacobian from shape (B, F_out, B*num_nodes, F_in)
    # to shape (B, F_out, B, num_nodes, F_in)
    jacobian = jacobian.view(B, jacobian.shape[1], B, num_nodes, jacobian.shape[3])
    # For each graph j, extract the corresponding slice: jacobian[j, :, j, :, :]
    idx = torch.arange(B)
    partial_grads = jacobian[idx, :, idx, :, :]  # shape: (B, F_out, num_nodes, F_in)

    # Compute numerator: for each graph, take the gradient for the first node (index 0)
    # and compute its Frobenius norm (i.e. norm over output feature and input feature dimensions)
    numerator = torch.norm(partial_grads[:, :, 0, :], p=2, dim=(1, 2))

    # Compute denominator: for each graph, compute the Frobenius norm over the output and input feature
    # dimensions for each node, then sum over all nodes in that graph.
    # Here, torch.norm(..., dim=(1, 3)) computes a norm over the F_out and F_in dims, leaving a tensor of shape (B, num_nodes)
    denominator = torch.norm(partial_grads, p=2, dim=(1, 3)).sum(dim=1) + 1e-6

    # Compute per-graph energy and average over batch
    energy = numerator / denominator
    return energy.mean()




    