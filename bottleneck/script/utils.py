import torch_geometric
import pathlib
import torch
import yaml
from easydict import EasyDict
from torch import nn
from torch_geometric.nn import GCNConv, GatedGraphConv, GINConv, GATConv, SAGEConv, TransformerConv
from torch_geometric.data import Dataset
from torch.utils.data import Dataset, Subset
import numpy as np
from models.fsw.fsw_layer import FSW_conv
from data_generate.graphs_generation import TreeDataset, CliquePath, RingDataset
import time
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer import Trainer
import random                                                                                                                                                                
from lightning.pytorch.callbacks import ModelCheckpoint
import os
from pytorch_lightning import seed_everything

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
    def __init__(self, target_acc=1.0):
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
    TUDatasets = ['MUTAG','Protein','PTC','NCI']

    task_datasets = {
        'Tree': lambda: TreeDataset(args=args).generate_data(args.train_fraction),
        'Ring': lambda: RingDataset(args=args, add_crosses=False).generate_data(),
        'CrossRing': lambda: RingDataset(args=args, add_crosses=True).generate_data(),
        'CliquePath': lambda: CliquePath(args=args).generate_data(),
        'Actor': lambda: torch_geometric.datasets.Actor(data_path),
        'Squi': lambda: torch_geometric.datasets.WikipediaNetwork(data_path, name='squirrel'),
        'Cham': lambda: torch_geometric.datasets.WikipediaNetwork(data_path, name='chameleon'),
        'Texas': lambda: torch_geometric.datasets.WebKB(data_path, name='Texas'),
        'Corn': lambda: torch_geometric.datasets.WebKB(data_path, name='Cornell'),
        'Wisc': lambda: torch_geometric.datasets.WebKB(data_path, name='Wisconsin'),
        'Cora': lambda: torch_geometric.datasets.Planetoid(data_path, split='geom-gcn', name='Cora'),
        'Cite': lambda: torch_geometric.datasets.Planetoid(data_path, split='geom-gcn', name='CiteSeer'),
        'Pubm': lambda: torch_geometric.datasets.Planetoid(data_path, split='geom-gcn', name='PubMed'),
        'MUTAG': lambda: torch_geometric.datasets.TUDataset(data_path, name='MUTAG'),
        'Protein': lambda: torch_geometric.datasets.TUDataset(data_path, name='PROTEINS'),
        'NCI': lambda: torch_geometric.datasets.TUDataset(data_path, name='NCI1'),
    }

    dataset = task_datasets[args.task_type]()

    if args.task_type in transductive:
        args.in_dim = dataset.num_node_features
        args.out_dim = dataset.num_classes
        return dataset, dataset, dataset
    elif args.task_type in TUDatasets:
        args.in_dim = dataset.num_node_features
        args.out_dim = dataset.num_classes
        args.num_edge_features = dataset.num_edge_features
        train_dataset, test_dataset = split_dataset_for_10_fold(dataset=dataset, fold_id=args.split_id)
        return train_dataset, test_dataset, test_dataset
    else:
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

def compute_gnn_derivative(model, source, target, data, epsilons, num_trials):
    """
    Compute the derivative-like quantity for a GNN model.

    Parameters:
    - model: The GNN model (assumes a PyTorch model).
    - source: Index of the source node.
    - target: Index of the target node.
    - x: Feature matrix (torch.Tensor).
    - epsilons: List of epsilon values to use.
    - num_trials: Number of trials with random perturbation vectors.

    Returns:
    - results: Dictionary with epsilons as keys and average derivative estimates as values.
    """
    device = next(model.parameters()).device  # Ensure computations are on the model's device
    x = data.x
    x = x.to(device)  # Move x to the same device as the model

    results = {}
    x = x.clone()  # Ensure x is not modified in-place
    softmax = torch.nn.Softmax(dim=None)
    for eps in epsilons:
        derivatives = []
        for _ in range(num_trials):
            # Generate a random vector of the same shape as x[source]
            random_vector = torch.randn_like(x[source], device=device)
            random_vector = random_vector / random_vector.norm(p=2)  # Normalize the vector (L2 norm)

            # Perturb x[source] with eps * random_vector
            x_perturbed = x.clone()
            x_perturbed[source] += eps * random_vector

            # Compute F(x+eps*v)[target] and F(x)[target]
            with torch.no_grad():
                data.root_mask = data.train_mask
                data.x = x_perturbed
                F_x_perturbed = softmax(model(data))
                data.x = x 
                F_x = softmax(model(data))

            # Compute the difference and divide by eps
            derivative = (F_x_perturbed - F_x) / eps

            # Take the norm of the derivative (you can customize this)
            derivatives.append(derivative.norm(p=2).item())

        # Store the average derivative for this epsilon
        return np.mean(derivatives)

def worker_init_fn(seed: int):
    np.random.seed(seed)
    random.seed(seed)
def create_trainer(args, task_specific, seed, metric_callback, need_time=True):
    model_dir, _ = create_model_dir(args, task_specific, seed=seed)
    checkpoint_callback = ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-f{val_acc:.5f}',
        save_top_k=10,
        monitor='val_acc',
        save_last=True,
        mode='max')
    stop_callback = StopAtValAccCallback() if args.task_type in ['Ring','CliquePath','CrossRing','Tree'] else None    
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