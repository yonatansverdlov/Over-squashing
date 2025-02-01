import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel, StopAtValAccCallback
from utils import get_args, create_model_dir, compute_energy, return_datasets
from models.graph_model import GraphModel
import argparse
from easydict import EasyDict
import random
import torch
from torch_geometric.data import Data
import numpy as np
import os
# Set random seed for reproducibility
import torch
import random
import numpy as np
from TMD.tmd import TMD
# Ensure PyTorch uses deterministic algorithms
import torch.nn as nn
import torch
from torch_geometric.nn.models import GIN, GAT, GCN
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv, GINConv
from rival_models import GraphLevelGCN, GraphLevelGIN, GraphLevelGAT
from models.SortMPNN import SortMPNN

import torch
import cvxpy as cp
import numpy as np
from torch_geometric.utils import to_dense_adj

def dsmetric(data1,data2, lambda_features=1,w = 1, L = 1, use_squared_dists=False, return_S=False):
    """
    Compute the doubly stochastic metric between two vertex-featured graphs.
    
    Args:
        A1 (torch.Tensor): Adjacency matrix of graph 1 (n x n).
        V1 (torch.Tensor): Vertex-feature matrix of graph 1 (n x d).
        A2 (torch.Tensor): Adjacency matrix of graph 2 (n x n).
        V2 (torch.Tensor): Vertex-feature matrix of graph 2 (n x d).
        lambda_features (float): Weight of the vertex-feature term in the objective.
        use_squared_dists (bool): Whether to use squared distances in the feature term.
        return_S (bool): Whether to return the optimal doubly-stochastic matrix S in addition to the DS metric.
    
    Returns:
        float: Optimal value of the objective.
        torch.Tensor: Optimized doubly stochastic matrix (n x n). (optional)
    """
    V1, V2 = data1.x, data2.x
    A1, A2 =  data1.edge_index, data2.edge_index
    A1, A2 = to_dense_adj(A1).squeeze(), to_dense_adj(A2).squeeze()
    [n, d] = V1.shape
    [n2, d2] = V2.shape
    
    assert n == n2, "Graph sizes (number of nodes) must match."
    assert d == d2, "Feature dimensions must match."

    assert not (A1.is_sparse or A2.is_sparse or V1.is_sparse or V2.is_sparse), 'All input tensors must be dense'

    # Convert PyTorch tensors to NumPy arrays for CVXPY
    A1_np = A1.detach().numpy()
    V1_np = V1.detach().numpy()
    A2_np = A2.detach().numpy()
    V2_np = V2.detach().numpy()

    # Compute pairwise distances
    diff = V1_np[:, None, :] - V2_np[None, :, :]  # Shape (n, n, d)
    dists = np.linalg.norm(diff, axis=2)  # Pairwise L2 norms, shape (n, n)

    # Define CVXPY variable for S
    S = cp.Variable((n, n))

    # Constraints for doubly stochastic matrix
    constraints = [
        S >= 0,  # Non-negativity
        cp.sum(S, axis=1) == 1,  # Rows sum to 1
        cp.sum(S, axis=0) == 1   # Columns sum to 1
    ]

    # Objective terms
    structure_term = cp.norm(A1_np @ S - S @ A2_np, "fro")
    if use_squared_dists:
        feature_term = cp.sqrt(cp.sum(cp.multiply(S, dists**2)))
    else:
        feature_term = cp.sum(cp.multiply(S, dists))

    # Define and solve the optimization problem
    objective = cp.Minimize(structure_term + lambda_features * feature_term)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    # Ensure solver succeeded
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Solver failed with status: {problem.status}")

    # Retrieve the optimal value and convert back to PyTorch tensors if needed
    optimal_value = problem.value
    S_optimized = torch.tensor(S.value, dtype=torch.float32).to(A1.device)  # Convert S to PyTorch tensor

    if return_S:
        return optimal_value, S_optimized
    else:
        return optimal_value

def compute_ratio_flipped(model: nn.Module, create_data_func, d,n, n_times: int,depth:int,eps_vals:list,dtype ):
    """
    Computes the ratio |model(data1) - model(data2)| / |TMD(data1) - TMD(data2)|
    for N_times iterations, using flipped Data objects created dynamically.

    Args:
    - model (Module): A PyTorch model that takes a Data object as input.
    - create_data_func (function): Function to create flipped Data objects.
    - tmd (function): A transformation function TMD(data) that computes a scalar for the Data object.
    - n (int): Number of nodes in the graph.
    - d (int): Dimensionality of node features.
    - n_times (int): Number of times to compute the ratio.

    Returns:
    - min_ratio (float): The minimum ratio computed.
    - max_ratio (float): The maximum ratio computed.
    """
    min_ratio = float('inf')
    max_ratio = float('-inf')
    for _ in range(n_times):
        # Create two flipped Data objects
        for eps in eps_vals:
            data1, data2 = create_data_func(d = d, eps = eps, n = n,dtype = dtype )

            # Compute model outputs
            model_out1 = model(data1.cuda()).detach()
            model_out2 = model(data2.cuda()).detach()

            # Compute TMD outputs
            data1, data2 = data1.cpu(), data2.cpu()
            tmd_out = dsmetric(data1, data2, w = 1, L = depth)

            # Calculate the numerator and denominator of the ratio
            numerator = torch.norm(model_out1 - model_out2, p=2).item()
            denominator = tmd_out

            # Avoid division by zero
            if denominator == 0:
                continue

            # Compute the ratio
            ratio = numerator / denominator

            # Update min and max ratios
            min_ratio = min(min_ratio, ratio)
            max_ratio = max(max_ratio, ratio)

    return  max_ratio / min_ratio

def generate_vectors_with_noise_vectorized(d, n, eps,dtype):
    """
    Generate a base vector of norm 1 in R^d and create n vectors 
    by flipping the base vector and adding noise scaled by sigma.

    Args:
        d (int): Dimension of the space (R^d).
        n (int): Number of noisy vectors to generate.
        sigma (float): Noise scale.

    Returns:
        torch.Tensor: A tensor of shape (n, d) containing the noisy vectors.
    """
    # Generate a random base vector of norm 1
    base_vector = torch.randn(d,dtype=dtype)
    base_vector /= base_vector.norm()  # Normalize to norm 1

    # Flip the base vector for all n
    flipped_vectors = base_vector.repeat(n, 1)  # Shape: (n, d)

    # Add noise to the flipped vectors
    noise = torch.rand(n, d,dtype=dtype)  # Shape: (n, d)
    noise /= noise.norm(dim=1, keepdim=True)  # Normalize each noise vector to norm 1

    # Scale the noise and add to flipped vectors
    noisy_vectors = flipped_vectors + eps * noise
    return flipped_vectors, noisy_vectors

def create_square_graph_with_zero_features(d, eps, n, dtype):
    """
    Create two square (2-regular) graph data objects:
    1. One with computed node features based on a random vector.
    2. One with all node features set to zero.
    
    Args:
        d (int): Dimension of the random vector.
        eps (float): Scaling factor for the node features.
        
    Returns:
        tuple: Two PyTorch Geometric Data objects (with computed features, with zero features).
    """
    # Define the square graph as edges
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 0],  # Source nodes
        [1, 0, 2, 1, 3, 2, 0, 3]   # Target nodes
    ], dtype=torch.long)

    # Generate a random vector in R^d with norm 1
    random_vector = torch.randn(d,dtype=dtype)
    random_vector = random_vector / random_vector.norm()  # Normalize to unit length
    second_random_vector = torch.randn(d,dtype=dtype)
    second_random_vector = second_random_vector / second_random_vector.norm()  # Normalize to unit length
    # Compute node features
    node_features = torch.stack([
        eps * random_vector,     # Node 0
        -eps * random_vector,    # Node 1
        -eps * random_vector,    # Node 2
        eps * random_vector      # Node 3
    ], dim=0)
    node_features = node_features + second_random_vector

    # Create the data object with computed features
    data_with_features = Data(x=node_features, edge_index=edge_index)

    # Create the data object with zero features
    zero_features = node_features = torch.stack([
       0 * random_vector,     # Node 0
        0 * random_vector,    # Node 1
        0 * random_vector,    # Node 2
        0 * random_vector      # Node 3
    ], dim=0) # Four nodes, each with d-dimensional zero features
    zero_features = zero_features + second_random_vector
    data_with_zero_features = Data(x=zero_features, edge_index=edge_index)

    return data_with_features, data_with_zero_features

def create_symmetric_graph_with_two_features_and_batch(n, d, eps,dtype):
    """
    Create a symmetric graph with n nodes and two sets of randomly initialized n x d node features
    (each with Frobenius norm 1), using the same graph structure, and include batch information for pooling.

    Args:
    - n (int): Number of nodes in the graph.
    - d (int): Dimension of node features.

    Returns:
    - Data, Data: Two PyTorch Geometric Data objects with the same graph structure but different node features.
    """
    # Generate a random symmetric adjacency matrix
    adj_matrix = torch.randint(0, 2, (n, n),dtype=dtype)  # Random 0/1 adjacency matrix
    adj_matrix = torch.triu(adj_matrix)  # Upper triangular part
    adj_matrix = adj_matrix + adj_matrix.T  # Make it symmetric
    adj_matrix.fill_diagonal_(0)  # Remove self-loops

    # Convert adjacency matrix to edge indices
    edge_indices = torch.stack(adj_matrix.nonzero(as_tuple=True), dim=0)

    # Create two sets of random node features
    node_features1, node_features2 = generate_vectors_with_noise_vectorized(d=d, n=n,eps=eps,dtype = dtype)

    #node_features2 = torch.randn((n, d))  # Second random feature matrix
    #node_features2 /= node_features2.norm()  # Normalize to Frobenius norm 1

    # Batch tensor: all nodes belong to the same graph, so batch is all zeros
    batch = torch.zeros(n, dtype=torch.long)  # All nodes are part of a single graph

    # Create PyTorch Geometric Data objects
    data1 = Data(x=node_features1.cuda(), edge_index=edge_indices.cuda(), batch=batch.cuda())
    data2 = Data(x=node_features2.cuda(), edge_index=edge_indices.cuda(), batch=batch.cuda())

    return data1, data2

dtype_mapping = {
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.float16": torch.float16,
    "torch.int32": torch.int32,
    # Add more mappings as needed
}

def train_graphs(args: EasyDict):
    """
    Train, validate, and optionally compute Dirichlet energy for a graph model.

    Args:
        args (EasyDict): Configuration dictionary containing model hyperparameters and dataset parameters.
        task_id (int): Identifier for multi-task settings to specify which sub-task to train on.

    Returns:
        float: Computed Dirichlet energy (if `compute_dirichlet` is enabled in args).
        float: Test accuracy of the model on the test set.

    The function:
    1. Creates a directory for saving model checkpoints and logs.
    2. Sets up the `StopAtValAccCallback` for early stopping based on validation accuracy.
    3. Initializes the PyTorch Lightning `Trainer` with custom callbacks and configurations.
    4. Loads the training, validation, and test datasets.
    5. Trains the model, evaluates on the test set, and optionally computes the Dirichlet energy.
    """
    # Set up directory and callbacks

    # Load datasets based on task type
    n = 10
    # Initialize model for the specified task
    if args.gnn_type == 'SW':
        base_model = GraphModel(args=args).cuda()
    elif args.gnn_type == 'GCN':
        base_model = GraphLevelGCN(input_dim=args.in_dim,output_dim=args.out_dim,hidden_dim=args.dim,num_layers=args.depth).cuda()
    elif args.gnn_type == 'GAT':
        base_model = GraphLevelGAT(input_dim=args.in_dim,output_dim=args.out_dim,hidden_dim=args.dim,num_layers=args.depth).cuda()
    elif args.gnn_type == 'GIN':
        base_model = GraphLevelGIN(input_dim=args.in_dim,output_dim=args.out_dim,hidden_dim=args.dim,num_layers=args.depth).cuda()
    elif args.gnn_type == 'Sort':
        base_model = SortMPNN(dim_in=args.in_dim, dim_out=args.out_dim,num_layers=args.depth, max_nodes=2).cuda().double()
    dtype = dtype_mapping[args.dtype]
    ratio_eps = compute_ratio_flipped(base_model, create_square_graph_with_zero_features,n = n, d = args.in_dim, n_times = 100,depth=args.depth,eps_vals=[10**(-k) for k in range(7)],dtype=dtype)
    # print(f"With eps, we have ratio {ratio_eps}")
    return ratio_eps

def parse_arguments():
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        argparse.Namespace: Parsed arguments with keys `dataset_name`, `radius`, `repeat`, and `all`.

    This function enables command-line customization of the dataset, radius, repeat count, 
    and depth exploration (all depths vs. specific depth).
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument('--dataset_name', type=str, default='lifshiz_comp', help='Name of the dataset to use for training')
    parser.add_argument('--model_type', type=str,default='Sort', help='Flag to run experiments over all depth values')
    return parser.parse_args()

def main(depth):
    """
    Main function to execute the training, testing, and logging of model accuracy across various depth values.

    This function:
    1. Parses the command-line arguments to customize dataset, depth, and repeat settings.
    2. Sets the depth range based on the `--all` argument for exhaustive vs. single depth exploration.
    3. Iterates over the depth range, trains and evaluates the model for each depth, and logs the results.
    4. Prints test accuracy for each depth level for the specified dataset.

    Command-line arguments:
    - `--dataset_name` (str): Specifies the dataset to use.
    - `--radius` (int): Radius value for model depth.
    - `--repeat` (int): Number of times to repeat each experiment.
    - `--all` (bool): Whether to run experiments across all depths.
    """
    # Parse command-line arguments
    args = parse_arguments()
    task, model_type = args.dataset_name, args.model_type
    # Determine depth range based on the dataset and the `--all` flag
    tests = []
    args, task_specific = get_args(depth=depth, gnn_type=model_type, task_type=task)
    
    # Repeat training to calculate average accuracy over specified repeats
    test_acc = train_graphs(args=args)
    return test_acc


if __name__ == "__main__":
    mean = 0
    num_seeds = 10
    lists = []
    for depth in range(1, 10):
        for seed in range(num_seeds):
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            pl.seed_everything(seed, workers=True)
            acc = main(depth = depth)
            mean+=acc/num_seeds
        lists.append(mean)

        print(f"With number of layers {depth}, the ratio is {mean}")
        mean = 0 
    print(lists)