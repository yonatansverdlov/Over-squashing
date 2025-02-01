import torch
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

def calculate_proportion_of_unique_norms(data, epsilon):
    """
    Calculate the proportion of unique norms in the centralized molecule.
    
    Args:
        data (torch_geometric.data.Data): A single molecule data object.
        epsilon (float): The threshold for distinguishing two norms.
    
    Returns:
        float: The proportion of unique norms.
    """
    # Center the molecule by its positional mean
    centralized_pos = data.pos - data.pos.mean(dim=0, keepdim=True)

    # Calculate norms of all nodes
    norms = torch.norm(centralized_pos, dim=1)

    # Sort norms for efficient difference computation
    sorted_norms, _ = torch.sort(norms)

    # Compute pairwise differences between consecutive norms
    diffs = torch.diff(sorted_norms)

    # Identify unique norms by comparing differences with epsilon
    unique_mask = diffs >= epsilon
    unique_count = unique_mask.sum().item() + 1  # Add 1 for the first unique element

    return unique_count / len(norms)

def mean_proportion_unique_norms(dataset, epsilon):
    """
    Calculate the mean proportion of unique norms over all molecules in the dataset.

    Args:
        dataset (torch_geometric.data.Dataset): The QM9 dataset.
        epsilon (float): The threshold for distinguishing two norms.

    Returns:
        float: The mean proportion of unique norms over all molecules.
    """
    proportions = []
    for data in dataset:
        proportions.append(calculate_proportion_of_unique_norms(data, epsilon))

    return torch.tensor(proportions).mean().item()

# Load QM9 dataset
dataset = QM9(root='./data/QM9')
train_dataset = dataset  # Split for training data
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

# Set epsilon value
epsilon = 1e-2
# Calculate mean proportion of unique norms in training set
mean_proportion = mean_proportion_unique_norms(train_loader.dataset, epsilon)
print(f"Mean proportion of unique norms: {mean_proportion}")
