import pytorch_lightning as pl
import torch
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel, StopAtValAccCallback
from utils import get_args, create_model_dir, compute_energy, return_datasets
from models.graph_model import GraphModel
import argparse
from easydict import EasyDict
import random
import os
# Set random seed for reproducibility
import torch
import random
import numpy as np

seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
pl.seed_everything(seed, workers=True)
# Ensure PyTorch uses deterministic algorithms

import torch

def compare_model_weights(model1, model2, epsilon=1e-6):
    """
    Compares the weights of two models to check if they are identical up to a given tolerance (epsilon).
    
    Args:
        model1: First PyTorch model.
        model2: Second PyTorch model.
        epsilon: Tolerance level for comparing the weights.
        
    Returns:
        bool: True if all weights are identical up to epsilon, False otherwise.
    """
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if name1 != name2:
            print(f"Layer name mismatch: {name1} != {name2}")
            return False
        
        if not torch.allclose(param1, param2, atol=epsilon):
            print(name1)
            print(f"Weights in layer {name1} differ by more than {epsilon}.")
            return False
    
    print("All corresponding weights are identical up to epsilon.")
    return True

# Example usage
# model1 = MyModel()  # Initialize or load your first model
# model2 = MyModel()  # Initialize or load your second model
# result = compare_model_weights(model1, model2)


def train_graphs(args: EasyDict,task_specific:str, task_id: int,take_last = False):
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
    model_dir, path_to_project = create_model_dir(args, task_specific)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-f{val_acc:.5f}',
        save_top_k=10,
        monitor='val_acc',
        save_last=True,
        mode='max'
    )
    if args.task_type in ['Tree','Ring','CrossRing','CliqueRing']:
        stop_callback = StopAtValAccCallback(target_acc=1.0)
        callbacks = [checkpoint_callback, stop_callback]
    else:
        callbacks = [checkpoint_callback]

    # Initialize the trainer with the specified callbacks and configurations
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=True,
        check_val_every_n_epoch=args.eval_every,
        callbacks=callbacks,
        default_root_dir=f'{path_to_project}/data/lightning_logs'
    )

    # Load datasets based on task type
    X_train, X_test, X_val = return_datasets(args=args)

    # Initialize model for the specified task
    base_model = GraphModel(args=args)
    model = LightningModel(args=args, task_id=task_id,model=base_model)

    # Create dataloaders for training, validation, and testing
    train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers, persistent_workers=True)
    val_loader = DataLoader(X_val, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.loader_workers)
    test_loader = DataLoader(X_test, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.loader_workers)

    # Train the model using the train and validation loaders
    print('Starting experiment')
    trainer.fit(model, train_loader, val_loader)

    # Test the model using the best checkpoint saved during validation
    if not take_last:
        best_checkpoint_path = checkpoint_callback.best_model_path
        model = LightningModel.load_from_checkpoint(best_checkpoint_path, args=args, task_id=task_id,model = base_model)
    test_results = trainer.test(model, test_loader, verbose=False)
    test_acc = test_results[0]['test_acc'] * 100
 
    # Compute Dirichlet energy if specified in args
    return test_acc

def parse_arguments():
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        argparse.Namespace: Parsed arguments with keys `dataset_name`, `radius`, `repeat`, and `all`.

    This function enables command-line customization of the dataset, radius, repeat count, 
    and depth exploration (all depths vs. specific depth).
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument('--dataset_name', type=str, default='PROTEIN', help='Name of the dataset to use for training')
    parser.add_argument('--radius', type=int, default=2, help='Radius value for model depth (e.g., neighborhood size)')
    parser.add_argument('--repeat', type=int, default=1, help='Number of times to repeat training for averaging')
    parser.add_argument('--all', type=bool,default=False, help='Flag to run experiments over all depth values')
    return parser.parse_args()

def main():
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
    task, depth, repeats, alls = args.dataset_name, args.radius, args.repeat, args.all

    # Determine depth range based on the dataset and the `--all` flag
    first, end = (2, 21) if alls and task in ['Ring', 'CliqueRing', 'CrossRing'] else (2, 9) if alls else (depth, depth + 1)

    results = {}
    for depth in range(first, end):
        test_acc_sum = 0.0
        tests = []
        args, task_specific = get_args(depth=depth, gnn_type='SW', task_type=task)
        
        # Repeat training to calculate average accuracy over specified repeats
        for i in range(repeats):
            args.split_id = i
            test_acc = train_graphs(args=args,task_specific = task_specific, task_id=i,take_last = True)
            test_acc_sum += test_acc
            tests.append(test_acc)

        # Store average test accuracy for current depth
        results[depth] = test_acc_sum / repeats
        print(tests)
        print(f"On radius {depth} the accuracy is {results[depth]:.2f}% on the {task} dataset")
        print(args)

if __name__ == "__main__":
    main()


