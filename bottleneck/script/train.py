import argparse
import os
import random
import numpy as np
import torch
from easydict import EasyDict
from pytorch_lightning import Trainer, seed_everything, callbacks
from torch_geometric.loader import DataLoader

from models.lightning_model import LightningModel, StopAtValAccCallback
from models.graph_model import GraphModel
from utils import get_args, create_model_dir, return_datasets


def worker_init_fn(seed: int):
    """
    Initializes random seeds for reproducibility in data loading workers.

    Args:
        seed (int): The seed value to ensure consistent data shuffling.
    """
    np.random.seed(seed)
    random.seed(seed)


def train_graphs(args: EasyDict, task_specific: str, task_id: int, seed: int) -> float:
    """
    Train, validate, and test a graph model on the specified dataset.

    Args:
        args (EasyDict): Configuration containing hyperparameters and dataset details.
        task_specific (str): Task-specific identifier used for creating model directories.
        task_id (int): Unique identifier for multi-task settings.
        seed (int): Random seed for reproducibility.

    Returns:
        float: Test accuracy achieved by the model.
    """
    # Set up directories and callbacks
    model_dir, path_to_project = create_model_dir(args, task_specific, seed=seed)
    checkpoint_callback = callbacks.ModelCheckpoint(
        dirpath=model_dir,
        filename='{epoch}-f{val_acc:.5f}',
        save_top_k=10,
        monitor='val_acc',
        save_last=True,
        mode='max'
    )

    # Additional stopping callback for specific task types
    stop_callback = StopAtValAccCallback(target_acc=1.0) if args.task_type in ['Tree', 'Ring', 'CrossRing', 'CliqueRing'] else None
    callbacks_list = [callback for callback in [checkpoint_callback, stop_callback] if callback]

    # Trainer setup
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        enable_progress_bar=True,
        check_val_every_n_epoch=args.eval_every,
        callbacks=callbacks_list,
        default_root_dir=f'{path_to_project}/data/lightning_logs'
    )

    # Load datasets
    X_train, X_test, X_val = return_datasets(args=args)

    # Initialize the graph model and Lightning wrapper
    base_model = GraphModel(args=args)
    model = LightningModel(args=args, task_id=task_id, model=base_model)

    # Prepare data loaders
    train_loader = DataLoader(
        X_train, batch_size=args.batch_size, shuffle=True, num_workers=args.loader_workers,
        persistent_workers=True, worker_init_fn=lambda _: worker_init_fn(seed)
    )
    val_loader = DataLoader(
        X_val, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.loader_workers
    )
    test_loader = DataLoader(
        X_test, batch_size=args.val_batch_size, shuffle=False, pin_memory=True, num_workers=args.loader_workers
    )

    # Train the model
    print('Starting training...')
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint if not using the last model
    if not args.take_last:
        print("Loading best model checkpoint...")
        best_checkpoint_path = checkpoint_callback.best_model_path
        model = LightningModel.load_from_checkpoint(best_checkpoint_path, args=args, task_id=task_id, model=base_model)

    # Evaluate on the test set
    test_results = trainer.test(model, test_loader, verbose=False)
    return test_results[0]['test_acc'] * 100


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument('--dataset_name', type=str, default='Tree', help='Dataset to use for training.')
    parser.add_argument('--radius', type=int, default=2, help='Radius value for model depth.')
    parser.add_argument('--repeat', type=int, default=1, help='Number of training repetitions.')
    parser.add_argument('--all', action='store_true', help='Run experiments across all depth values.')
    parser.add_argument('--model_type', type=str, default='SW', help='Model type for training.')
    return parser.parse_args()


def main():
    """
    Main function to execute training and testing over various depth values.
    """
    args = parse_arguments()
    task, depth, repeats, alls, model_type = args.dataset_name, args.radius, args.repeat, args.all, args.model_type

    # Set depth range based on task type and arguments
    if alls:
        first, end = (2, 16) if task in ['Ring', 'CliqueRing', 'CrossRing'] else (2, 9)
    else:
        first, end = depth, depth + 1

    depth_accuracies = []
    for current_depth in range(first, end):
        args, task_specific = get_args(depth=current_depth, gnn_type=model_type, task_type=task)
        test_acc_avg = 0.0

        # Repeat training to compute average accuracy
        for repeat_idx in range(repeats):
            seed = args.seed
            os.environ["PYTHONHASHSEED"] = str(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)
            seed_everything(seed, workers=True)
            args.split_id = repeat_idx
            test_acc = train_graphs(args=args, task_specific=task_specific, task_id=repeat_idx, seed=seed)
            test_acc_avg += test_acc / repeats

        depth_accuracies.append(test_acc_avg)

    # Display results
    for idx, acc in enumerate(depth_accuracies, start=first):
        print(f"With depth {idx}, the accuracy is {acc:.2f}% on the {task} dataset.")


if __name__ == "__main__":
    main()



