import argparse
from easydict import EasyDict
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel
from utils import (
    get_args, return_dataloader, average_oversmoothing_metric, 
    create_trainer, MetricAggregationCallback, fix_seed
)

def parse_arguments() -> EasyDict:
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        EasyDict: Parsed arguments in an easy-to-use dictionary.
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument('--task_type', type=str, default='Ring', help='Dataset to use for training.')
    parser.add_argument('--min_radius', type=int, default=2, help='Minimum radius value for model depth.')
    parser.add_argument('--max_radius', type=int, default=3, help='Maximum radius value for model depth.')
    parser.add_argument('--repeat', type=int, default=1, help='Number of training repetitions.')
    parser.add_argument('--model_type', type=str, default='GIN', help='Model type for training.')
    
    return EasyDict(vars(parser.parse_args()))  # Convert argparse.Namespace to EasyDict


def train_graphs(args: EasyDict, task_specific: str, metric_callback, measure_oversmoothing=False) -> tuple:
    """
    Train and evaluate a graph model on a dataset.

    Args:
        args (EasyDict): Experiment configuration.
        task_specific (str): Task-specific identifier.
        metric_callback (MetricAggregationCallback): Callback for tracking metrics.
        measure_oversmoothing (bool): Whether to measure oversmoothing.

    Returns:
        tuple: (Test accuracy, MAD energy if oversmoothing is measured)
    """
    trainer, checkpoint_callback = create_trainer(args, task_specific, metric_callback)
    
    # Load dataset
    train_loader, val_loader, test_loader, X_val = return_dataloader(args)

    # Initialize graph model
    model = LightningModel(args=args)

    # Train model
    print('Starting training...')
    trainer.fit(model, train_loader, val_loader)

    # Load best checkpoint if necessary
    if not args.take_last:
        print("Loading best model checkpoint...")
        model = LightningModel.load_from_checkpoint(
            checkpoint_callback.best_model_path, args=args, model=model.model
        )
    energy = trainer.callback_metrics.get("grad_ratio")
    # Test model
    test_results = trainer.test(model, test_loader, verbose=False)
    test_acc_model = test_results[0]['test_acc'] * 100


    if measure_oversmoothing:
        val_loader = DataLoader(X_val, batch_size=1, shuffle=False, 
                                pin_memory=True, num_workers=args.loader_workers)
        energy = average_oversmoothing_metric(model, val_loader)
        print(f"MAD Energy: {energy:.4f}")

    return test_acc_model, energy
def main():
    """
    Main function to run training experiments for various model depths.
    """
    args = parse_arguments()

    # Extract command-line arguments
    task, min_radius, max_radius, repeats, model_type = (
        args.task_type, args.min_radius, args.max_radius, args.repeat, args.model_type
    )
    accuracy_results = {}
    energy_results = {} 
    # Iterate over depth range
    for current_depth in range(min_radius, max_radius):
        args, task_specific = get_args(depth=current_depth, gnn_type=model_type, task_type=task)
        metric_callback = MetricAggregationCallback(eval_every=args.eval_every)

        for repeat_idx in range(repeats):
            fix_seed(args.seed)
            args.split_id = repeat_idx
            _, energy = train_graphs(args, task_specific, metric_callback, measure_oversmoothing=False)
        
        # Retrieve best epoch results
        best_mean, best_std = metric_callback.get_best_epoch()
        accuracy_results[current_depth] = (best_mean, best_std)
        energy_results[current_depth] = energy  # Store energy for each radius

    # Display final accuracy and energy results
    print("\nFinal Accuracy and Energy Results for All Radii:")
    for radius, (mean_acc, std_acc) in accuracy_results.items():
        energy = energy_results[radius]
        print(f"GNN: {model_type} |Task: {task} | Radius: {radius} | Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}% |")

if __name__ == "__main__":
    main()