import argparse
from easydict import EasyDict
from torch_geometric.loader import DataLoader
from models.lightning_model import LightningModel
from models.graph_model import GraphModel
from utils import get_args, return_datasets, average_oversmoothing_metric, create_trainer, worker_init_fn, MetricAggregationCallback, fix_seed

def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for dataset training configuration.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Train graph models on specified datasets.")
    parser.add_argument('--task_type', type=str, default='Ring', help='Dataset to use for training.')
    parser.add_argument('--min_radius', type=int, default=2, help='Radius value for model depth.')
    parser.add_argument('--max_radius', type=int, default=3, help='Radius value for model depth.')
    parser.add_argument('--repeat', type=int, default=1, help='Number of training repetitions.')
    parser.add_argument('--model_type', type=str, default='SW', help='Model type for training.')
    return parser.parse_args()

def train_graphs(args: EasyDict, task_specific: str, seed: int, metric_callback,measure_oversmoothing = False) -> float:
    trainer, checkpoint_callback = create_trainer(args, task_specific, seed, metric_callback)
    energy = 0.0
    # Load datasets
    X_train, X_test, X_val = return_datasets(args=args)

    # Initialize the graph model and Lightning wrapper
    base_model = GraphModel(args=args)
    model = LightningModel(args=args, model=base_model)

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
        model = LightningModel.load_from_checkpoint(best_checkpoint_path, args=args, model=base_model)
    # Evaluate on the test set
    test_results = trainer.test(model, test_loader, verbose=False)
    test_acc_model = test_results[0]['test_acc'] * 100
    print(f"Test accuracy: {test_acc_model:.2f}%")
    if measure_oversmoothing:
        val_loader = DataLoader(
        X_val, batch_size=1, shuffle=False, pin_memory=True, num_workers=args.loader_workers
                    )
        energy = average_oversmoothing_metric(model, val_loader)
        print(f"The MAD energy {energy:.4f}")

    return test_acc_model, energy

def main():
    args = parse_arguments()
    task, min_radius, max_radius, repeats, model_type = args.task_type, args.min_radius,args.max_radius, args.repeat, args.model_type
    accuracy_results = {}
    # Set up depth range
    for current_depth in range(min_radius, max_radius):
        args, task_specific = get_args(depth=current_depth, gnn_type=model_type, task_type=task)
        metric_callback = MetricAggregationCallback(eval_every=args.eval_every)
        seed = args.seed
        for repeat_idx in range(repeats):
            fix_seed(seed)  
            args.split_id = repeat_idx
            train_graphs(args, task_specific, seed, metric_callback)
        best_mean, best_std = metric_callback.get_best_epoch()
        accuracy_results[current_depth] = (best_mean, best_std)
    print("\nFinal Accuracy Results for All Radii:")
    for radius, (mean_acc,std_acc) in accuracy_results.items():
        print(f"On task {task}, With Radius {radius} the accuracy is {mean_acc:.2f}%, with std {std_acc:.2f}%")

if __name__ == "__main__":
    main()



