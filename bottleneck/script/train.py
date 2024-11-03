import pytorch_lightning as pl
import torch
from torch_geometric.loader.dataloader import DataLoader

from models.lightning_model import LightningModel, StopAtValAccCallback
from utils import GNN_TYPE, get_args, create_model_dir, compute_energy, return_datasets
import random
import numpy as np
import argparse

seed = 0
torch.manual_seed(seed)
random.seed(seed)

def train_graphs(args:dict, task_id:int):
    model_dir, path_to_project = create_model_dir(args, task_specific)
    stop_at_val_acc_callback = StopAtValAccCallback(target_acc=1.0)
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=model_dir,
                                              filename='{epoch}-f{val_acc:.5f}',
                                              save_top_k=10,
                                              monitor=f'val_acc',
                                              save_last=True, mode='max'),
                                              stop_at_val_acc_callback]

    trainer = pl.Trainer(max_epochs=args.max_epochs, accelerator='gpu' if torch.cuda.is_available() else 'cpu', enable_progress_bar=True,
                         check_val_every_n_epoch=args.eval_every, callbacks=callbacks,
                         default_root_dir=f'{path_to_project}/data/lightning_logs')
    # The datasets.
    X_train, X_test, X_val = return_datasets(args=args)

    # Create model.
    model = LightningModel(args=args,task_id=task_id)
    training = True
    test = True
    compute_direchlet = False
    energy = 0.0
    train_loader = DataLoader(X_train, batch_size=args.batch_size, shuffle=True,
                              pin_memory=True, num_workers=args.loader_workers)
    val_loader = DataLoader(X_val, batch_size=args.val_batch_size, shuffle=False,
                            pin_memory=True, num_workers=args.loader_workers)
    test_loader = DataLoader(X_test, batch_size=args.val_batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.loader_workers)

    if training:
        print(f'Starting experiment')
        trainer.fit(model, train_loader, val_loader)
    if test:
        best_checkpoint_path = callbacks[0].best_model_path
        model = LightningModel.load_from_checkpoint(
            checkpoint_path=best_checkpoint_path,
            args=args,      # Pass any required arguments your model needs
            task_id=task_id  # Pass other necessary parameters
        )
        test_acc = trainer.test(model, test_loader,verbose = False)[0]['test_acc']
    if compute_direchlet:
        energy = compute_energy(X_test[0], model=model.model)
        print(f"The direchlet energy is {compute_energy(X_test[0], model=model.model)}")

    return energy, test_acc

parser = argparse.ArgumentParser(description="Process dataset with a specified radius.")

# Add dataset_name as a string argument (positional)
parser.add_argument('--dataset_name', type=str, help='Name of the dataset',default = 'Ring')

# Add radius as an integer argument (positional)
parser.add_argument('--radius', type=int, help='Radius value',default=20)

parser.add_argument('--repeat', type=int, help='Number of repeats',default=1)

parser.add_argument('--all', type=bool, help='Run all',default=False)
# Parse the arguments
args = parser.parse_args()

# Access the arguments
task = args.dataset_name
depth = args.radius
repeats = args.repeat
alls = args.all
tests = dict()
if alls:
    if task in ['Ring','CliqueRing','CrossRing']:
        first, end = 2, 16
    else:
        first, end = 2, 9
else:
    first, end = depth, depth + 1

for depth in range(first,end):
    test_accs = 0.0
    args, task_specific = get_args(depth=depth, gnn_type=GNN_TYPE.SW, task_type=task)
    for repeat in range(repeats):
        energy, test_acc = train_graphs(args = args,task_id=repeat)
        test_accs+=test_acc
    tests[depth] = test_accs / repeats
for depth in range(first, end):
    print(f"On radius {depth} the accuracy is {tests[depth]} on the {str(task)}")

