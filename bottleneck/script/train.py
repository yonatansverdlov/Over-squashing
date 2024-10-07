import pytorch_lightning as pl
import torch
from torch_geometric.loader.dataloader import DataLoader

from models.lightning_model import LightningModel
from utils import GNN_TYPE, get_args, create_model_dir, compute_energy, return_datasets
import random
import numpy as np
import argparse

seed = 0
torch.manual_seed(seed)
random.seed(seed)

def train_graphs(args:dict, task:str, depth:int, gnn_type,task_id:int):
    model_dir, path_to_project = create_model_dir(args, task_specific)
    callbacks = [pl.callbacks.ModelCheckpoint(dirpath=model_dir,
                                              filename='{epoch}-f{val_acc:.5f}',
                                              save_top_k=10,
                                              monitor=f'val_acc',
                                              save_last=True, mode='max')]

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
    test_loader = DataLoader(X_test, batch_size=args.test_batch_size, shuffle=False,
                             pin_memory=True, num_workers=args.loader_workers)

    if training:
        print(f'Starting experiment')
        trainer.fit(model, train_loader, val_loader)
    if test:
        checkpoint = torch.load(callbacks[0].best_model_path,weights_only=True)
        if args.take_best:
          model.load_state_dict(checkpoint['state_dict'])
        test_acc = trainer.test(model, test_loader, verbose=False)[0]['test_acc']
        print(f"The test accuracy if {test_acc}")
        # train_acc = trainer.test(model, train_loader, verbose=False)[0]['test_acc']
    if compute_direchlet:
        energy = compute_energy(X_test[0], model=model.model)
        print(f"The direchlet energy is {compute_energy(X_test[0], model=model.model)}")
    print(args)

    return energy, test_acc

parser = argparse.ArgumentParser(description="Process dataset with a specified radius.")

# Add dataset_name as a string argument (positional)
parser.add_argument('--dataset_name', type=str, help='Name of the dataset',default = 'Cora')

# Add radius as an integer argument (positional)
parser.add_argument('--radius', type=int, help='Radius value',default=2)

parser.add_argument('--repeat', type=int, help='Radius value',default=1)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
task = args.dataset_name
depth = args.radius
repeat = args.repeat
tests = []
if task in ['Ring','CliqueRing','CrossRing','Tree']:
    num_layers = depth
else:
    num_layers = 2
args, task_specific = get_args(depth=depth, gnn_type=GNN_TYPE.SW, num_layers=num_layers, task_type=task)
for seed_id in range(repeat):
    energy, test_acc = train_graphs(args = args, task=task,depth=depth,gnn_type=GNN_TYPE.SW,task_id=seed_id)
    tests.append(test_acc)
    #print(energy)                  
    #print(test_acc)
tests = np.array(tests)
print(tests)
print(np.mean(tests))

