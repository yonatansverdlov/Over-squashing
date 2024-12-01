import pytorch_lightning as pl
import torch
from torch import Tensor
from easydict import EasyDict
from models.graph_model import GraphModel
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

class StopAtValAccCallback(pl.Callback):
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


class LightningModel(pl.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a graph neural network model on various datasets.
    
    Args:
        args (EasyDict): Configuration dictionary containing model parameters.
        task_id (int): Task index for multi-task training. Default is 0.
    """
    def __init__(self, args: EasyDict,model:GraphModel, task_id=0):
        super().__init__()
        self.task_id = task_id  # Identifier for the current task in multi-task settings
        self.lr = args.lr  # Learning rate for the optimizer
        self.lr_factor = args.lr_factor  # Factor by which the learning rate decreases on plateau
        self.optim_type = args.optim_type  # Optimizer type (e.g., 'Adam' or 'AdamW')
        self.weight_decay = args.wd  # Weight decay for regularization
        self.task_type = args.task_type  # Task type, affecting dataset and model structure
        self.is_mutag = self.task_type in ['MUTAG','PROTEIN']
        # Determine if the task is on a single-graph dataset
        self.single_graph_datasets = {'Cora', 'Actor', 'Corn', 'Texas', 'Wisc', 'Squi', 
                                      'Cham', 'Cite', 'Pubm', 'MUTAG', 'PROTEIN', 'lifshiz_comp'}
        self.need_continuous_features = self.task_type in self.single_graph_datasets
        # Initialize the graph model based on the given configuration
        self.model = model

    def forward(self, X: Data) -> Tensor:
        """
        Forward pass through the model.
        
        Args:
            X (Data): Torch Geometric Data object containing node features and edge indices.
        
        Returns:
            Tensor: Predicted outputs for the nodes specified by the root mask.
        """
        return self.model(X)

    def compute_node_embedding(self, X: Data) -> Tensor:
        """
        Compute node embeddings for the input graph data.
        
        Args:
            X (Data): Torch Geometric Data object containing node features and edge indices.
        
        Returns:
            Tensor: Node embeddings computed by the model.
        """
        return self.model.compute_node_embedding(X)

    def training_step(self, batch: Data, _):
        """
        Computes training loss and accuracy for a batch and logs them.
        
        Args:
            batch (Data): A batch of graph data containing node features, labels, and masks.
        
        Returns:
            Tensor: Training loss computed using cross-entropy.
        """
        self.model.train()  # Set model to training mode
        # Select the appropriate labels and root mask based on task type
        if self.is_mutag:
            label = batch.y
        else:
            label = batch.y if not self.need_continuous_features else batch.y[batch.train_mask[:, self.task_id]]
            batch.root_mask = batch.train_mask if not self.need_continuous_features else batch.train_mask[:, self.task_id]
        
        # Forward pass through the model and compute loss
        result = self.model(batch)
        loss = torch.nn.CrossEntropyLoss()(result, label)  # Cross-entropy loss for classification
        
        # Calculate accuracy as the mean of correct predictions
        acc = (torch.argmax(result, -1) == label).float().mean()
        
        # Log training loss and accuracy
        self.log("train_loss", loss, batch_size=label.size(0))
        self.log("train_acc", acc, batch_size=label.size(0))
        return loss

    def validation_step(self, batch: Data, _):
        """
        Computes validation loss and accuracy for a batch and logs them.
        
        Args:
            batch (Data): A batch of validation graph data.
        
        Returns:
            Tensor: Validation loss computed using cross-entropy.
        """
        self.model.eval()  # Set model to evaluation mode
        if self.is_mutag:
            label = batch.y
        else:
            # Select the appropriate labels and root mask for validation based on task type
            label = batch.y if not self.need_continuous_features else batch.y[batch.val_mask[:, self.task_id]]
            batch.root_mask = batch.val_mask if not self.need_continuous_features else batch.val_mask[:, self.task_id]
        
        # Disable gradient computation for validation
        with torch.no_grad():
            result = self.model(batch)
            loss = torch.nn.CrossEntropyLoss()(result, label)  # Cross-entropy loss for validation
            acc = (torch.argmax(result, -1) == label).float().mean()  # Calculate accuracy
        
        # Log validation loss and accuracy
        self.log("val_loss", loss, batch_size=label.size(0))
        self.log("val_acc", acc, batch_size=label.size(0))
        return loss

    def test_step(self, batch: Data, _):
        """
        Computes test loss and accuracy for a batch and logs them.
        
        Args:
            batch (Data): A batch of test graph data.
        
        Returns:
            Tensor: Test loss computed using cross-entropy.
        """
        self.model.eval()  # Set model to evaluation mode
        # Select appropriate labels and root mask for test based on task type
        if self.is_mutag:
            label = batch.y
        else:
            label = batch.y if not self.need_continuous_features else batch.y[batch.test_mask[:, self.task_id]]
            batch.root_mask = batch.test_mask if not self.need_continuous_features else batch.test_mask[:, self.task_id]
        
        # Disable gradient computation for test
        with torch.no_grad():
            result = self.model(batch)
            loss = torch.nn.CrossEntropyLoss()(result, label)  # Cross-entropy loss for testing
            acc = (torch.argmax(result, -1) == label).float().mean()  # Calculate accuracy
        
        # Log test loss and accuracy
        self.log("test_loss", loss, batch_size=label.size(0))
        self.log('test_acc', acc, batch_size=label.size(0))
        return loss

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        
        Returns:
            Tuple[List[Optimizer], Dict]: List containing the optimizer and a dictionary with the 
                                          learning rate scheduler configuration.
        """
        # Select optimizer type and initialize with weight decay
        optimizer_cls = torch.optim.Adam if self.optim_type == 'Adam' else torch.optim.AdamW
        optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        # Configure learning rate scheduler to reduce on plateau based on train accuracy
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, mode='max')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "train_acc",  # Monitors training accuracy
        }
        return [optimizer], lr_scheduler_config

