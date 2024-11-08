"""
Lighting model.
"""
import pytorch_lightning as pl
import torch
from torch import Tensor
from easydict import EasyDict
from models.graph_model import GraphModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data

class StopAtValAccCallback(pl.Callback):
    def __init__(self, target_acc=1.0):
        """
        Callback for early stopping in Over-squashing tasks.
        """
        super().__init__()
        self.target_acc = target_acc

    def on_validation_epoch_end(self, trainer, _):
        # Access the logged metrics
        val_acc = trainer.callback_metrics.get('val_acc')
        # Check if the validation accuracy has reached or exceeded the target
        if val_acc is not None and val_acc >= self.target_acc:
            trainer.should_stop = True
            print(f" Stopping training as `val_acc` reached {val_acc:.2f}")
        else:
            print(f" The current val accuracy is {val_acc}")

class LightningModel(pl.LightningModule):
    def __init__(self, args: EasyDict,task_id = 0):
        """
        The graph Model.
        Args:
            args: The config.
        """
        super().__init__()
        self.task_id = task_id
        self.lr = args.lr
        self.lr_factor = args.lr_factor
        self.optim_type = args.optim_type
        self.wd = args.wd
        self.task_type = args.task_type
        self.single_graph = ['Cora','Actor','Corn','Texas','Wisc','Squir','Cham','Cite','Pubm']
        self.is_real = str(self.task_type) in self.single_graph
        args.global_task = self.is_real
        self.model = GraphModel(args=args)

    def forward(self, X:Data)->Tensor:
        return self.model(X)
    
    def compute_node_embedding(self,X):
        return self.model.compute_node_embedding(X)

    def training_step(self, batch: Data, _) -> torch.float:
        """
        The training step.
        Args:
            batch: The batch.
            batch_idx: The batch index-not used.

        Returns: The loss.

        """
        self.model.train()
        label = batch.y if not self.is_real else batch.y[batch.train_mask[:,self.task_id]] 
        batch.root_mask = batch.train_mask if not self.is_real else batch.train_mask[:,self.task_id]
        result = self.model(batch)
        loss = torch.nn.CrossEntropyLoss()(result, label)
        acc = (torch.argmax(result, -1) == label).float().mean()
        self.log("train_loss", loss, batch_size=label.size(0))
        self.log('train_acc', acc, batch_size=label.size(0))
        return loss

    def validation_step(self, batch: Data, _):
        """
        The validation step.
        Args:
            batch: The batch.
            batch_idx: The batch index-not used.

        Returns: The loss.

        """
        self.model.eval()
        label = batch.y if not self.is_real else batch.y[batch.val_mask[:,self.task_id]] 
        batch.root_mask = batch.val_mask if not self.is_real else batch.val_mask[:,self.task_id]
        with torch.no_grad():
            result = self.model(batch)
            loss = torch.nn.CrossEntropyLoss()(result, label)
            acc = (torch.argmax(result, -1) == label).float().mean()
        self.log("val_loss", loss, batch_size=label.size(0))
        self.log("val_acc", acc, batch_size=label.size(0))
        return loss

    def test_step(self, batch: Data, _):
        """
        The validation step.
        Args:
            batch: The batch.
            batch_idx: The batch index-not used.

        Returns: The loss.

        """
        self.model.eval()
        label = batch.y if not self.is_real else batch.y[batch.test_mask[:,self.task_id]] 
        batch.root_mask = batch.test_mask if not self.is_real else batch.test_mask[:,self.task_id]
        with torch.no_grad():
            result = self.model(batch)
            loss = torch.nn.CrossEntropyLoss()(result, label)
            acc = (torch.argmax(result, -1) == label).float().mean()
        self.log("test_loss", loss, batch_size=label.size(0))
        self.log('test_acc', acc, batch_size=label.size(0))
        return loss

    def configure_optimizers(self):
        """
        Return optimizer.
        """
        if self.optim_type == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.wd)  # 0.001
        elif self.optim_type == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.wd)  # 0.001
        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, threshold_mode='abs', mode='max')
        lr_scheduler_config = {
            "scheduler": lr_scheduler,
            "interval": "epoch",
            "monitor": "train_acc",
        }

        return [optimizer], lr_scheduler_config
