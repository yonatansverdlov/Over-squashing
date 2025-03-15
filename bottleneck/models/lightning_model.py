import torch
from torch import Tensor
from easydict import EasyDict
from models.graph_model import GraphModel
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import Data
import lightning
<<<<<<< HEAD
=======
from utils import compute_os_energy_batched
>>>>>>> add_batched


class LightningModel(lightning.LightningModule):
    """
    PyTorch Lightning Module for training and evaluating a graph neural network model on various datasets.
    
    Args:
        args (EasyDict): Configuration dictionary containing model parameters.
        model (GraphModel): The graph neural network model to be used.
    """
    def __init__(self, args: EasyDict):
        super().__init__()
        self.task_id = args.split_id
        self.lr = args.lr
        self.lr_factor = args.lr_factor
        self.optim_type = args.optim_type
        self.weight_decay = args.wd
        self.task_type = args.task_type
<<<<<<< HEAD
        self.is_mutag = self.task_type in {'MUTAG', 'Protein'}

=======
        self.radius = args.depth
>>>>>>> add_batched
        # Determine if dataset needs continuous features
        self.need_continuous_features = self.task_type in {
            'Cora', 'Actor', 'Corn', 'Texas', 'Wisc', 'Squi',
            'Cham', 'Cite', 'Pubm', 'MUTAG', 'Protein'
        }
        self.model = GraphModel(args)

    def forward(self, X: Data) -> Tensor:
        """Forward pass through the model."""
        return self.model(X)

    def compute_node_embedding(self, X: Data) -> Tensor:
        """Compute node embeddings for the input graph data."""
        return self.model.compute_node_embedding(X)

    def _get_labels_and_mask(self, batch: Data, mask_attr: str):
        """
        Extracts labels and mask for different stages, ensuring safe indexing.
        
        Args:
            batch (Data): Batch of graph data containing node features, labels, and masks.
            mask_attr (str): Attribute name for the mask (e.g., 'train_mask', 'val_mask', 'test_mask').
        
        Returns:
            Tuple[Tensor, Optional[Tensor]]: Extracted labels and mask, or None if not needed.
        """
        mask = getattr(batch, mask_attr, None)

        if mask is None:
            raise ValueError(f"Mask '{mask_attr}' not found in batch.")

        if self.need_continuous_features:
            # Ensure mask is 2D before indexing
            if mask.dim() < 2:
                raise ValueError(f"Expected 2D mask but got shape {mask.shape} for task_id {self.task_id}")

            return batch.y[mask[:, self.task_id]], mask[:, self.task_id]
        
        return batch.y, mask  # Keep the original mask unchanged


    def _shared_step(self, batch: Data, stage: str):
        """
        Generic step used for training, validation, and testing.
        
        Args:
            batch (Data): A batch of graph data.
            stage (str): One of "train", "val", or "test".
        
        Returns:
            Tensor: Computed loss.
        """
        label, mask = self._get_labels_and_mask(batch, f"{stage}_mask")
        if mask is not None:
            batch.root_mask = mask  # Assign root mask if necessary
        
        result = self.model(batch)
        loss = torch.nn.CrossEntropyLoss()(result, label)
        acc = (torch.argmax(result, -1) == label).float().mean()

        self.log(f"{stage}_loss", loss, batch_size=label.size(0))
        self.log(f"{stage}_acc", acc, batch_size=label.size(0))
        return loss

    def training_step(self, batch: Data, _):
        """Computes training loss and accuracy."""
        self.model.train()
        return self._shared_step(batch, "train")

    def validation_step(self, batch: Data, _):
        """Computes validation loss and accuracy."""
        self.model.eval()
<<<<<<< HEAD
=======
        forbenius_energy = compute_os_energy_batched(self.model, batch,option='forb')
        singular_energy = compute_os_energy_batched(self.model, batch,option = 'singular')
        self.log("val_singular_energy", singular_energy, batch_size=batch.y.size(0))
        self.log("val_forbenius_energy", forbenius_energy, batch_size=batch.y.size(0))
>>>>>>> add_batched
        with torch.no_grad(): 
            return self._shared_step(batch, "val")

    def test_step(self, batch: Data, _):
        """Computes test loss and accuracy."""
        self.model.eval()
        with torch.no_grad():
            return self._shared_step(batch, "test")

    def configure_optimizers(self):
        """
        Sets up the optimizer and learning rate scheduler.
        
        Returns:
            Tuple[List[Optimizer], Dict]: List containing the optimizer and a dictionary with the 
                                          learning rate scheduler configuration.
        """
        optimizer_cls = getattr(torch.optim, self.optim_type, torch.optim.Adam)
        optimizer = optimizer_cls(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        lr_scheduler = ReduceLROnPlateau(optimizer, factor=self.lr_factor, mode='max')
        return [optimizer], {"scheduler": lr_scheduler, "interval": "epoch", "monitor": "train_acc"}
