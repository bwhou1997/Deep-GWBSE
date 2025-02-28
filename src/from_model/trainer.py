import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter 

class Trainer:
    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, loss) -> None:
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            # FIXME: change to logger
            print("The program is running on CPUs. Performance may be bad!")
        self.model = model
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.loss = loss
        
        # Temporary files
        self.save_path = os.getcwd()
        
        self.tb_writer = SummaryWriter(os.path.join(self.save_path, "tensorboard"))

        pass
    
    #region The real training part
    def train_each_epoch(self):
        """
        Expected to be implemented by subclasses.
        """
        pass
    
    def get_loss(self, X_batch: torch.Tensor, Y_batch: torch.Tensor):
        """
        Calculate the loss function, with the label being `X_batch` and the predicted output being `Y_batch`,
        both of which are assumed to be batches.

        More efficient implementation may be available.
        Here we give a generic implementation.
        """
        X_batch = X_batch.to(self.device)
        Y_batch = Y_batch.to(self.device)
        X_hat = self.model(X_batch)
        loss = self.loss(X_hat, X_batch).mean().item()
        return loss

    def train(self, epoches: int):
        """
        The batch size should already be defined in `optimizer`.
        In this method we do not provide hooks for defining the batch size.
        """
        for epoch in range(epoches):
            self.model.train()
            self.train_each_epoch()
            
            self.get_loss()
            # Now we collect intermediate variables and pass them to self.record
            self.record()

    
    def evaluate(self):
        pass
    #endregion 

    #region Logging


    def record(self, loss: float):
        """
        This method is expected to be called by `train_each_epoch`.
        `train_each_epoch` is expected to pass information like loss to this method.
        Details about how this information is collected are left to subclasses to implement.
        """

        # Use Tensorboard?
        self.tb_writer.add_scalar("Loss", loss)

    #endregion
        