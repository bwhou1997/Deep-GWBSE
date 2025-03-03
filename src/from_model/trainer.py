import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter 

class Trainer:
    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, loss, 
                save_path=os.getcwd(),
                model_name="model",
                overwrite=False) -> None:
        """
        Note that different subclasses are expected to put different requirements how `loss` is called.
        We do not impose hard constraints on the function signature of `loss`. 
        See :func:`get_loss`.
        """
        
        
        # Temporary files
        self.save_path = save_path
        self.model_name = model_name
        self.full_save_path = os.path.join(self.save_path, f"{self.model_name}.pth")
        
        # Load the existing model, if it's there
        if os.path.exists(self.full_save_path) and not overwrite:
            print("Saved model loaded.")
            model.load_state_dict(torch.load(self.full_save_path), strict=False)
            self.loaded_from_file = True
        else:
            # FIXME: change to logger
            if overwrite:
                print("Saved model not loaded because it is to be overwritten.")
            else:
                print("Saved model does not exist.")
            self.loaded_from_file = False
        
        # Move the model to GPU, if any
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            # FIXME: change to logger
            print("The program is running on CPUs. Performance may be bad!")
        self.model = model.to(self.device)    
        
        self.training_dataloader = training_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.loss = loss
        
        
        self.tb_writer = SummaryWriter(os.path.join(self.save_path, f"{self.model_name}-tensorboard"))

        pass
    
    #region The real training part
    def train_each_epoch(self, epoch_idx: int):
        """
        Expected to be implemented by subclasses.
        """
        pass
    
    def get_loss(self, **kwargs):
        """
        This method uses `self.loss` to calculate the actual loss of one batch.
        The function signature is intentional left behind,
        as different models have different inputs and outputs.
        Subclasses should override the definition of this method,
        including its function signature.
        """
        #X_batch = X_batch.to(self.device)
        #Y_batch = Y_batch.to(self.device)
        #X_hat = self.model(X_batch)
        #loss = self.loss(X_hat, X_batch).mean().item()
        #return loss
        pass

    def train(self, epoches: int, continued=False):
        """
        The batch size should already be defined in `optimizer`.
        In this method we do not provide hooks for defining the batch size.
        """
        
        if self.loaded_from_file and not continued:
            print("Model loaded from file: no training is done. Set continued to True to train on top of existing model.")
            return

        for epoch in range(epoches):
            self.model.train()
            self.train_each_epoch(epoch)
        
        torch.save(self.model.state_dict(), self.full_save_path)


    def evaluate(self, input=None):
        """
        To be overwritten by subclasses.
        """
        self.model.eval()
        
        pass
    #endregion 

    #region Logging


    def record(self, epoch: int, **kwargs):
        """
        This method is expected to be called by `train_each_epoch`.
        `train_each_epoch` is expected to pass information like loss to this method.
        Details about how this information is collected are left to subclasses to implement.
        """
 
        self.tb_writer.add_scalar("Loss", kwargs["loss"], global_step=epoch)

    #endregion
        