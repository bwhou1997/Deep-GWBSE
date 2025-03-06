import os
import logging
import time
import torch
from tqdm import tqdm
import math
from torch.utils.tensorboard import SummaryWriter 

class Trainer:
    def __init__(self, model, training_dataloader, validation_dataloader, optimizer, loss, 
                save_path=os.getcwd(),
                model_name="model",
                overwrite=False,
                checkpoint=False,
                best_model=False) -> None:
        """
        `kwargs` includes 
        - `overwrite`: set to `True` when we do not want to reuse the model stored in previous trainings.
        - `checkpoint`: set to `True` to save the model each time a epoch finishes.
        - `best_model`: set to `True` to save the model with the lowest loss in `model_name_best.pth`.

        Note that different subclasses are expected to put different requirements how `loss` is called.
        We do not impose hard constraints on the function signature of `loss`. 
        See :func:`get_loss`.
        """
        
        
        # Temporary files
        # Saving models
        self.save_path = save_path
        self.model_name = model_name
        self.current_model_path = os.path.join(self.save_path, f"{self.model_name}.pth")
        self.best_model_path = os.path.join(self.save_path, f"{self.model_name}_best.pth")
        self.best_model = best_model
        self.checkpoint = checkpoint
        self.minimum_validation_loss = math.inf
        
        # Logging
        self.logger = logging.getLogger(f"logger of model {model_name}")
        self.logger.setLevel(logging.DEBUG)
        self.verbose_logger = logging.getLogger(f"logger about everything of model {model_name}")
        self.verbose_logger.setLevel(logging.DEBUG)
        # The log file
        self.train_log_path = os.path.join(self.save_path, "log.txt")
        file_handler = logging.FileHandler(self.train_log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        # Everything going to self.logger will go to both the log file and the console;
        # everything going to self.verbose_logger will only go to the file
        self.logger.addHandler(file_handler)
        self.verbose_logger.addHandler(file_handler)
        # Print the log to the console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(console_handler)
       
        # Load the existing model, if it's there
        if os.path.exists(self.current_model_path) and not overwrite:
            self.logger.info("Saved model loaded.")
            model.load_state_dict(torch.load(self.current_model_path), strict=False)
            self.loaded_from_file = True
        else:
            if overwrite:
                self.logger.info("Saved model not loaded because it is to be overwritten.")
            else:
                self.logger.info("Saved model does not exist.")
            self.loaded_from_file = False
        
        # Move the model to GPU, if any
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cpu":
            self.logger.warn("The program is running on CPUs. Performance may be bad!")
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
        What is presented here is a generic training procedure.
        The method can be overriden by another procedure in subclasses.
        In this case, do not forget to call `self.record` at the end of each epoch.
        """

        total_loss = 0.0

        start_time = time.process_time()
        for x, _ in tqdm(self.training_dataloader, f"Epoch {epoch_idx+1}"):
            x = x.to(self.device)
            self.optimizer.zero_grad()
            this_loss = self.get_loss(x)
            this_loss.backward()
            self.optimizer.step()
            total_loss += this_loss.item()
        
        validation_loss = self.validate()
        end_time = time.process_time()
        self.record(epoch_idx, 
                    training_loss=total_loss / len(self.training_dataloader),
                    validation_loss=validation_loss,
                    elapsed_time=end_time-start_time)
    
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
            self.logger.warn("Model loaded from file: no training is done. Set continued to True to train on top of existing model.")
            return
 
        for epoch in range(epoches):
            self.model.train()
            self.train_each_epoch(epoch)
            if self.checkpoint:
                torch.save(self.model.state_dict(), self.current_model_path)
        
        torch.save(self.model.state_dict(), self.current_model_path)
        self.verbose_logger.info("The final model saved. Training ends.")


    def evaluate(self, input=None):
        """
        To be overwritten by subclasses.
        This method takes an optional `input` and return the predication of the model based on `input`.
        When `input` is not given, its default value is the first batch in the validation dataset.
        The return values should better be NumPy arrays,
        instead of tensors or tensors on GPUs:
        when the latter is needed, you can always directly call `self.model(...)`.
        """
        self.model.eval()
        # ...
        pass
    
    def validate(self, input=None):
        """
        To be overwritten by subclasses.
        This method takes an optional `input` and return the loss of `input`.
        When `input` is not given, its default value is the first batch in the validation dataset.
        Used to calculate e.g. the validation loss.
        """
        self.model.eval()
        # ...
        pass
    #endregion 

    #region Logging


    def record(self, epoch: int, **kwargs):
        """
        This method is expected to be called by `train_each_epoch`.
        It should be called once an epoch finishes.
        `train_each_epoch` is expected to pass information like loss to this method.
        Details about how this information is collected are left to subclasses to implement.
        """
 
        training_loss = kwargs["training_loss"]
        validation_loss = kwargs["validation_loss"]
        elapsed_time = kwargs["elapsed_time"]

        self.tb_writer.add_scalar("Training loss", training_loss, global_step=epoch)
        self.tb_writer.add_scalar("Validation loss", validation_loss, global_step=epoch)

        if self.minimum_validation_loss > validation_loss and self.best_model:
            torch.save(self.model.state_dict(), self.best_model_path)
            self.minimum_validation_loss = validation_loss
            self.verbose_logger.info(f"Eopch {epoch+1} finishes in {elapsed_time:.1f}s | Training loss {training_loss:.2f} | validation loss {validation_loss:.2f} | Best model")
        else:
            self.verbose_logger.info(f"Eopch {epoch+1} finishes in {elapsed_time:.1f}s | Training loss {training_loss:.2f} | validation loss {validation_loss:.2f}")
        

    #endregion
        