class Trainer:
    def __init__(self, model, dataloader, optimizer, loss) -> None:
        pass
    
    def train_each_batch(self):
        """
        To be implemented by daughter classes.
        """
        pass
    
    def train(self):
        pass
    
    def evaluate(self):
        pass
    
    def record(self):
        # Use Tensorboard?
        pass
