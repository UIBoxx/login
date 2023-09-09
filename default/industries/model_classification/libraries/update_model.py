import os
import pathlib

class UpdateModel:

    def __init__(self):
        
        self.active_model_dir= "../models/active"
        self.train_model_dir =  "../models/train"

        self.script_dir = pathlib.Path(__file__).parent.absolute()
      
        self.active_models_dir = os.path.join(self.script_dir, self.active_model_dir)
        self.train_models_dir = os.path.join(self.script_dir, self.train_model_dir)

    
    def prediction_on_train_model(self):
        pass
        