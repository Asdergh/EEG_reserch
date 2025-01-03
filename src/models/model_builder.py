import torch as th
import yaml
from ae_net import AeEegNet
from layers import *


class ModelBuilder:


    def __init__(self, params: dict | str) -> None:
        
        self._models = {
            "AeEegNet": AeEegNet,
        }
        if isinstance(params, str):
            with open(params, "r") as yaml_file:
                params = yaml.full_load(yaml_file)

        self.models = []
        for model in params:
            self.models.append(self._models[model](**params[model]))
    
    @property
    def model(self):
        return self.models


if __name__ == "__main__":

    models = ModelBuilder(params="C:\\Users\\1\\Desktop\\PythonProjects\\EegProject\\ae_eeg.yaml")
    print(models.models[0].named_children)

