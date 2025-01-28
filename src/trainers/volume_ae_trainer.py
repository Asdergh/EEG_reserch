import torch as th
import tqdm as tq
import os 
import matplotlib.pyplot as plt
import numpy as np


from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image



class VolumeMRITrainer:


    def __init__(
        self,
        target_folder: str,
        model: Module,
        optimizer: Optimizer,
        loader: DataLoader,
        loss: th.Tensor,
        device: str
    ) -> None:
        
        self.tar_f = target_folder
        if not(os.path.exists(self.tar_f)):
            os.mkdir(self.tar_f)
         
        self.model = model
        self.optim = optimizer
        self.loader = loader
        self.device = device
    
    def _save_out_samples_(self, samples_n: int, epoch: int) -> None:
        
        in_samples = next(iter(self.loader))[:samples_n].to(self.device)
        gen_samples = self.model(in_samples)
        show_tensor = th.zeros((
            samples_n * in_samples.size()[1],
            samples_n * in_samples.size()[2],
            samples_n * in_samples.size()[3],
        ))
        idx = 0
        for i in range(samples_n):
            for j in range(samples_n):
                for k in range(samples_n):

                    show_tensor[
                        i * in_samples.size()[1]: (i + 1) * in_samples.size()[1],
                        j * in_samples.size()[2]: (j + 1) * in_samples.size()[2],
                        k * in_samples.size()[3]: (k + 1) * in_samples.size()[3],
                    ] = gen_samples[idx]
                    idx += 1
        
        epoch_path = os.path.join(self.gen_path, f"SampleOnEpoch{epoch}.pt")
        th.save(show_tensor, epoch_path)
    
    def _train_on_epoch_(self) -> float:

        local_loss = 0.0
        for sample in self.loader:

            self.optim.zero_grad()
            gen_samples = self.model(sample)
            loss = self.loss(gen_samples, sample)
            local_loss += loss.item()

            loss.backward()
            self.optim.step()
        
        return local_loss

    def train(self, epoch: int) -> list[float]:

        loss_history = []
        for epoch in range(epoch):

            loss_history.append(self._train_on_epoch_())
            self._save_out_samples_(samples_n=4, epoch=epoch)
        
        return loss_history
    







        
        

            


        

        


        
        