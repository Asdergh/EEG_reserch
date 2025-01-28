import torch as th
import tqdm as tq
import os

from torch.optim import Optimizer
from torch.nn import Module
from torch.utils.data import DataLoader




class Eeg2BOLDDmriTrainer:


    def __init__(
        self,
        model: Module,
        optim: Optimizer,
        loss_fn: Module,
        loader: DataLoader,
        device: str = "cpu",
    ) -> None:

        super().__init__()
        self.device = device
        self.model = model
        self.optim = optim
        self.loss_fn = loss_fn
        self.loader = loader

    def _train_on_epoch(self, epoch: int) -> float:
        
        total_local_loss = 0.0
        for (mel, vol_sample) in tq.tqdm(self.loader, colour="red", ascii=">="):
            
            self.optim.zero_grad()
            model_out = self._model_(mel.to(self.device))
            loss = self._loss_fn_(model_out, vol_sample.to(self.device))
            total_local_loss += loss.item()

            loss.backward()
            self.optim.step()

            del mel, vol_sample
            th.cuda.empty_cache()
        
        return total_local_loss

    def train(self, epochs: int) -> list[float]:

        losses = []
        for epoch in tq.tqdm(range(epochs), colour="green", ascii="=>"):
        
            loss = self._train_on_epoch(epoch=epoch)
            losses.append(loss)
            print(f"EPOCH: [{epoch}]; LOSS: [{loss}]")
        
        return losses



if __name__ == 