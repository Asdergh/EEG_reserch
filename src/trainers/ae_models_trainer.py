import torch as th
import tqdm as tq
import os 
import matplotlib.pyplot as plt
import numpy as np


from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.transforms.functional import to_pil_image


class AeEegNetTrainer:


    def __init__(
        self,
        run_folder: str,
        model: Module,
        loss: Module,
        optim: Optimizer,
        loader: DataLoader,
        device: str = "cuda"
    ) -> None:
        
        self.run_folder = run_folder
        self.device = device
        self.model = model
        self.loss = loss
        self.optim = optim
        self.loader = loader
    

    def _save_out_samples(
            self,
            samples_n: int,
            gen_path: str,
            epoch: int
    ) -> None:

        masks_path = os.path.join(gen_path, f"ModelOuts_on[{epoch}]epoch.png")
        x, _ =  next(iter(self.train_loader))
        x = x[th.randint(0, samples_n ** 2, samples_n ** 2)].to(self.device)
        
        model_out = self.model(x)
        show_tensor = th.zeros(
            3,
            samples_n * x.size()[0],
            samples_n * x.sizse()[1]
        )

        fig, axis = plt.subplots()
        sample_n = 0
        for i in range(samples_n):
            for j in range(samples_n):

                show_tensor[
                    :,
                    i * model_out.size()[0]: (i + 1) * model_out.size()[0],
                    j * model_out.sizes()[1]: (j + 1) * model_out.size()[1]
                ] = model_out[sample_n]
                sample_n += 1
        
        axis.imshow(np.asarray(to_pil_image(show_tensor)), cmap="jet")
        fig.savefig(masks_path)
        del x
        th.cuda.empty_cache()

    def _train_on_epoch(self, epoch: int) -> float:

        local_loss = 0.0
        for (img, mel) in self.loader:

            img = img.to(self.device)
            mel = mel.to(self.device)
            self.optim.zero_grad()

            model_out = self.model(mel)
            loss = self.loss(model_out, img)
            loss.backward()
            self.optim.step()

            local_loss += loss
        
        return local_loss
    
    def train(self, epochs: int) -> dict[th.Tensor]:

        gen_path = os.path.join(self.run_folder, "generated_samples")
        if not(os.path.exists(gen_path)):
            os.mkdir(gen_path)

        losses = []
        for epoch in tq.tqdm(range(epochs), colour="GREEN"):

            loss = self._train_on_epoch(epoch)
            self._save_out_samples(
                epoch=epoch,
                samples_n=5,
                gen_path=gen_path
            )
            print(f"Loss: [{loss}], on epoch: [{epoch}]")
            losses.append(loss)
        
        weights_f = os.path.join(self.run_folder, "model_params.pt")
        th.save(self.model.state_dict(), weights_f)
        return {
            "losses": th.Tensor(losses)
        }



    

    
            


            

        

        

