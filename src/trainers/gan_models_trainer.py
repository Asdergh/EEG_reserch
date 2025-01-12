import torch as th
import os
import matplotlib.pyplot as plt
import numpy as np
import tqdm as tq


from torch.nn import (
    Module,
    BCELoss
)
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms.functional import to_pil_image
from torchvision.transforms import (
    ToTensor,
    Resize,
    Compose
)



class GanModelsTrainer:

    def __init__(
        self,
        run_folder: str,
        noise_dim: int,
        img_size: tuple[int],
        models: dict[Module],
        optims: dict[Module],
        shuffle_learning: bool = True,
        batch_size: int = 32,
        device: str = None, 
        dataset_root: str = None,
    ) -> None:
        
        self.root = run_folder
        if not(os.path.exists(self.root)):
            os.mkdir(self.root)

        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.device = device
        
        self._models_ = models
        self._optims_ = optims
        self._loss_ = BCELoss()

        self.transforms = Compose([
            ToTensor(),
            Resize(img_size)
        ])

        download_mode = False
        if dataset_root is None:
            dataset_root = "dataset_root"
            os.mkdir(dataset_root)
            download_mode = True

        dataset = CIFAR10(
            root=dataset_root,
            download=download_mode,
            train=True
        )
        self._loader_ = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle_learning
        )
    

    def _gen_loss_(self) -> th.tensor:
        
        self._optims_["generator"].zero_grad()
        noise = th.normal(0.12, 1.12, (self.batch_size, self.noise_dim))
        labels = th.ones((self.batch_size, 1))
        
        gan_out = self._models_["generator"](noise)
        dis_out = self._models_["discriminator"](gan_out)

        loss = self._loss_(dis_out, labels)
        loss.backward()
        self._optims_["generator"].step()

        return loss
    
    def _dis_loss_(self, real_samples: th.Tensor) -> th.tensor:

        self._optims_["discriminator"].zero_grad()
        noise = th.normal(0.12, 1.12, (self.batch_size, self.noise_dim))
        real_labels = th.ones((self.batch_size, 1))
        fake_labels = th.ones((self.batch_size, 1))

        gan_out = self._models_["generator"](noise)
        dis_fake = self._models_["discriminator"](gan_out)
        dis_real = self._models_["discriminator"](real_samples)

        loss_fake = self._loss_(dis_fake, fake_labels)
        loss_real = self._loss_(dis_real, real_labels)
        loss_res = loss_fake + loss_real

        loss_fake.backward()
        loss_real.backward()
        self._optims_["discriminator"].step()

        return loss_res

    def _save_out_samples(
            self,
            samples_n: int,
            gen_path: str,
            epoch: int
    ) -> None:


        epoch_f = os.path.join(gen_path, f"epoch{epoch}.png")
        x, _ =  next(iter(self.train_loader))
        x = x[th.randint(0, samples_n ** 2, samples_n ** 2)]
        if self.device is not None:
            x = x.to(self.device)
        
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
        fig.savefig(epoch_f)
        if self.device is not None:
            del x
            th.cuda.empty_cache()


    def _train_on_epoch_(self) -> th.tensor:

        gen_local_loss = 0
        dis_local_loss = 0
        for (img, _) in self._loader_:

            if self.device is not None:
                img = img.to(self.device)
            
            dis_loss = self._dis_loss_(real_samples=img)
            gen_loss = self._gen_loss_()

            gen_local_loss += gen_loss.item()
            dis_local_loss += dis_loss.item()
        
        return (gen_local_loss, dis_local_loss)

    def train(self, epochs: int) -> dict[th.Tensor]:

        gen_path = os.path.join(self.root, "gen_path")
        params = os.path.join(self.root, "weights")
        paths = [gen_path, params]
        
        for path in paths:
            if not(os.path.exists(path)):
                os.mkdir(path)
            

        dis_loss = []
        gen_loss = []
        for epoch in tq.tqdm(range(epochs)):
            
            dls, gls = self._train_on_epoch_()
            self._save_out_samples(
                gen_path=gen_path, 
                samples_n=5, 
                epoch=epoch
            )
            dis_loss.append(dls)
            gen_loss.append(gls)
        
        for m_key in self._models_.keys():
            
            params = self._models_[m_key].state_dict()
            th.load(params, os.path.join(params, f"{m_key}.pt"))
            
        return {
            "discriminator_loss": th.Tensor(dis_loss),
            "generator_loss": th.Tensor(gen_loss)
        }

        
    