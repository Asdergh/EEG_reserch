import torch as th
import os

from torch.utils.data import Dataset
from torchvision.transforms import Resize



class Eeg2ImgSet(Dataset):

    def __init__(
        self,
        data_dir: str,
        img_tar_s: tuple[int],
        mels_tar_s: tuple[int],
        split: str = "train"
    ) -> None:
        
        super().__init__()
        self.img_res = Resize(size=img_tar_s)
        self.mels_res = Resize(size=mels_tar_s)
        self.images, self.mels = th.load(os.path.join(data_dir, f"{split}_data.pt"))
    
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> tuple[th.Tensor]:
        return (
            self.img_res(self.images[idx].permute(2, 0, 1)) / 256.0,
            self.mels_res(self.mels[idx].permute(2, 0, 1))
        )

