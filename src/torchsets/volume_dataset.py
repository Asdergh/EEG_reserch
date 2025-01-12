import torch as th
import os
import pyvista as pv
from torch.utils.data import Dataset, DataLoader


class MRTVolumeDataset(Dataset):

    def __init__(
        self,
        source_folder: str,
        data_split: str = "T1W"
    ):

        super().__init__()
        self.data = th.load(os.path.join(source_folder, f"{data_split}.pt"))
    
    def __len__(self) -> int:
        return self.data.size()[0]
    
    def __getitem__(self, idx: int) -> th.Tensor:
        return self.data[idx].unsqueeze(dim=0)




        
