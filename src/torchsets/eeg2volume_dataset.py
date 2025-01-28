import torch as th
import scipy as sci
import os
import numpy as np
import pyedflib as edf
import nibabel as nib
import time

from torchvision.transforms import Resize
from scipy.signal import spectrogram
from torch.utils.data import (
    Dataset,
    DataLoader
)



class EegBOLDFmriDataset(Dataset):

    def __init__(
        self,
        source_folder: str,
        split: str,
        mel_size: tuple[int],
        vol_size: tuple[int],
        ses_split: str = "ses-01"
    ) -> None:

        super().__init__()
        start_time = time.time()
        print(f"READING DATA!!!")
        self.root = os.path.join(source_folder, split)
        self._mel_tf_ = Resize(mel_size)
        try:
            file_object = edf.EdfReader(os.path.join(
                self.root,
                "ses-EEG",
                "eeg",
                f"{split}_ses-EEG_task-inner_eeg.bdf"
            ))
        
        except:
            file_object = edf.EdfReader(os.path.join(
                self.root,
                "ses-EEG",
                "eeg",
                f"{split}_ses-EEG_eeg.bdf"
            ))
            
        n = file_object.signals_in_file
        buffer = np.zeros((
            n, 
            file_object.getNSamples()[0]
        ))
        for i in range(n):
            buffer[i, :] = file_object.readSignal(i)
        buffer = np.max(buffer, axis=0)
        self.mel = self._mel_tf_(th.Tensor(spectrogram(buffer)[-1]).unsqueeze(dim=0)).squeeze(dim=0)

        vol_path = os.path.join(
            self.root,
            f"{ses_split}",
            "func",
            f"{split}_{ses_split}_task-inner_bold.nii.gz"
        )
        tmp = th.Tensor(nib.load(vol_path).get_fdata()).permute((-1, 0, 1, 2))
        self.buffer = th.zeros((tmp.size()[0], ) + vol_size)
        self.buffer[
            :, :tmp.size()[1], 
            :tmp.size()[2], 
            :tmp.size()[3]
        ] = tmp
        del tmp
        end_time = time.time()
        print(f"READING COMPLITED!!! with total time: {end_time - start_time}")
        print(f"VOLUMETRICS BOLD FMRI SHAPE: [{self.buffer.size()}]")
        print(f"EEG SPECTROGRAM SHAPE: [{self.mel.size()}]")
    

    def __len__(self) -> int:
        return self.buffer.size()[0]

    def __getitem__(self, idx: int) -> tuple[th.Tensor]:
        return (
            self.mel,
            self.buffer[idx]
        )


