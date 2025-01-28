import pyedflib as edf
import torch as th
import nibabel as nib
import os
import numpy as np
import tqdm

from torchvision.transforms import Resize
from scipy.signal import spectrogram



class Eeg2VolumeDataFormater:

    def __init__(
        self,
        source_folder: str,
        target_folder: str,
        target_size_mel: tuple[int],
        target_size_vol: tuple[int]
    ) -> None:
        
        idx = 0
        self.sr_root = source_folder
        self.tar_root = target_folder
        self.mel_tf = Resize(target_size_mel)
        except_files = [
            "CHANGES",
            "dataset_description.json",
            "participants.json",
            "participants.tsv",
            "README"
        ]

        print("LOADING BEGIN!!!")
        for path in tqdm.tqdm(os.listdir(self.sr_root), colour="blue"):
            
            full_path = os.path.join(self.sr_root, path)
            if not(path in except_files):

                try:
                    file_object = edf.EdfReader(os.path.join(
                        full_path,
                        "ses-EEG",
                        "eeg",
                        f"{path}_ses-EEG_task-inner_eeg.bdf"
                    ))
                
                except:
                    file_object = edf.EdfReader(os.path.join(
                        full_path,
                        "ses-EEG",
                        "eeg",
                        f"{path}_ses-EEG_eeg.bdf"
                    ))
                    
                n = file_object.signals_in_file
                buffer = np.zeros((
                    n, 
                    file_object.getNSamples()[0]
                ))
                for i in range(n):
                    buffer[i, :] = file_object.readSignal(i)
                buffer = np.max(buffer, axis=0)
                mel = self.mel_tf(th.Tensor(spectrogram(buffer)[-1]).unsqueeze(dim=0)).squeeze(dim=0)

                for sess in os.listdir(full_path):

                    sess_path = os.path.join(full_path, sess)
                    if sess != "ses-EEG":
                        vol_path = os.path.join(
                            sess_path,
                            "func",
                            f"{path}_{sess}_task-inner_bold.nii.gz"
                        )
                        tmp = th.Tensor(nib.load(vol_path).get_fdata()).permute((-1, 0, 1, 2))
                        buffer = th.zeros((tmp.size()[0], ) + target_size_vol)
                        buffer[
                            :, :tmp.size()[1], 
                            :tmp.size()[2], 
                            :tmp.size()[3]
                        ] = tmp
                        del tmp

                        print("SOTARING_DATA !!!")
                        for vol_sample in tqdm.tqdm(buffer, colour="red"):
                            th.save(
                                (mel, vol_sample),
                                os.path.join(self.tar_root, f"Sample{idx}.pt")
                            )
                            idx += 1
                    


if __name__ == "__main__":

    formater = Eeg2VolumeDataFormater(
        source_folder="C:\\Users\\1\\Desktop\\fmri_eeg_dataset",
        target_folder="C:\\Users\\1\\Desktop\\target_root",
        target_size_mel=(40, 200),
        target_size_vol=(128, 128, 128)
    )
        
                        
                                            

                
        