import torch as th
import os
import nibabel as nib
import tqdm
from torch.nn import Upsample



class VolData:


    def __init__(
        self,
        source_folder: str,
        target_folder: str,
        target_size: tuple[int],
    ) -> None:
        
        self.source_folder = source_folder
        self.target_folder = target_folder
        self.tar_size = target_size

        self.res = Upsample(self.tar_size)
        self._data_to_pt_()
        

    def _data_to_pt_(self) -> None:

        collection_t1, collection_flair = [], []
        if not(os.path.exists(self.target_folder)):
            os.mkdir(self.target_folder)
        
        for path in tqdm.tqdm(os.listdir(self.source_folder)):
            
            if path in ["CHANGES", "dataset_description.json", "README"]:
                continue

            str_id = path[path.find("-") + 1:]
            t1_sample_path = os.path.join(
                self.source_folder,
                path,
                "anat",
                f"sub-{str_id}_T1w.nii.gz"
            )
            flair_sample_path = os.path.join(
                self.source_folder,
                path,
                "anat",
                f"sub-{str_id}_FLAIR.nii.gz"
            )
            
            t1_sample = self.res(th.Tensor(nib.load(t1_sample_path).get_fdata()).unsqueeze(dim=0).unsqueeze(dim=1)).squeeze(dim=0)
            try:
                flair_sample = self.res(th.Tensor(nib.load(t1_sample_path).get_fdata()).unsqueeze(dim=0).unsqueeze(dim=1)).squeeze(dim=0)
            
            except BaseException:
                flair_sample = th.zeros(self.tar_size)
            
            collection_t1.append(t1_sample)
            collection_flair.append(flair_sample)
        
        collection_t1 = th.cat(collection_t1, dim=0)
        collection_flair = th.cat(collection_flair, dim=0)
        print(collection_flair.size(), collection_t1.size())
        paths = {
            "T1W.pt": collection_t1,
            "FLAIR.pt": collection_flair
        }
        
        for path in paths.keys():

            print(os.path.join(self.target_folder, path), paths[path].size())
            th.save(paths[path], os.path.join(self.target_folder, path))


        
        
        
