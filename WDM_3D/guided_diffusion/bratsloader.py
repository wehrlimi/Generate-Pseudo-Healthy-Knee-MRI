import os
import os.path
import sys
import nibabel
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

sys.path.append("..")
sys.path.append(".")

from dataset.lakefsloader import LakeFSLoader
from dataset.datalist import DataList

import pdb

print('bratlsoader active now')


class BRATSVolumes(torch.utils.data.Dataset):
    def __init__(
        self, directory, test_flag=True, normalize=None, mode="test", img_size=256, config=None
    ):
        """
        directory is expected to contain some folder structure:
                  if some subfolder contains only files, all of these
                  files are assumed to have a name like
                  brats_train_NNN_XXX_123_w.nii.gz
                  where XXX is one of t1, t1ce, t2, flair, seg
                  we assume these five files belong to the same image
                  seg is supposed to contain the segmentation
        """
        super().__init__()
        self.mode = mode
        self.directory = os.path.expanduser(directory)
        self.normalize = normalize or (lambda x: x)
        self.test_flag = test_flag
        self.img_size = img_size
        self.lakefs_loader = None
        self.database = []
        self.config = config

        if test_flag:
            self.seqtypes = ["diseased", "mask"]#"voided", "mask"
            mode_temp = 'sample'
        else:
            self.seqtypes = ["diseased", "mask", "healthy"]
            mode_temp = 'train'

        if config:
            datalist = DataList.from_lakefs(filepath=config["lakefs"]["input_path"], data_config=config["data"], lakefs_config=config["lakefs"], mode=mode)
            self.database = datalist.data[mode_temp] #["training"]

            #pdb.set_trace()

        else:
            self._populate_database_from_directory()

        #pdb.set_trace()


        self.seqtypes_set = set(self.seqtypes)

    def _populate_database_from_directory(self):
        for root, dirs, files in os.walk(self.directory):
            if not dirs:
                files.sort()
                datapoint = dict()
                for f in files:
                    seqtype = f.split("_")[-1].split(".")[0]
                    if seqtype in self.seqtypes:
                        datapoint[seqtype] = os.path.join(root, f)
                        #pdb.set_trace()
                        self.database.append(datapoint)


    def __getitem__(self, x):
        #pdb.set_trace()
        mode = self.mode
        filedict = self.database[x]
        out_single = []

        if self.test_flag:
            for seqtype in self.seqtypes:
                #pdb.set_trace()
                file_path = filedict[seqtype]

                try:
                    # Ensure the file is cached locally
                    if self.lakefs_loader:
                        self.lakefs_loader.check_file(file_path)
                    # Load the file
                    nib_img = np.array(nibabel.load(file_path).dataobj).astype(np.float32)
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    raise e

                img_preprocessed = torch.tensor(nib_img)
                out_single.append(img_preprocessed)

            out_single = torch.stack(out_single)
            image = out_single

            path =[filedict[seqtype] for seqtype in self.seqtypes]
            return image, path

        else:
            for seqtype in self.seqtypes:
                file_path = filedict[seqtype]
                #print(f"filedict: {filedict[seqtype]}")

                # Try except block for dataloader

                try:
                    #Ensure the file is cached locally
                    if self.lakefs_loader:
                        self.lakefs_loader.check_file(file_path)
                    # Load the file
                    nib_img = np.array(nibabel.load(filedict[seqtype]).dataobj).astype(
                        np.float32
                    )
                except Exception as e:
                    print(f"Error loading file {file_path}: {e}")
                    raise e

                nib_tensor = torch.tensor(nib_img)
                out_single.append(nib_tensor)
                #print(f"shape of the nib_tensor: {nib_tensor.shape}")
            out_single = torch.stack(out_single)
            image = out_single[:2, ...]
            label = out_single[2, ...]

            label = label.unsqueeze(0)

            return (image, label)

    def __len__(self):
        return len(self.database)
