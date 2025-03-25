import torch
import nibabel as nib

from pathlib import Path

from torch import Tensor
from typing import List, Any, Dict
from torch.utils.data import Dataset
from torchvision import transforms

from dataset.lakefsloader import LakeFSLoader


class LakeFSCacheError(Exception):
    """
    Raise this error in the special case when you are able to download files into the cache from lakefs, but the file disappears.
    """


class FishTankDataset(Dataset):
    """
    Custom Fish Tank dataset.
    Gets the datapaths from a list, and gets the data from the S3 storage if it isnt found.

    Args:
        data (List(Dict)): A list of Dictionaries, each for one datasample
        lakefs_loader (LakeFSLoader): The lakefs loader to load data from the S3 storage
        transform (torchvision.transforms): The transforms to use on the images
        max_attempts (int): how often to try to reload the data from S3
    """
    def __init__(
            self, 
            data: List,
            lakefs_loader: LakeFSLoader,
            transforms: transforms = None,
            max_attempts: int = 10
        ) -> None:

        self.data = data
        self.lakefs_loader = lakefs_loader
        self.max_attempts = max_attempts
        self.transforms = transforms

    
    def preprocess(self, image: Tensor, label: Tensor):
        """
        Normalize the image and extract the wanted mask values from the label.
        """
        # Normalize the image
        normalize = transforms.Normalize(
            mean=torch.mean(image, (1,2)), 
            std=torch.std(image, (1,2)), 
        )
        image = normalize(image)

        # Extract the fish (2) and sharks (3) labels
        label = torch.logical_or(label == 2, label == 3).float()

        return image, label
    

    def load_data(self, sample: Dict):
        """
        Load the data and ensure that it exists in the cache. 
        An example how to redownload a datafile from S3 storage if it gets deleted from the cache during training.
        """
        # get the paths
        image_path = sample["image"]
        label_path = sample["label"]

        # wrap your own function within a try except block to catch if the cache gets cleared during training
        #-------------------------------------------------------------------
        for _ in range(self.max_attempts):
            try:
                ###### your own function which loads the file
                ###
                image = nib.load(image_path)
                label = nib.load(label_path)
                ###
                ###### 
                return image, label
            
            except FileNotFoundError:
                # redownload if it is not downloaded
                self.lakefs_loader.check_file(image_path)
                self.lakefs_loader.check_file(label_path)

        # raise custom error in case the cache is behaving off
        raise LakeFSCacheError(f"Tried {self.max_attempts} times to access the files from {image_path} and {label_path}. \
                               Downloading the files from Lakefs has worked, but they seem to be deleted before reading them. \
                               Check the cache hanlding.")
        #-------------------------------------------------------------------


    def __len__(self):
        # get the number of samples within this dataset
        return len(self.data)
    

    def __getitem__(self, index) -> Any:
        # get the datasample according to the index
        sample = self.data[index]
        root_path = str(Path(sample["image"]).parent)

        # get the image and label
        image, label = self.load_data(sample)

        # convert to tensor
        image = torch.from_numpy(image.get_fdata()).float()
        label = torch.from_numpy(label.get_fdata()).float()

        # pre processing
        image, label = self.preprocess(image, label)
        sample = {"image": image, "label": label, "root_path": root_path}

        # apply transforms
        if self.transforms:
            sample = self.transforms(sample)

        return sample