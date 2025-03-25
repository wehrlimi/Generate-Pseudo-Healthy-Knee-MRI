"""
Datalist class to create a json datalist from the brats data which is also suitable for the auto3dseg
"""
import os
from pathlib import Path
from typing import Dict
import random

import json
import logging

from dataset.lakefsloader import LakeFSLoader
import pdb

class DataList():

    def __init__(
            self,
            data: Dict,
    ) -> None:
        """
        Initialize a json compatible datalist.
        """
        self.data = data

    @classmethod
    def from_json(
        cls,
        filepath: Path,
        lakefs_config: dict = {},
        ) -> None:
        """
        Create datalist from a json file. 
        """
        filepath = Path(filepath)
        with open(filepath) as json_file:
            data = json.load(json_file)

        # configure lakefs
        if lakefs_config:
            lakefs_loader = LakeFSLoader(
                local_cache_path=lakefs_config["cache_path"],
                repo_name=lakefs_config["data_repository"],
                branch_id=lakefs_config["branch"],
                ca_path=lakefs_config["ca_path"],
                endpoint=lakefs_config["s3_endpoint"],
                secret_key=lakefs_config["secret_key"],
                access_key=lakefs_config["access_key"],
            )

        # make sure that the file in the data exists
        logging.info(f"Checking that every entry in the data is valid.")
        for _, set_value in data.items():
            # each patient
            for patient in set_value:
                # split the keys
                for patient_key, patient_value in patient.items():
                    # only consider image and label
                    if patient_key == "image" or patient_key == "label":
                        # handle both lists (multichannel image) and single images
                        if isinstance(patient_value, str):
                            patient_value = [patient_value]
                        for pv in patient_value:
                            if lakefs_config:
                                _, extension = os.path.splitext(pv)
                                if extension:
                                    lakefs_loader.check_file(pv)
                                else:
                                    lakefs_loader.check_dir(pv)
                            elif not os.path.exists(pv):
                                logging.warning(f"{pv} was not found, make sure to use a valid datalist.json")
                                raise FileNotFoundError(f"{pv} could not be found, make sure that it exists or use the s3 storage")

        return cls(data=data)

    @classmethod
    def from_file(
        cls,
        datapath: Path,
        config: Dict,
        ):
        """
        Create datalist from a single datafile.
        """
        datapath = Path(datapath)
        img_tag = config["img_tag"]

        # get the file
        case = ''
        for file in datapath.iterdir():
            if file.is_file() and img_tag in file.name.lower():
                case = file

        # create the datalist
        if case == '':
            data = None
        else:
            data = {"testing": [], "training": []}
            data["testing"].append({"image": str(case)})

        return cls(data=data)

    @classmethod
    def from_directory(
            cls,
            datapath: Path,
            config: Dict,
            include_root: bool = True,
            shuffle: bool = True,
            mode = None,
    ):

        directory = Path(datapath)

        if mode == "sample":
            seqtypes = ["diseased", "mask"] #"voided", "mask"
            data = {"sample": []}
        elif mode == "train":
            seqtypes = ["diseased", "mask", "healthy"]
            data = {"train": []}
        else:
            raise KeyError("Invalid data_mode: must be 'sample' or 'train'")

        for root, dirs, files in os.walk(directory):
            if not dirs:
                files.sort()
                datapoint = dict()
                for f in files:
                    seqtype = f.split("_")[-1].split(".")[0]
                    if seqtype in seqtypes:
                        datapoint[seqtype] = os.path.join(root, f)
                        data[mode].append(datapoint)

       #pdb.set_trace()
        return cls(data=data)



    @classmethod
    def from_directory_old(
            cls,
            datapath: Path,
            config: Dict,
            include_root: bool = True,
            shuffle: bool = True,
            ):
        """
        Create datalist from a directory. It is expected that the directory has a subdirectory for each patient.
        
        Args:
            datapath (str): The local path to the data
            config (dict): According to config.yaml - data
            include_root (bool): Whether or not to include the root in the datalist
            shuffle: (bool): Shuffle the dataset
        """
        datapath = Path(datapath)
        img_tag = config["img_tag"]
        lbl_tag = config["lbl_tag"]
        random.seed(42)

        # handle the train-test set splitting
        train_patient_list = []
        test_patient_list = []
        if config["train_test_already_split"]:
            # get the corresponding subdirectories
            for sub_dir in datapath.iterdir():
                if sub_dir.is_dir() and 'test' in sub_dir.name.lower():
                    test_patient_list = [f for f in sub_dir.iterdir()]
                    if shuffle: random.shuffle(test_patient_list)
                    test_patient_list = test_patient_list[:int(config["use_only_fraction_of_data"]*len(test_patient_list))]
                elif sub_dir.is_dir() and 'train' in sub_dir.name.lower():
                    train_patient_list = [f for f in sub_dir.iterdir()]
                    if shuffle: random.shuffle(train_patient_list)
                    train_patient_list = train_patient_list[:int(config["use_only_fraction_of_data"]*len(train_patient_list))]

        else:
            # split the patient subdirectories into test and train
            patient_list = [f for f in datapath.iterdir()]
            if shuffle: random.shuffle(patient_list)
            patient_list = patient_list[:int(config["use_only_fraction_of_data"]*len(patient_list))]
            train_patient_list = patient_list[:int((1-config["test_split"])*len(patient_list))]
            test_patient_list = patient_list[int((1-config["test_split"])*len(patient_list)):]

        # create data skeleton
        data = {"testing": [], "training": []}
        #pdb.set_trace()
        # populate testing data
        for patient in test_patient_list:

            # get the img and label
            patient_test_img = ''
            patient_test_lbl = ''
            for file in patient.iterdir():
                if img_tag in file.name.lower():
                    patient_test_img = file
                elif lbl_tag in file.name.lower():
                    patient_test_lbl = file
            
            # skip the subfolders if the data isnt available (can lead to slight deviations in the demanded test/train split)
            if patient_test_img == '':
                continue
            
            # fill the data
            if include_root:
                if patient_test_lbl == '':
                    data["testing"].append(
                        {"image": str(patient_test_img)}
                    )
                else:
                    data["testing"].append(
                        {"image": str(patient_test_img), "label": str(patient_test_lbl)}
                    )
            else:
                # check if the dataset was already split
                if config["train_test_already_split"]:
                    if patient_test_lbl == '':
                        data["testing"].append(
                            {"image": (patient_test_img.parts[-3] + '/' + patient_test_img.parts[-2] + '/' + patient_test_img.parts[-1])}
                        )
                    else:
                        data["testing"].append(
                            {"image": (patient_test_img.parts[-3] + '/' + patient_test_img.parts[-2] + '/' + patient_test_img.parts[-1]),
                             "label": (patient_test_lbl.parts[-3] + '/' + patient_test_lbl.parts[-2] + '/' + patient_test_lbl.parts[-1])}
                        )
                else:
                    if patient_test_lbl == '':
                        data["testing"].append(
                            {"image": (patient_test_img.parent.name + '/' + patient_test_img.name)}
                        )
                    else:
                        data["testing"].append(
                            {"image": (patient_test_img.parent.name + '/' + patient_test_img.name),
                             "label": (patient_test_lbl.parent.name + '/' + patient_test_lbl.name)}
                        )

        #pdb.set_trace()
        # populate training data
        for patient in train_patient_list:

            # get the img and label
            patient_train_img = ''
            patient_train_lbl = ''
            for file in patient.iterdir():
                if img_tag in file.name.lower():
                    patient_train_img = file
                elif lbl_tag in file.name.lower():
                    patient_train_lbl = file
            # skip the subfolders if the data isnt available (can lead to slight deviations in the demanded test/train split)
            if patient_train_img == '' or patient_train_lbl == '':
                continue

            # fill the data
            if include_root:
                data["training"].append(
                    {"image": str(patient_train_img), "label": str(patient_train_lbl), "fold": 0}
                )
            else:
                # check if the dataset was already split
                if config["train_test_already_split"]:
                    data["training"].append(
                        {"image": (patient_train_img.parts[-3] + '/' + patient_train_img.parts[-2] + '/' + patient_train_img.parts[-1]),
                         "label": (patient_train_lbl.parts[-3] + '/' + patient_train_lbl.parts[-2] + '/' + patient_train_lbl.parts[-1])}
                    )
                else:
                    data["training"].append(
                        {"image": (patient_train_img.parent.name + '/' + patient_train_img.name),
                         "label": (patient_train_lbl.parent.name + '/' + patient_train_lbl.name)}
                    )

            # split the folds
            num_folds = config["folds"]
            if num_folds > 1:
                fold_size = len(data["training"]) // num_folds
                for fold_number in range(num_folds):
                    for i in range(fold_size):
                        data["training"][fold_number * fold_size + i]["fold"] = fold_number

        #pdb.set_trace()
        return cls(data=data)
    
    @classmethod
    def from_lakefs(
        cls,
        data_config,
        lakefs_config,
        filepath: str = '',
        include_root: bool = True,
        shuffle: bool = True,
        mode = None,
    ):
        """
        Create the datalist from the s3 storage
        
        Args:
            data_config (dict): According to config.yaml - data
            lakefs_config (dict): According to config.yaml
            filepath (str): The relative path, starting from within the lakefs branch
            include_root (bool): Whether or not to include the root in the datalist
            shuffle: (bool): Shuffle the dataset
        """
        # parse lakefs config
        lakefs_loader = LakeFSLoader(
                local_cache_path=lakefs_config["cache_path"],
                repo_name=lakefs_config["data_repository"],
                branch_id=lakefs_config["branch"],
                ca_path=lakefs_config["ca_path"],
                endpoint=lakefs_config["s3_endpoint"],
                secret_key=lakefs_config["secret_key"],
                access_key=lakefs_config["access_key"],
            )
        
        # iterate through the s3 objects, only considering the ones of interest
        tags = (data_config["img_tag"], data_config["lbl_tag"])
        objects = lakefs_loader.read_s3_objects(filter=tags)
        logging.info(f"Creating a datalist from {len(objects)} files.")

        # check if they are available in the cache, otherwise download
        logging.info(f"Checking cache... Need to download {lakefs_loader.check_num_missing_files(objects)} files.")
        for obj in objects:
            lakefs_loader.check_file(obj)
        logging.info(f"Finished loading the cache.")

        # create the datalist
        datapath = lakefs_loader.get_branch_dir() / filepath
        #pdb.set_trace()
        return cls.from_directory(
            datapath=datapath,
            config=data_config,
            include_root=include_root,
            shuffle=shuffle,
            mode=mode,
            )

    def save_datalist_to_json(self, path: Path, remember_path: bool = True) -> None:
        """
        Save the datalist to file.
        """
        # save the filepath
        if remember_path:
            self.filepath = path

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)
        logging.info(f"Datalist saved to {path}")
        
