"""
Collection of functions to interact with lakefs and the s3 storage.
"""

import os
import pdb

import boto3
import logging
from pathlib import Path

from typing import Tuple, List

class LakeFSLoader():

    def __init__(
            self, 
            repo_name: str,
            branch_id: str,
            local_cache_path: str,
            endpoint: str | None,
            ca_path: str | None,
            access_key: str | None,
            secret_key: str | None,
            ):

        self.repo_name = repo_name
        self.branch_id = branch_id
        if not self.branch_id.endswith('/'):
            self.branch_id += '/'
        self.local_cache_path = local_cache_path
        if not self.local_cache_path.endswith('/'):
            self.local_cache_path += '/'

        self.endpoint = endpoint
        self.ca_path = ca_path
        self.access_key = access_key
        self.secret_key = secret_key
        self._lakefs = None  # Delay the S3 client initialization


    @property
    def lakefs(self):
        if not self._lakefs:
            # Initialize the S3 client on demand
            if self.endpoint and self.ca_path and self.access_key and self.secret_key:
                self._lakefs = boto3.client(
                    's3',
                    endpoint_url=self.endpoint,
                    aws_access_key_id=self.access_key,
                    aws_secret_access_key=self.secret_key,
                    verify=self.ca_path,
                )
            else:
                self._lakefs = boto3.client('s3')
        return self._lakefs


    def check_num_missing_files(self, object_names: List):
        """
        Check the objects if they are already in the cache, returns the number which still need to be downloaded.

        Return:
            (int): Number of files that still need to be downloaded.
        """
        not_in_cache = 0
        for obj_name in object_names:
            file_path = Path(self.local_cache_path) / obj_name
            if not os.path.exists(file_path):
                not_in_cache += 1

        return not_in_cache
    

    def get_local_and_obj_names(self, name: str) -> Tuple:
        """
        Returns both the full local cache path and the object name (s3 path starting with the branch id),
        with which it can be identified within LakeFS.

        Args:
            name (string): Either the object name or local name.
        Return:
            Tuple(string, string): (local name, object name)
        """
        if name.startswith(self.branch_id):
            return str(Path(self.local_cache_path) / name), name
        else:
            return name, name.replace(self.local_cache_path, '').lstrip('/')
    

    def get_branch_dir(self):
        """
        Gets the directory to the cached files

        Return:
            (pathlib.Path): The directory path of the cache corresponding to the lakefs branch id.
        """
        return Path(self.local_cache_path) / self.branch_id
    

    def check_file(self, file_name: str):
        """
        Checks if a file exists, if not downloads it.
        Args:
            file_name (string): Either the local or s3 file name
        """
        local_name, obj_name = self.get_local_and_obj_names(file_name)
        if not os.path.exists(local_name):
            os.makedirs(Path(local_name).parent, exist_ok=True)
            logging.info(f"{obj_name} missing in the cache - downloading now from s3 storage.")
            self.lakefs.download_file(self.repo_name, obj_name, local_name)


    def check_dir(self, dir_name: str):
        """
        Checks if a local file directory exists, if not downloads it and its subfiles.
        Args:
            local_name (string): The object name (s3 path starting with the branch id).
        """
        local_dir_name, obj_dir_name = self.get_local_and_obj_names(dir_name)
        if not os.path.exists(local_dir_name):
            os.makedirs(Path(local_dir_name), exist_ok=True)
            # search for all matching objects in lakefs
            dir_objects = self.read_s3_objects(prefix=obj_dir_name.replace(self.branch_id, ''))

            # download
            for dir_object in dir_objects:
                self.check_file(dir_object)


    def read_s3_objects(self, filter: str | Tuple = '', prefix: str = ''):
        """
        Return a list of object names found within the s3 storage system.

        Args:
            filter (string or Tuple): string or Tuple of strings. 
                Will filter out any object where the name (path in the bucket) doesnt contain any of the strings.
            prefix (str): prefix to add after the branch id, before any further subdirectories. Can indicate a single subdirectory to consider.
        Return:
            (List): List of keys (the path names starting with the branch id).
        """
        # get all objects
        paginator = self.lakefs.get_paginator("list_objects_v2")
        search_prefix = self.branch_id + prefix
        pages = paginator.paginate(Bucket=self.repo_name, Prefix=search_prefix)
        obj_keys = []
        for page in pages:
            for obj in page["Contents"]:
    
                key = obj["Key"] #.replace(self.branch_id, '')
                # filter for a certain part in the key (path name, e.g. seg.nii.gz)
                if filter:
                    if isinstance(filter, str):
                        filter = [filter]
                    for tag in filter:
                        if tag in key:
                            obj_keys.append(key)
                else:
                    obj_keys.append(key)
        return obj_keys

    def check_file(self, file_name: str):
        local_name, obj_name = self.get_local_and_obj_names(file_name)
        if not os.path.exists(local_name):
            os.makedirs(Path(local_name).parent, exist_ok=True)
            logging.info(f"{obj_name} missing in the cache - downloading now from s3 storage.")
            self.lakefs.download_file(self.repo_name, obj_name, local_name)