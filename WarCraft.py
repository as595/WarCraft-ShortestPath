from __future__ import print_function
from PIL import Image
import os
import os.path
import numpy as np
import sys

import torch.utils.data as data
from torchvision.datasets.utils import download_url, check_integrity


class Warcraft12x12(data.Dataset):
    """

    Inspired by `HTRU1 <https://as595.github.io/HTRU1/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``MiraBest-full.py` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    base_folder = 'warcraft_shortest_path_oneskin/12x12'
    filename = 'warcraft_maps.tar'
    tar_md5 = '2b8c21192ee3b40b85c090bccbacbf2c'
    train_list = [
                  ['train_maps.npy', 'c31760c5fbd9d034b6e21a6612895e69'],
                  ['train_vertex_weights.npy', '761cc8098730504335efdefec3b221a6'],
                  ['train_shortest_paths.npy', '1483015985796f3bf9f3e5e00cb8e855'],
                ]

    val_list  = [
                  ['val_maps.npy', 'cfe9288a537d46be48c0c287885f8405'],
                  ['val_vertex_weights.npy', '452753f9d8b31ac03bbd532953974072'],
                  ['val_shortest_paths.npy', '95e81a2a3263d6520fa5f07a8701c74a'],
                ]

    test_list = [
                  ['test_maps.npy', 'a5c50ae11e4bd01cb1a2cd76e5849b22'],
                  ['test_vertex_weights.npy', '071dbcf2aa9d98fc255abdcb780151b1'],
                  ['test_shortest_paths.npy', 'deae60b10d932e2a37271df4b44e295d'],
                ]
    
    def __init__(self, root, train=True,
                 transform=None, target_transform=None,
                 download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list + self.val_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.weights = []
        self.paths = []

        # now load the numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)

            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = np.load(f)
                else:
                    entry = np.load(f, encoding='latin1')

                if 'maps' in file_name:
                    entry = entry.transpose(0, 3, 1, 2)
                    self.data.append(entry)
                elif 'vertex_weights' in file_name:
                    self.weights.append(entry)
                elif 'shortest_paths' in file_name:
                    self.paths.append(entry)

        self.data = np.vstack(self.data).reshape(-1, 3, 96, 96)
        self.data = self.data.transpose((0, 2, 3, 1))
        
        self.weights = np.vstack(self.weights).reshape(-1, 12, 12)
        self.paths = np.vstack(self.paths).reshape(-1, 12, 12)

        self._load_meta()

    def _load_meta(self):

        metadata = {
                    "input_image_size": self.data[0].shape[1],
                    "output_features": self.weights[0].shape[0] * self.weights[0].shape[1],
                    "num_channels": self.data[0].shape[-1]
                  }
            
        self.metadata = metadata

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, weights, target = self.data[index], self.weights[index], self.paths[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, weights, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.val_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


