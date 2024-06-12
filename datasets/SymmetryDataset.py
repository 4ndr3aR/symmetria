import copy
import os
from typing import Optional, Callable, Tuple, List

import lzma
from pathlib import Path
import pandas as pd

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import random_split, DataLoader

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
from .build import DATASETS


class Shrec2023Transform(ABC):

    @abstractmethod
    def transform(
            self,
            idx: int,
            points: torch.Tensor,
            symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, torch.Tensor):
        pass

    @abstractmethod
    def inverse_transform(
            self,
            idx: int,
            points: torch.Tensor,
            symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, torch.Tensor):
        pass

    def __call__(
            self,
            idx: int,
            points: torch.Tensor,
            symmetries: Optional[torch.Tensor]
    ) -> (int, torch.Tensor, torch.Tensor):
        return self.transform(idx, points, symmetries)


class ComposeTransform(Shrec2023Transform):
    def inverse_transform(self, idx: int, points: torch.Tensor, symmetries: Optional[torch.Tensor]) -> (
            int, torch.Tensor, torch.Tensor):
        for a_transform in reversed(self.transforms):
            idx, points, symmetries = a_transform.inverse_transform(idx, points, symmetries)
        return idx, points, symmetries

    def __init__(
            self,
            transforms: List[Shrec2023Transform]
    ):
        self.transforms = transforms

    def transform(self, idx: int, points: torch.Tensor, symmetries: Optional[torch.Tensor]) -> (
            int, torch.Tensor, torch.Tensor):
        for a_transform in self.transforms:
            idx, points, symmetries = a_transform.transform(idx, points, symmetries)
        return idx, points, symmetries


class UnitSphereNormalization(Shrec2023Transform):
    def __init__(self):
        self.centroid = None
        self.farthest_distance = None

    def _validate_self_attributes_are_not_none(self) -> Optional[Exception]:
        if self.centroid is None or self.farthest_distance is None:
            raise Exception(f"Transform variables where null when trying to execute a method that needs them."
                            f"Variables; Centroid: {self.centroid} | Farthest distance {self.farthest_distance}")
        return None

    def _normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        self.centroid = torch.mean(points, dim=0)
        points = points - self.centroid
        self.farthest_distance = torch.max(torch.linalg.norm(points, dim=1))
        points = points / self.farthest_distance
        return points

    def _normalize_planes(self, symmetries: torch.Tensor) -> torch.Tensor:
        self._validate_self_attributes_are_not_none()
        symmetries[:, 3:6] = (symmetries[:, 3:6] - self.centroid) / self.farthest_distance
        return symmetries

    def _inverse_normalize_points(self, points: torch.Tensor) -> torch.Tensor:
        self._validate_self_attributes_are_not_none()
        points = (points * self.farthest_distance) + self.centroid
        return points

    def _inverse_normalize_planes(self, symmetries: torch.Tensor) -> torch.Tensor:
        self._validate_self_attributes_are_not_none()
        symmetries[:, 3:6] = (symmetries[:, 3:6] * self.farthest_distance) + self.centroid
        return symmetries

    def _handle_device(self, device):
        self.centroid=self.centroid.to(device)
        self.farthest_distance=self.centroid.to(device)

    def inverse_transform(self, idx: int, points: torch.Tensor, symmetries: Optional[torch.Tensor]) \
            -> (int, torch.Tensor, torch.Tensor):
        self._validate_self_attributes_are_not_none()
        self._handle_device(points.device)
        points = self._inverse_normalize_points(points)
        if symmetries is not None:
            symmetries = self._inverse_normalize_planes(symmetries)
        return idx, points, symmetries

    def transform(self, idx: int, points: torch.Tensor, symmetries: Optional[torch.Tensor]) \
            -> (int, torch.Tensor, torch.Tensor):
        points = self._normalize_points(points)
        if symmetries is not None:
            symmetries = self._normalize_planes(symmetries)
        return idx, points, symmetries


class RandomSampler(Shrec2023Transform):
    def __init__(self, sample_size: int = 1024, keep_copy: bool = True):
        self.sample_size = sample_size
        self.keep_copy = keep_copy
        self.points_copy = None

    def transform(self, idx: int, points: torch.Tensor, symmetries: Optional[torch.Tensor]) \
            -> (int, torch.Tensor, torch.Tensor):
        if self.keep_copy:
            self.points_copy = points.clone()
        chosen_points = torch.randint(high=points.shape[0], size=(self.sample_size,))
        sample = points[chosen_points]
        return idx, sample, symmetries

    def inverse_transform(self, idx: int, points: torch.Tensor, symmetries: Optional[torch.Tensor]) -> (
            int, torch.Tensor, torch.Tensor):
        if self.keep_copy:
            return idx, self.points_copy, symmetries
        else:
            return idx, points, symmetries

if __name__ == "__main__":
        import sys 
        sys.path.insert(0, '../..')


def default_symmetry_dataset_collate_fn(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = torch.nn.utils.rnn.pad_sequence([item[2] for item in batch], batch_first=True)
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms


def default_symmetry_dataset_collate_fn_list_sym(batch):
    idxs = torch.tensor([item[0] for item in batch])
    points = torch.stack([item[1] for item in batch])
    sym_planes = [item[2] for item in batch]
    transforms = [item[3] for item in batch]
    return idxs, points, sym_planes, transforms

@DATASETS.register_module()
class SymmetryShape(Dataset):
    def __init__(self, config, debug=False):
        
        """
        Dataset used for a track of SHREC2023. It contains a set of 3D points
        and planes that represent reflective symmetries.
        :param data_source_path: Path to folder that contains the points and symmetries.
        :param transform: Transform applied to dataset item.
        """
        self.data_source_path = Path(config.DATA_PATH) / config.subset
        #self.transform = transform
        self.debug = debug
        self.npoints = config.N_POINTS
        self.subset = config.subset


        scaler = UnitSphereNormalization()
        sampler = RandomSampler(sample_size=self.npoints, keep_copy=False)
        default_transform = ComposeTransform([scaler, sampler])
        self.transform = default_transform

        if self.debug:
           print(f'Searching xz-compressed point cloud files in {self.data_source_path}...')
        self.flist  = list(self.data_source_path.rglob(f'*/*.xz'))
        self.length = len(self.flist)
        if self.debug:
            print(f'{self.data_source_path.name}: found {self.length} files:\n{self.flist[:5]}\n{self.flist[-5:]}\n')

    def fname_from_idx(self, idx: int) -> str:
        if idx < 0 or idx >= len(self.flist):
            raise IndexError(f"Invalid index: {idx}, dataset size is: {len(self.flist)}")
        fname = self.flist[idx]
        if self.debug:
           print(f'Opening file: {fname.name}')
        return fname, str(fname).replace('.xz', '-sym.txt')

    def read_points(self, idx: int) -> torch.Tensor:
        """
        Reads the points with index idx.
        :param idx: Index of points to be read. Not to be confused with the shape ID, this is now just the index in self.flist
        :return: A tensor of shape N x 3 where N is the amount of points.
        """
        fname, _ = self.fname_from_idx(idx)

        points = None
        with lzma.open(fname, 'rb') as fhandle:
            points = torch.tensor(np.loadtxt(fhandle))

        if self.debug:
           torch.set_printoptions(linewidth=200)
           torch.set_printoptions(precision=3)
           torch.set_printoptions(sci_mode=False)
           print(f'[{idx}]: {points.shape = }\n{points = }')

        return points

    def read_planes(self, idx: int) -> torch.Tensor:
        """
        Read symmetry planes from file with its first line being the number of symmetry planes
        and the rest being the symmetry planes.
        :param idx: The idx of the syms to reads
        :return: A tensor of planes represented by their normals and points. N x 6 where
        N is the amount of planes and 6 because the first 3 elements
        are the normal and the last 3 are the point.
        """
        _, sym_fname = self.fname_from_idx(idx)
        with open(sym_fname) as f:
            n_planes = int(f.readline().strip())
            #converters = {1: lambda s: [0 if s == 'plane' else 1]}
            #sym_planes = torch.tensor(np.loadtxt(f, converters=converters, usecols=range(1,7)))
            #sym_planes = torch.tensor(np.loadtxt(f, usecols=range(1,8)))
            if self.debug:
                print(f'Reading CSV dataframe with filename:\n{Path(sym_fname).name}')
            #df = pd.read_csv(f, sep=' ', header=None, usecols=range(0,8), names=['type', 'nx', 'ny', 'nz', 'cx', 'cy', 'cz', 'theta']).fillna(-1) # 'ϑ'
            try:
                if self.debug:
                    print(f'Reading CSV dataframe with theta column')
                df = pd.read_csv(f, sep=' ', header=None, names=['type', 'nx', 'ny', 'nz', 'cx', 'cy', 'cz', 'theta']).fillna(-1) # 'ϑ'
            except pd.errors.ParserError:
                if self.debug:
                    print(f'Re-reading CSV dataframe without theta column')
                # NOTE here that we read the file directly, so we must throw away the first row
                df = pd.read_csv(sym_fname, sep=' ', header=None, names=['type', 'nx', 'ny', 'nz', 'cx', 'cy', 'cz']).fillna(-1) # 'ϑ'
                df = df.iloc[1:]
            if self.debug:
                print(f'Read dataframe:\n{df}')
            df['type'] = np.where(df['type'] == 'plane', 0, 1)
            if self.debug:
                print(f'Converted dataframe:\n{df}')
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            '''
            # TODO: enable these lines throw away some information present in the new version of the dataset
            if (df['type'] == 1).any() == True:          # there is an axial symmetry
                df = df[df['type'] != 1]                 # throw away axial symmetry rows
                df = df.drop('theta', axis=1)            # throw away last column == angles
                n_planes = n_planes - 1                  # decrease the number of reported symmetries
            '''
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            ''' ------------------------------------------------------------------------------------- '''
            df = df.drop('type', axis=1)                 # throw away 1st column  == plane/axis flags
            if self.debug:
                print(f'Resized dataframe:\n{df}')

            sym_planes = torch.tensor(df.values)
            if self.debug:
                print(f'Exported dataframe to torch.tensor with shape: {sym_planes.shape}\n{sym_planes}')
        #if n_planes == 1:
        #    sym_planes = sym_planes.unsqueeze(0)

        if self.debug:
           print(f'[{idx}]: {n_planes = }\n{sym_planes}')

        return sym_planes

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int) -> (int, torch.Tensor, Optional[torch.Tensor], List[Shrec2023Transform]):
        points = self.read_points(idx)
        planes = None

        #if self.has_ground_truth:
        #    planes = self.read_planes(idx)

        if self.transform is not None:
            idx, points, planes = self.transform(idx, points, planes)

        transform_used = copy.deepcopy(self.transform)
        #return idx, points.float(), planes.float(), transform_used
        return 0, 0, points.float()


#scaler = UnitSphereNormalization()
#sampler = RandomSampler(sample_size=1024, keep_copy=False)
# default_transform = ComposeTransform([scaler, sampler])
#default_transform = scaler


if __name__ == "__main__":
    
    DATA_PATH = "/data/Software/Symmetry_v2/repository2/Symmetry_Dataset/sym-10k-xz-split-class-noparallel/train"

    scaler = UnitSphereNormalization()
    sampler = RandomSampler(keep_copy=False)
    default_transform = ComposeTransform([scaler, sampler])

    train_dataset = SymmetryDataset(Path(DATA_PATH), default_transform)

    print(f'Number of samples: {len(train_dataset)}')