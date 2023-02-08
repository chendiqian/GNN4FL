import copy
import os
from typing import Optional, Callable, List, Union, Tuple, Dict

import torch
from torch_geometric.data import Data
from torch_geometric.data.dataset import Dataset
from torch_geometric.data.separate import separate
from tqdm import tqdm

from graph_utils import separate, my_hetero_collate


def linear_mapping(model: Dict[str, torch.Tensor],
                   target_dim: int,
                   norm: bool = True,
                   fix_seed: int = 2023):
    torch.manual_seed(fix_seed)
    torch.cuda.manual_seed(fix_seed)
    target_vec = 0.
    for k, weight in model.items():
        if weight.dim() == 0:
            weight = torch.tensor([weight])

        flat_weight = weight.reshape(weight.shape[0], -1)
        map_mat = torch.rand(flat_weight.shape[-1], target_dim)
        target_vec += flat_weight @ map_mat

    if norm:
        target_vec = (target_vec - target_vec.mean()) / target_vec.std()

    return target_vec


class HeteroGraphDataset(Dataset):
    def __init__(self, root: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, self.processed_file_names)
        self.data = torch.load(path)
        # self.data_list = [None] * len(self)

    @property
    def raw_file_names(self) -> Union[str, List[str], Tuple]:
        files = os.listdir(self.raw_dir)
        files = [f for f in files if f.endswith('.pt')]
        return files

    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return 'graphs.pt'

    def len(self) -> int:
        if isinstance(self.data, list):
            return len(self.data)
        else:
            return len(self.data.ptr_feature) - 1

    # def get(self, idx: int) -> Data:
    #     if self.data_list[idx] is not None:
    #         return self.data_list[idx]
    #
    #     data = separate(self.data, idx)
    #     new_data = copy.deepcopy(data)
    #     self.data_list[idx] = new_data
    #     return new_data

    def get(self, idx: int) -> Data:
        return self.data[idx]

    def __getitem__(self, idx):
        return self.get(idx)

    def process(self):
        graphs = []
        for filename in tqdm(self.raw_file_names):
            g = torch.load(os.path.join(self.raw_dir, filename))
            graphs.append(g)

        # torch.save(my_hetero_collate(graphs),
        #            os.path.join(self.processed_dir, self.processed_file_names))
        torch.save(graphs, os.path.join(self.processed_dir, self.processed_file_names))
