import os
import random
import copy

import torch
from torch_geometric.data import Dataset
from tqdm import tqdm


class DefaultTransform:
    def __call__(self, weights):
        return weights, 1.


class AdditiveNoise:
    def __init__(self, perturb_prob=0.5, std=1.0):
        assert 0 <= perturb_prob <= 1
        self.p = perturb_prob
        self.std = std

    def __call__(self, weights):
        if random.random() < self.p:
            for k in weights.keys():
                v = weights[k]
                noise = torch.randn(v.shape) * self.std
                weights[k] = v + noise
            label = 0.
        else:
            label = 1.
        return weights, label


class SignFlip:
    def __init__(self, perturb_prob=0.5):
        assert 0 <= perturb_prob <= 1
        self.p = perturb_prob

    def __call__(self, weights):
        if random.random() < self.p:
            for k in weights.keys():
                v = weights[k]
                weights[k] = -v
            label = 0.
        else:
            label = 1.
        return weights, label


class PretrainedWeights(Dataset):

    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=DefaultTransform(),
                 pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'models.pt')
        self.data = torch.load(path)

    @property
    def raw_file_names(self):
        models = os.listdir(self.raw_dir)
        models = [m for m in models if m.endswith('.pt')]
        return models

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed')

    @property
    def processed_file_names(self):
        return 'models.pt'

    def len(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        weights, label = self.data[idx]
        return_weights = copy.deepcopy(weights)
        if self.transform is not None:
            return_weights, label = self.transform(return_weights)
        return return_weights, label

    def process(self):
        # idx = {'train': range(int(len(self.raw_file_names) * 0.8)),
        #        'val': range(int(len(self.raw_file_names) * 0.8),
        #                     int(len(self.raw_file_names) * 0.9)),
        #        'test': range(int(len(self.raw_file_names) * 0.9),
        #                      int(len(self.raw_file_names)))}

        models = []
        for filename in tqdm(self.raw_file_names):
            model = os.path.join(self.raw_dir, filename)
            weights = torch.load(model, map_location='cpu')
            models.append(self.pre_transform(weights))

        torch.save(models, os.path.join(self.processed_dir, f'models.pt'))
