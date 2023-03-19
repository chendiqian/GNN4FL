import json
import os
from collections import defaultdict
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset


class FEMNIST(InMemoryDataset):
    """
    This dataset is derived from the Leaf repository
    (https://github.com/TalwalkarLab/leaf) pre-processing of the Extended MNIST
    dataset, grouping examples by writer. Details about Leaf were published in
    "LEAF: A Benchmark for Federated Settings" https://arxiv.org/abs/1812.01097.
    """

    def __init__(self, root='./datasets/FEMNIST', train=True, transform=None, pre_transform=None, pre_filter=None):
        super(FEMNIST, self).__init__(root, transform, pre_transform, pre_filter)
        self.transform = transform
        self.pre_transform = pre_transform
        self.train = train
        self.data, self.labels, self.dic_users = torch.load(os.path.join(self.processed_dir,
                                                                         'train.pt' if train else 'test.pt'))

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt']

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

    def get_client_dic(self):
        return self.dic_users

    def process(self):
        train_clients, train_groups, train_data_temp, test_data_temp = read_data("./datasets/FEMNIST/train",
                                                                                 "./datasets/FEMNIST/test")
        dic_users = {}
        train_data_x = []
        train_data_y = []
        for i in range(len(train_clients)):
            dic_users[i] = set()
            l = len(train_data_x)
            cur_x = train_data_temp[train_clients[i]]['x']
            cur_y = train_data_temp[train_clients[i]]['y']
            for j in range(len(cur_x)):
                dic_users[i].add(j + l)
                data = np.array(cur_x[j]).reshape(1, 28, 28)
                data = torch.from_numpy((0.5 - data)/0.5).float()
                train_data_x.append(data)
                train_data_y.append(cur_y[j])
        data = train_data_x
        label = train_data_y

        torch.save((torch.stack(data, dim=0), torch.tensor(label), dic_users), os.path.join(self.processed_dir, 'train.pt'))

        test_data_x = []
        test_data_y = []
        for i in range(len(train_clients)):
            cur_x = test_data_temp[train_clients[i]]['x']
            cur_y = test_data_temp[train_clients[i]]['y']
            for j in range(len(cur_x)):
                data = np.array(cur_x[j]).reshape(1, 28, 28)
                data = torch.from_numpy((0.5 - data) / 0.5).float()
                test_data_x.append(data)
                test_data_y.append(cur_y[j])
        data = test_data_x
        label = test_data_y

        torch.save((torch.stack(data, dim=0), torch.tensor(label), None), os.path.join(self.processed_dir, 'test.pt'))


def read_dir(data_dir):
    clients = []
    groups = []
    data = defaultdict(lambda: None)

    files = os.listdir(data_dir)
    files = [f for f in files if f.endswith('.json')]
    for f in files:
        file_path = os.path.join(data_dir, f)
        with open(file_path, 'r') as inf:
            cdata = json.load(inf)
        clients.extend(cdata['users'])
        if 'hierarchies' in cdata:
            groups.extend(cdata['hierarchies'])
        data.update(cdata['user_data'])

    clients = list(sorted(data.keys()))
    return clients, groups, data


def read_data(train_data_dir, test_data_dir):
    train_clients, train_groups, train_data = read_dir(train_data_dir)
    test_clients, test_groups, test_data = read_dir(test_data_dir)

    assert train_clients == test_clients
    assert train_groups == test_groups

    return train_clients, train_groups, train_data, test_data


def get_femnist():
    train_dataset = FEMNIST(train=True)
    val_dataset = FEMNIST(train=False)
    return train_dataset, val_dataset
