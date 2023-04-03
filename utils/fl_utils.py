import torch
from typing import List, Dict
import random
import copy
from tqdm import tqdm
import numpy as np


def fed_avg(models: Dict[str, torch.Tensor]):
    avg_weights = {}
    for k, v in models.items():
        avg_weights[k] = torch.mean(v, dim=0)
    return avg_weights


def fed_sum(models: Dict[str, torch.Tensor]):
    avg_weights = {}
    for k, v in models.items():
        avg_weights[k] = torch.sum(v, dim=0)
    return avg_weights


class LocalUpdate(object):
    def __init__(self, args, dataloader):
        self.args = args
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.ldr_train = dataloader

    def train(self, net):
        net.train()
        # train and update
        optimizer = torch.optim.Adam(net.parameters(), lr=self.args.lr,)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)

        epoch_loss = []
        pbar = tqdm(range(self.args.local_epoch))
        for _ in pbar:
            batch_loss = []
            counts = 0
            corrects = 0
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                # images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                outputs = net(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()

                counts += images.shape[0]
                corrects += (outputs.argmax(1) == labels).sum().item()

                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            pbar.set_postfix({'train loss': epoch_loss[-1], 'train acc': corrects / counts})
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0]


def package_loss(models: Dict[str, torch.Tensor],
                 lost_keys: List[str] = None,
                 ratio_lost: float = 0.1,
                 seed: int = None):
    # if keys are given
    if lost_keys is not None and lost_keys:
        new_models = copy.deepcopy(models)
        for k in lost_keys:
            assert k in models.keys()
            v = new_models[k]
            if seed is not None:
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
            prob = torch.rand(v.shape)
            lost_mask = prob < ratio_lost
            v[lost_mask] = 0.
            new_models[k] = v
        return new_models

    # else drop random weights globally
    keys = list(models.keys())
    numels = []
    flattened_weights = []
    shapes = []
    for k in keys:
        v = models[k]
        numels.append(v.numel())
        shapes.append(v.shape)
        flattened_weights.append(v.reshape(-1))

    flattened_weights = torch.cat(flattened_weights, dim=0)
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    prob = torch.rand(flattened_weights.shape)
    lost_mask = prob < ratio_lost
    flattened_weights[lost_mask] = 0.
    new_weights = torch.split(flattened_weights, numels)

    new_models = {}
    for i, k in enumerate(keys):
        new_models[k] = new_weights[i].reshape(shapes[i])

    return new_models


def fed_avg_weight_lost(models: Dict[str, torch.Tensor]):
    keys = list(models.keys())
    numels = []
    flattened_weights = []
    shapes = []
    for k in keys:
        v = models[k]
        _shape = v.shape[1:]
        numels.append(torch.prod(torch.tensor(_shape)))
        shapes.append(_shape)
        flattened_weights.append(v.reshape(v.shape[0], -1))

    flattened_weights = torch.cat(flattened_weights, dim=-1)
    true_mask = flattened_weights != 0.

    sum_weights = fed_sum(models)
    counts = true_mask.sum(0).to(torch.float)
    counts[counts == 0] += 1.e-5
    counts = torch.split(counts, numels)

    new_models = {}
    for i, k in enumerate(keys):
        new_models[k] = sum_weights[k] / counts[i].reshape(shapes[i])

    return new_models


def make_gradient_ascent(dataloader, model, iters, prob):
    def gradient_ascent(weights):
        if random.random() > prob:
            return weights, 1.
        model.load_state_dict(weights)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(dataloader):
            if i >= iters:
                break
            inputs, labels = data

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = - criterion(outputs, labels)   # flip the sign of loss
            loss.backward()
            optimizer.step()

        return model.state_dict(), 0.
    return gradient_ascent


def make_all_to_label(dataloader, model, iters=1, prob=0., label=0):
    def all_to_label(weights):
        if random.random() > prob:
            return weights, 1.
        model.load_state_dict(weights)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()

        for i, data in enumerate(dataloader):
            if i >= iters:
                break
            inputs, labels = data
            labels = (labels.new_ones(labels.shape) * label).to(torch.long)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        return model.state_dict(), 0.
    return all_to_label


def mnist_get_client(lens, num_clients, iid=True):
    assert iid, "Not implemented non-iid"
    np.random.seed(42)
    idx = np.random.permutation(lens)
    client_dict = {i: part for i, part in enumerate(np.array_split(idx, num_clients))}
    return client_dict
