import argparse
import os
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score

from customed_datasets.graph_datasets import HeteroGraphDataset
from models.mlp import MLP
from utils.graph_utils import model_params_collate
from utils.train_utils import mlp_validation


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for training graph dataset')
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--runs', type=int, default=5)
    parser.add_argument('--epoch', type=int, default=5)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--in_feature', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()

    dataset = HeteroGraphDataset(os.path.join(args.root, args.name))
    train_loader = DataLoader(dataset[:int(len(dataset) * 0.8)],
                              shuffle=True,
                              batch_size=args.batchsize,
                              collate_fn=model_params_collate)
    val_loader = DataLoader(dataset[int(len(dataset) * 0.8): int(len(dataset) * 0.9)],
                            shuffle=True,
                            batch_size=args.batchsize,
                            collate_fn=model_params_collate)
    test_loader = DataLoader(dataset[int(len(dataset) * 0.9):],
                             shuffle=True,
                             batch_size=args.batchsize,
                             collate_fn=model_params_collate)

    model = MLP(in_feature=args.in_feature,
                hidden=args.hidden,
                layers=args.layers,
                num_classes=1,
                dropout=args.dropout)
    criterion = torch.nn.BCEWithLogitsLoss()

    test_accs = []
    test_f1s = []
    for _ in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=1.e-3)
        pbar = tqdm(range(args.epoch))
        best_model_dict = None
        best_val_acc = 0.
        for epoch in pbar:  # loop over the dataset multiple times
            losses = 0.
            counts = 0
            labels = []
            preds = []
            model.train()
            for i, (data, label) in enumerate(train_loader):
                # get the inputs; data is a list of [inputs, labels]
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(data)
                loss = criterion(outputs, label)
                loss.backward()
                optimizer.step()

                losses += loss.item() * label.shape[0]
                counts += label.shape[0]
                pred = (outputs > 0.).detach().to(torch.float)
                labels.append(label)
                preds.append(pred)

            losses /= counts
            preds = torch.cat(preds, dim=0)
            labels = torch.cat(labels, dim=0)
            train_acc = (preds == labels).sum().item() / labels.shape[0]
            train_f1 = f1_score(labels.numpy(), preds.numpy())

            model.eval()
            preds, labels = mlp_validation(val_loader, model)
            val_acc = (preds == labels).sum().item() / labels.shape[0]
            val_f1 = f1_score(labels.numpy(), preds.numpy())

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_dict = deepcopy(model.state_dict())

            pbar.set_postfix({'train_acc': train_acc, 'val_acc': val_acc, 'train_f1': train_f1, 'val f1': val_f1})

        model.load_state_dict(best_model_dict)
        model.eval()
        preds, labels = mlp_validation(test_loader, model)
        test_acc = (preds == labels).sum().item() / labels.shape[0]
        test_f1 = f1_score(labels.numpy(), preds.numpy())
        print(f'test acc: {test_acc}, test f1: {test_f1}')
        test_accs.append(test_acc)
        test_f1s.append(test_f1)

    print(f'acc mean: {np.mean(test_accs)} ± std: {np.std(test_accs)}')
    print(f'f1 mean: {np.mean(test_f1s)} ± std: {np.std(test_f1s)}')
