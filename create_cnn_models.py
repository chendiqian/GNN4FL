import argparse
from copy import deepcopy
import os

import numpy as np
import torch
from tqdm import tqdm

from customed_datasets.get_mnist import get_mnist
from models.mnist_cnn import MNIST_CNN
from utils.train_utils import mnist_validation


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    parser.add_argument('--digitsPermodel', type=int, default=10, help='number of MNIST digits per CNN')
    parser.add_argument('--subset', type=float, default=0.2, help='subset ratio of MNIST for faster taining')
    parser.add_argument('--epoch', type=int, default=10, help='train CNN on MNIST dataset')
    parser.add_argument('--batchsize', type=int, default=64, help='batch size of MNIST for training CNN')
    parser.add_argument('--modelsPerseed', type=int, default=100, help='number of models per seed')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    train_dataset, train_labels, val_dataset, val_labels = get_mnist()
    num_classes = max(train_labels) + 1
    cnn = MNIST_CNN()
    criterion = torch.nn.CrossEntropyLoss()

    root = f'./datasets/CNN_digits{args.digitsPermodel}'
    if not os.path.isdir(root):
        os.mkdir(root)
    root = os.path.join(root, 'raw')
    if not os.path.isdir(root):
        os.mkdir(root)

        # create ONE graph, models share the same seed
    if args.seed is not None:
        sl = [args.seed]
    else:
        sl = range(1, 20)
    for seed in sl:
        for m in range(args.modelsPerseed):
            print(f'seed: {seed}, {m}th model')
            train_digits = np.random.permutation(num_classes)[:args.digitsPermodel]
            train_idx = np.in1d(train_labels, train_digits).nonzero()[0]
            train_idx = np.random.permutation(train_idx)[:int(len(train_idx) * args.subset)]
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=args.batchsize,
                                                       shuffle=True)

            val_idx = np.in1d(val_labels, train_digits).nonzero()[0]
            val_idx = np.random.permutation(val_idx)[:int(len(val_idx) * args.subset)]
            val_subset = torch.utils.data.Subset(val_dataset, val_idx)
            val_loader = torch.utils.data.DataLoader(val_subset,
                                                     batch_size=args.batchsize,
                                                     shuffle=True)

            cnn.reset_parameters(seed)
            cnn.train()
            optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
            best_val_acc = 0.

            pbar = tqdm(range(args.epoch))
            pbar.set_description(f'{m} th model')
            best_model = None
            for epoch in pbar:  # loop over the dataset multiple times
                counts = 0
                corrects = 0
                for i, data in enumerate(train_loader):
                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward + backward + optimize
                    outputs = cnn(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    counts += inputs.shape[0]
                    corrects += (outputs.argmax(1) == labels).sum().item()

                cnn.eval()
                val_acc = mnist_validation(val_loader, cnn)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = deepcopy(cnn.state_dict())
                pbar.set_postfix({'train acc': corrects / counts, 'val acc': val_acc})

            torch.save(best_model, os.path.join(root, f'model{m}.pt'))
