import argparse
from copy import deepcopy
import os
import random

import numpy as np
import torch
from tqdm import tqdm

from customed_datasets.get_femnist import get_femnist
from models.femnist_cnn import FEMNIST_CNN
from utils.train_utils import mnist_validation

# some training dynamics are taken from https://github.com/wenzhu23333/Federated-Learning
def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epoch', type=int, default=50, help='train CNN on MNIST dataset')
    parser.add_argument('--batchsize', type=int, default=64, help='batch size of MNIST for training CNN')
    parser.add_argument('--modelsPerseed', type=int, default=100, help='number of models per seed')
    parser.add_argument('--seed', type=int, default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    train_dataset, val_dataset = get_femnist()
    cnn = FEMNIST_CNN()
    criterion = torch.nn.CrossEntropyLoss()

    root = './datasets/FEMNIST_CNN'
    if not os.path.isdir(root):
        os.mkdir(root)
    root = os.path.join(root, 'raw')
    if not os.path.isdir(root):
        os.mkdir(root)

    dict_user = train_dataset.get_client_dic()
    # create ONE graph, models share the same seed
    if args.seed is not None:
        sl = [args.seed]
    else:
        sl = range(1, 20)
    for seed in sl:
        if not os.path.isdir(os.path.join(root, str(seed))):
            os.mkdir(os.path.join(root, str(seed)))
        for m in range(args.modelsPerseed):
            print(f'seed: {seed}, {m}th model')
            user_id = random.choice(list(dict_user.keys()))
            train_subset = torch.utils.data.Subset(train_dataset, np.array(list(dict_user[user_id])))
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=args.batchsize,
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_size=args.batchsize,
                                                     shuffle=True)

            cnn.reset_parameters(seed)
            cnn.train()
            optimizer = torch.optim.SGD(cnn.parameters(), lr=args.lr, momentum=0.5)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.995)
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
                    scheduler.step()

                    counts += inputs.shape[0]
                    corrects += (outputs.argmax(1) == labels).sum().item()

                cnn.eval()
                val_acc = mnist_validation(val_loader, cnn)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = deepcopy(cnn.state_dict())
                pbar.set_postfix({'train acc': corrects / counts, 'val acc': val_acc})

            torch.save(best_model, os.path.join(root, str(seed), f'model{m}.pt'))
