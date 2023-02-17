import argparse
from copy import deepcopy
import os
import yaml

import numpy as np
import torch
from tqdm import tqdm

from customed_datasets.get_mnist import get_mnist
from models.mnist_cnn import MNIST_CNN
from utils.train_utils import mnist_validation
from customed_datasets.weight_datasets import DefaultTransform, AdditiveNoise, SignFlip
from utils.fl_utils import fed_avg

from customed_datasets.graph_datasets import linear_mapping
from torch_geometric.data import HeteroData


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    parser.add_argument('--digitsPermodel', type=int, default=10, help='number of MNIST digits per CNN')
    parser.add_argument('--subset', type=float, default=0.2, help='subset ratio of MNIST for faster taining')
    parser.add_argument('--epoch_cnn', type=int, default=5, help='train CNN on MNIST dataset')
    parser.add_argument('--num_graphs', type=int, default=1, help='how many graphs you want in a graph dataset')
    parser.add_argument('--modelsPergraph', type=int, default=3, help='number of param nodes per graph')
    parser.add_argument('--aggrPergraph', type=int, default=2, help='number of aggr nodes per graph')
    parser.add_argument('--modeslPeraggr', type=int, default=2, help='number of param nodes per aggr')
    parser.add_argument('--exclusive', default=False, action='store_true',
                        help='whether one model is used ONCE for aggregation')
    parser.add_argument('--batchsize_cnn', type=int, default=64, help='batch size of MNIST for training CNN')
    parser.add_argument('--model_perturb', type=str, default=None, help='how to perturb data')
    parser.add_argument('--perturb_rate', type=float, default=0.3, help='ratio of models perturbed')
    parser.add_argument('--noise_scale', type=float, default=0.3, help='Gauss(mean=0, std=params.std() * noise_scale)')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    train_dataset, train_labels, val_dataset, val_labels = get_mnist(subset=5000)
    valset_full_digit = torch.utils.data.Subset(val_dataset, np.random.randint(0, int(len(val_dataset) * args.subset), 2000))
    val_loader_full = torch.utils.data.DataLoader(valset_full_digit, batch_size=args.batchsize_cnn, shuffle=True)
    num_classes = max(train_labels) + 1
    cnn = MNIST_CNN()
    criterion = torch.nn.CrossEntropyLoss()

    root = f'./datasets/GraphDatasetsMPG{args.modelsPergraph}_APG{args.aggrPergraph}_MPA{args.modeslPeraggr}'
    if args.exclusive:
        root += 'Exclusive'
    if not os.path.isdir(root):
        os.mkdir(root)

    with open(os.path.join(root, 'data.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

    root = os.path.join(root, 'raw')
    if not os.path.isdir(root):
        os.mkdir(root)

    for idx_graph in range(args.num_graphs):
        # create ONE graph, models share the same seed
        seed = np.random.randint(0, 10000, (1,)).item()
        best_models = []
        for m in range(args.modelsPergraph):
            train_digits = np.random.permutation(num_classes)[:args.digitsPermodel]
            train_idx = np.in1d(train_labels, train_digits).nonzero()[0]
            train_idx = np.random.permutation(train_idx)[:int(len(train_idx) * args.subset)]
            train_subset = torch.utils.data.Subset(train_dataset, train_idx)
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=args.batchsize_cnn,
                                                       shuffle=True)

            val_idx = np.in1d(val_labels, train_digits).nonzero()[0]
            val_idx = np.random.permutation(val_idx)[:int(len(val_idx) * args.subset)]
            val_subset = torch.utils.data.Subset(val_dataset, val_idx)
            val_loader = torch.utils.data.DataLoader(val_subset,
                                                     batch_size=args.batchsize_cnn,
                                                     shuffle=True)

            cnn.reset_parameters(seed)
            cnn.train()
            optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
            best_val_acc = 0.

            pbar = tqdm(range(args.epoch_cnn))
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

            best_models.append(deepcopy(best_model))

        # perturb some models within
        if args.model_perturb is None:
            perturb = DefaultTransform()
        elif args.model_perturb == 'noise':
            perturb = AdditiveNoise(args.perturb_rate, args.noise_scale)
        elif args.model_perturb == 'sign':
            perturb = SignFlip(args.perturb_rate)
        else:
            raise NotImplementedError

        best_models = [perturb(m) for m in best_models]
        weights, labels = torch.utils.data.default_collate(best_models)

        aggregated_val_accs = []
        if args.exclusive:
            assert args.aggrPergraph * args.modeslPeraggr <= args.modelsPergraph, "models sent to more than one aggr!"
            edges_rows = torch.randperm(args.modelsPergraph)[:args.aggrPergraph * args.modeslPeraggr]
        else:
            edges_rows = torch.randint(0, args.modelsPergraph, (args.aggrPergraph * args.modeslPeraggr,))
        edges_cols = torch.repeat_interleave(torch.arange(args.aggrPergraph), args.modeslPeraggr)
        pbar = tqdm(range(args.aggrPergraph))
        for aggr_idx in pbar:
            selected_model_idx = edges_rows[aggr_idx * args.modeslPeraggr: (aggr_idx + 1) * args.modeslPeraggr]

            selected_weights = {k: w[selected_model_idx, ...] for k, w in weights.items()}

            agg_weights = fed_avg(selected_weights)
            cnn.load_state_dict(agg_weights)
            cnn.eval()
            acc = mnist_validation(val_loader_full, cnn)
            aggregated_val_accs.append(acc)

            perturb = 1 - labels[selected_model_idx].sum().item() / len(selected_model_idx)
            pbar.set_postfix({'acc': acc, 'perturb rate': perturb})

        edge_index = torch.vstack([edges_rows, edges_cols])
        data = HeteroData(aggregator={'x': torch.tensor(aggregated_val_accs)[:, None]},
                          clients__aggregator={'edge_index': edge_index},
                          aggregator__clients={'edge_index': edge_index[torch.tensor([1, 0])]},
                          y=labels)

        # linear mapping
        features = linear_mapping(weights, 1024)
        g = HeteroData(aggregator={'x': torch.tensor(aggregated_val_accs)[:, None]},
                       clients={'x': features},
                       clients__aggregator={'edge_index': edge_index},
                       aggregator__clients={'edge_index': edge_index[torch.tensor([1, 0])]},
                       y=labels
                       )

        torch.save(g, os.path.join(root, f'model{idx_graph}.pt'))
