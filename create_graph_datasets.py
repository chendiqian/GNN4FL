import argparse
import os
import yaml
import random

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
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--cnn_db', type=str, help='where the cnn models are stored')
    parser.add_argument('--num_graphs', type=int, default=1, help='how many graphs you want in a graph dataset')
    parser.add_argument('--modelsPergraph', type=int, default=3, help='number of param nodes per graph')
    parser.add_argument('--aggrPergraph', type=int, default=2, help='number of aggr nodes per graph')
    parser.add_argument('--modeslPeraggr', type=int, default=2, help='number of param nodes per aggr')
    parser.add_argument('--exclusive', default=False, action='store_true',
                        help='whether one model is used ONCE for aggregation')
    parser.add_argument('--model_perturb', type=str, default=None, help='how to perturb data')
    parser.add_argument('--perturb_rate', type=float, default=0.3, help='ratio of models perturbed')
    parser.add_argument('--noise_scale', type=float, default=0.3, help='Gauss(mean=0, std=params.std() * noise_scale)')
    parser.add_argument('--target_dim', type=int, default=32, help='number of hidden dimensions for projection')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    _, train_labels, val_dataset, _ = get_mnist()
    val_loader_full = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)
    num_classes = max(train_labels) + 1
    cnn = MNIST_CNN()
    criterion = torch.nn.CrossEntropyLoss()

    root = f'./{args.root}/GraphDatasetsMPG{args.modelsPergraph}_APG{args.aggrPergraph}_MPA{args.modeslPeraggr}'
    if args.exclusive:
        root += 'Exclusive'
    if not os.path.isdir(root):
        os.mkdir(root)

    with open(os.path.join(root, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

    root = os.path.join(root, 'raw')
    if not os.path.isdir(root):
        os.mkdir(root)

    for idx_graph in range(args.num_graphs):
        # create ONE graph, models share the same seed
        seed = random.choice(os.listdir(f'./{args.root}/{args.cnn_db}/raw'))
        models = random.sample(os.listdir(f'./{args.root}/{args.cnn_db}/raw/{seed}'), args.modelsPergraph)
        models = [torch.load(os.path.join(f'./{args.root}/{args.cnn_db}/raw/{seed}', m)) for m in models]

        # perturb some models within
        if args.model_perturb is None:
            perturb = DefaultTransform()
        elif args.model_perturb == 'noise':
            perturb = AdditiveNoise(args.perturb_rate, args.noise_scale)
        elif args.model_perturb == 'sign':
            perturb = SignFlip(args.perturb_rate)
        else:
            raise NotImplementedError

        models = [perturb(m) for m in models]
        weights, labels = torch.utils.data.default_collate(models)

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
        features = linear_mapping(weights, args.target_dim)
        g = HeteroData(aggregator={'x': torch.tensor(aggregated_val_accs)[:, None]},
                       clients={'x': features},
                       clients__aggregator={'edge_index': edge_index},
                       aggregator__clients={'edge_index': edge_index[torch.tensor([1, 0])]},
                       y=labels
                       )

        torch.save(g, os.path.join(root, f'model{idx_graph}.pt'))
