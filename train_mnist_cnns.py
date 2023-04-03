import argparse
from copy import deepcopy
import random
import os
import yaml
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import torch

from customed_datasets.get_mnist import get_mnist
from customed_datasets.weight_datasets import DefaultTransform
from models.mnist_cnn import MNIST_CNN
from utils.train_utils import mnist_validation
from utils.fl_utils import LocalUpdate, fed_avg, make_gradient_ascent, make_all_to_label, mnist_get_client
from customed_datasets.graph_datasets import linear_mapping
from torch_geometric.data import HeteroData


# some training dynamics are taken from https://github.com/wenzhu23333/Federated-Learning
def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    # params for CNN training
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--models_per_epoch', type=int, default=20)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--global_epoch', type=int, default=10)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=24)
    # params for GNN creation
    parser.add_argument('--create_gnn', action='store_true')
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--aggrPergraph', type=int, default=2, help='number of aggr nodes per graph')
    parser.add_argument('--modeslPeraggr', type=int, default=2, help='number of param nodes per aggr')
    parser.add_argument('--exclusive', default=False, action='store_true',
                        help='whether one model is used ONCE for aggregation')
    parser.add_argument('--model_perturb', type=str, default=None, help='how to perturb data')
    parser.add_argument('--perturb_rate', type=float, default=0.3, help='ratio of models perturbed')
    parser.add_argument('--noise_scale', type=float, default=0.3, help='Gauss(mean=0, std=params.std() * noise_scale)')
    parser.add_argument('--ascent_steps', type=int, default=1)
    parser.add_argument('--target_dim', type=int, default=32, help='number of hidden dimensions for projection')
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    tensorboard_dir = f'./logs/train_mnist_perturb{args.model_perturb}'
    if not os.path.isdir('logs'):
        os.mkdir('logs')
    if not os.path.isdir(tensorboard_dir):
        os.mkdir(tensorboard_dir)
    folders = os.listdir(tensorboard_dir)
    writer = SummaryWriter(os.path.join(tensorboard_dir, f'{len(folders)}'))
    train_dataset, _, val_dataset, _ = get_mnist()
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batchsize,
                                             shuffle=True)

    cnn = MNIST_CNN()
    cnn.reset_parameters(args.seed)
    cnn.train()

    avg_weights_global = cnn.state_dict()

    if args.model_perturb is None:
        perturb = DefaultTransform()
    elif args.model_perturb == 'grad_ascent':
        perturb = make_gradient_ascent(torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True),
                                       cnn,
                                       args.ascent_steps,
                                       args.perturb_rate)
    elif args.model_perturb == 'label':
        perturb = make_all_to_label(torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True),
                                    cnn,
                                    args.ascent_steps,
                                    args.perturb_rate)
    else:
        raise NotImplementedError

    root = f'./{args.root}/{cnn}GraphDatasetsMPG{args.models_per_epoch}_APG{args.aggrPergraph}' \
           f'_MPA{args.modeslPeraggr}_steps{args.ascent_steps}_rate{args.perturb_rate}'
    if args.exclusive:
        root += 'Exclusive'
    if not os.path.isdir(root):
        os.mkdir(root)

    with open(os.path.join(root, 'config.yml'), 'w') as outfile:
        yaml.dump(args, outfile, default_flow_style=False)

    root = os.path.join(root, 'raw')
    if not os.path.isdir(root):
        os.mkdir(root)

    dict_user = mnist_get_client(len(train_dataset), args.num_client, True)
    id_users = list(dict_user.keys())

    best_acc = 0.
    learning_rate = [args.lr for i in range(len(id_users))]
    for epoch in range(args.global_epoch):
        print(f"=============== Training CNN ================")
        cnn.load_state_dict(avg_weights_global)
        if args.models_per_epoch >= len(id_users):
            selected_idx = list(range(len(id_users)))
        else:
            selected_idx = random.sample(list(range(len(id_users))), args.models_per_epoch)
        weights_locals = []
        for i in selected_idx:
            id_user = id_users[i]
            args.lr = learning_rate[id_user]
            train_subset = torch.utils.data.Subset(train_dataset,
                                                   np.array(list(dict_user[id_user])))
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=args.batchsize,
                                                       shuffle=True)
            local = LocalUpdate(args=args, dataloader=train_loader)
            w, _, curLR = local.train(net=deepcopy(cnn))
            learning_rate[i] = curLR
            weights_locals.append(deepcopy(w))

        aggr_weights_locals = torch.utils.data.default_collate(weights_locals)
        avg_weights_global = fed_avg(aggr_weights_locals)   # aggregated clean weights
        cnn.load_state_dict(avg_weights_global)

        # print accuracy
        cnn.eval()
        val_acc = mnist_validation(val_loader, cnn)
        best_acc = max(best_acc, val_acc)

        print(f'global epoch:, {epoch}, val acc: {val_acc}, best acc: {best_acc}')

        writer.add_scalar('val acc', val_acc, epoch)

        if not args.create_gnn:
            continue

        print(f"=============== Perturbing model weights ================")
        perturb_models = [perturb(deepcopy(m)) for m in weights_locals]
        weights, labels = torch.utils.data.default_collate(perturb_models)

        print(f"=============== Creating graphs ================")

        aggregated_val_accs = []
        if args.exclusive:
            assert args.aggrPergraph * args.modeslPeraggr <= args.models_per_epoch, "models sent to more than one aggr!"
            edges_rows = torch.randperm(args.models_per_epoch)[:args.aggrPergraph * args.modeslPeraggr]
        else:
            edges_rows = torch.randint(0, args.models_per_epoch, (args.aggrPergraph * args.modeslPeraggr,))
        edges_cols = torch.repeat_interleave(torch.arange(args.aggrPergraph), args.modeslPeraggr)
        pbar = tqdm(range(args.aggrPergraph))
        for aggr_idx in pbar:
            selected_model_idx = edges_rows[aggr_idx * args.modeslPeraggr: (aggr_idx + 1) * args.modeslPeraggr]

            selected_weights = {k: w[selected_model_idx, ...] for k, w in weights.items()}

            agg_weights = fed_avg(selected_weights)
            cnn.load_state_dict(agg_weights)
            cnn.eval()
            acc = mnist_validation(val_loader, cnn)
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

        files = os.listdir(root)
        files = [f for f in files if f.endswith('.pt')]
        torch.save(g, os.path.join(root, f'model{len(files)}.pt'))

    writer.flush()
    writer.close()
