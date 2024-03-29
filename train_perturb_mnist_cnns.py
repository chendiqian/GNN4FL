import argparse
from copy import deepcopy
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
import torch

from customed_datasets.get_mnist import get_mnist
from customed_datasets.weight_datasets import DefaultTransform
from models.mnist_cnn import MNIST_CNN
from utils.train_utils import mnist_validation
from utils.fl_utils import LocalUpdate, fed_avg, make_gradient_ascent, make_all_to_label, mnist_get_client
from customed_datasets.graph_datasets import linear_mapping
from torch_geometric.data import HeteroData
from models.hetero_gnn import HeteroGNN


# some training dynamics are taken from https://github.com/wenzhu23333/Federated-Learning
def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    # params for CNN training
    parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('--models_per_epoch', type=int, default=20)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--global_epoch', type=int, default=10)
    parser.add_argument('--num_client', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=24)
    # params for perturbation
    parser.add_argument('--model_perturb', type=str, default=None, help='how to perturb data')
    parser.add_argument('--perturb_rate', type=float, default=0.5, help='ratio of models perturbed')
    parser.add_argument('--ascent_steps', type=int, default=1)
    # params for GNN creation
    parser.add_argument('--filter_models', action='store_true')
    parser.add_argument('--modelpath', type=str, default='')
    parser.add_argument('--hidden', type=int, default=128)
    parser.add_argument('--layers', type=int, default=3)
    parser.add_argument('--in_feature', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--aggrPergraph', type=int, default=2, help='number of aggr nodes per graph')
    parser.add_argument('--modeslPeraggr', type=int, default=2, help='number of param nodes per aggr')
    parser.add_argument('--exclusive', default=False, action='store_true',
                        help='whether one model is used ONCE for aggregation')

    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    tensorboard_dir = f'./logs/train_mnist_perturb{args.model_perturb}'
    tensorboard_dir += '_filtered' if args.filter_models else '_nofilter'
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

    model = HeteroGNN(feature_in_channels=args.in_feature,
                      aggr_in_channels=1,
                      hidden_channels=args.hidden,
                      out_channels=1,
                      num_layers=args.layers,
                      dropout=args.dropout)
    if args.modelpath:
        model.load_state_dict(torch.load(args.modelpath))
    model.eval()

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

    dict_user = mnist_get_client(len(train_dataset), args.num_client, True)
    id_users = list(dict_user.keys())

    acc_test = []
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
            train_subset = torch.utils.data.Subset(train_dataset, dict_user[id_user])
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=args.batchsize,
                                                       shuffle=True)
            local = LocalUpdate(args=args, dataloader=train_loader)
            w, _, curLR = local.train(net=deepcopy(cnn))
            learning_rate[i] = curLR
            weights_locals.append(deepcopy(w))

        print(f"=============== Perturbing model weights ================")
        perturb_models = [perturb(deepcopy(m)) for m in weights_locals]
        perturbed_weights, labels = torch.utils.data.default_collate(perturb_models)

        if args.filter_models:
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

                selected_weights = {k: w[selected_model_idx, ...] for k, w in perturbed_weights.items()}

                agg_weights = fed_avg(selected_weights)
                cnn.load_state_dict(agg_weights)
                cnn.eval()
                acc = mnist_validation(val_loader, cnn)
                aggregated_val_accs.append(acc)

                perturb_rate = 1 - labels[selected_model_idx].sum().item() / len(selected_model_idx)
                pbar.set_postfix({'acc': acc, 'perturb rate': perturb_rate})

            edge_index = torch.vstack([edges_rows, edges_cols])
            data = HeteroData(aggregator={'x': torch.tensor(aggregated_val_accs)[:, None]},
                              clients__aggregator={'edge_index': edge_index},
                              aggregator__clients={'edge_index': edge_index[torch.tensor([1, 0])]},
                              y=labels)

            # linear mapping
            features = linear_mapping(perturbed_weights, args.in_feature)
            g = HeteroData(aggregator={'x': torch.tensor(aggregated_val_accs)[:, None]},
                           clients={'x': features},
                           clients__aggregator={'edge_index': edge_index},
                           aggregator__clients={'edge_index': edge_index[torch.tensor([1, 0])]},
                           y=labels
                           )

            prediction = model(g.x_dict, g.edge_index_dict)
            reliable_model_idx = torch.where(prediction > 0.)[0].numpy()
            perturb_idx = torch.where(prediction < 0.)[0].numpy()
            true_reliable_idx = torch.where(labels)[0].numpy()
            true_perturb_idx = torch.where(labels == 0)[0].numpy()
            print(
                f'true pos: {np.in1d(reliable_model_idx, true_reliable_idx).sum() / len(true_reliable_idx)}'
                f'unfiltered: {np.in1d(reliable_model_idx, true_perturb_idx).sum() / len(true_perturb_idx)}')

            # todo: this is debugging!
            if len(reliable_model_idx) == 0:  # no reliable
                pass
            selected_weights = {k: w[reliable_model_idx] for k, w in
                                perturbed_weights.items()}
            if len(reliable_model_idx) > 1:
                selected_weights = fed_avg(selected_weights)
            avg_weights_global = selected_weights
        else:
            avg_weights_global = fed_avg(perturbed_weights)

        cnn.load_state_dict(avg_weights_global)

        # print accuracy
        cnn.eval()
        val_acc = mnist_validation(val_loader, cnn)

        print(f'global epoch:, {epoch}, val acc: {val_acc}')
        writer.add_scalar('val acc', val_acc, epoch)

    writer.flush()
    writer.close()
