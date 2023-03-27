import argparse
from copy import deepcopy

import numpy as np
import torch

from customed_datasets.get_femnist import get_femnist
from models.femnist_cnn import FEMNIST_CNN
from utils.train_utils import mnist_validation
from utils.fl_utils import LocalUpdate, fed_avg

# some training dynamics are taken from https://github.com/wenzhu23333/Federated-Learning
def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    parser.add_argument('--lr', type=float, default=0.01, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.995, help="learning rate decay each round")
    parser.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--global_epoch', type=int, default=50)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--seed', type=int, default=24)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    train_dataset, val_dataset = get_femnist()
    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batchsize,
                                             shuffle=True)

    cnn = FEMNIST_CNN()
    cnn.reset_parameters(args.seed)
    cnn.train()

    weights_global = cnn.state_dict()

    # root = './datasets/FEMNIST_CNN'
    # if not os.path.isdir(root):
    #     os.mkdir(root)
    # root = os.path.join(root, 'raw')
    # if not os.path.isdir(root):
    #     os.mkdir(root)
    # if not os.path.isdir(os.path.join(root, str(args.seed))):
    #     os.mkdir(os.path.join(root, str(args.seed)))

    dict_user = train_dataset.get_client_dic()
    id_users = list(dict_user.keys())

    acc_test = []
    learning_rate = [args.lr for i in range(len(id_users))]
    best_val_acc = 0.
    for epoch in range(args.global_epoch):
        weights_locals, loss_locals = [], []
        for i, id_user in enumerate(id_users):
            args.lr = learning_rate[id_user]
            train_subset = torch.utils.data.Subset(train_dataset,
                                                   np.array(list(dict_user[id_user])))
            train_loader = torch.utils.data.DataLoader(train_subset,
                                                       batch_size=args.batchsize,
                                                       shuffle=True)
            local = LocalUpdate(args=args, dataloader=train_loader)
            w, loss, curLR = local.train(net=deepcopy(cnn))
            learning_rate[i] = curLR
            weights_locals.append(deepcopy(w))
            loss_locals.append(deepcopy(loss))

        weights_locals = torch.utils.data.default_collate(weights_locals)
        # update global weights
        weights_global = fed_avg(weights_locals)
        # copy weight to net_glob
        cnn.load_state_dict(weights_global)

        # print accuracy
        cnn.eval()
        val_acc = mnist_validation(val_loader, cnn)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = deepcopy(cnn.state_dict())

        print(f'global epoch:, {epoch}, val acc: {val_acc}')
