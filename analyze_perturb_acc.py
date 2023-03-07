import argparse
import os
import random

import torch
from tqdm import tqdm
import numpy as np

from customed_datasets.get_mnist import get_mnist
from models.mnist_cnn import MNIST_CNN
from utils.train_utils import mnist_validation
from customed_datasets.weight_datasets import AdditiveNoise, SignFlip
from utils.fl_utils import make_gradient_ascent, make_all_to_label


def args_parser():
    parser = argparse.ArgumentParser(description='hyper params for creating graph dataset')
    parser.add_argument('--root', type=str, default='./datasets')
    parser.add_argument('--cnn_db', type=str, help='where the cnn models are stored')
    parser.add_argument('--num_models', type=int, default=1, help='how many models to analyze')
    parser.add_argument('--model_perturb', type=str, default=None, help='how to perturb data')
    parser.add_argument('--noise_scale', type=float, default=0.3, help='Gauss(mean=0, std=params.std() * noise_scale)')
    parser.add_argument('--ascent_steps', type=int, default=1)
    return parser.parse_args()


if __name__ == '__main__':
    args = args_parser()
    train_dataset, train_labels, val_dataset, _ = get_mnist()
    val_loader_full = torch.utils.data.DataLoader(val_dataset, batch_size=256, shuffle=True)
    num_classes = max(train_labels) + 1
    cnn = MNIST_CNN()

    if args.model_perturb == 'noise':
        perturb = AdditiveNoise(1., args.noise_scale)
    elif args.model_perturb == 'sign':
        perturb = SignFlip(1.)
    elif args.model_perturb == 'grad_ascent':
        perturb = make_gradient_ascent(torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True),
                                       MNIST_CNN(),
                                       args.ascent_steps,
                                       1.)
    elif args.model_perturb == 'label':
        perturb = make_all_to_label(torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True),
                                    MNIST_CNN(),
                                    args.ascent_steps,
                                    1.)
    else:
        raise NotImplementedError

    perturbed_val_accs = []

    pbar = tqdm(range(args.num_models))
    for idx_model in pbar:
        # create ONE graph, models share the same seed
        seed = random.choice(os.listdir(f'./{args.root}/{args.cnn_db}/raw'))
        model = random.choice(os.listdir(f'./{args.root}/{args.cnn_db}/raw/{seed}'))
        model = torch.load(os.path.join(f'./{args.root}/{args.cnn_db}/raw/{seed}', model))

        model, _ = perturb(model)

        cnn.load_state_dict(model)
        cnn.eval()
        acc = mnist_validation(val_loader_full, cnn)
        perturbed_val_accs.append(acc)

        pbar.set_postfix({'acc': acc})

    print(f'mean: {np.mean(perturbed_val_accs)}, std: {np.std(perturbed_val_accs)}')
