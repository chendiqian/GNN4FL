import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import MNIST


def get_mnist(subset: int = None):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5,), (0.5,))])
    train_dataset = MNIST('./datasets', train=True, download=True, transform=transform)

    val_dataset = MNIST('./datasets', train=False, download=True, transform=transform)
    if subset is not None:
        np.random.seed(2023)
        val_dataset = torch.utils.data.Subset(val_dataset,
                                              np.random.randint(0, len(val_dataset), subset))
    train_labels = [y for _, y in train_dataset]
    val_labels = [y for _, y in val_dataset]
    return train_dataset, train_labels, val_dataset, val_labels
