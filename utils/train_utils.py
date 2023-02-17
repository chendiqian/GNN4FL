import torch
from torch.utils.data import DataLoader


@torch.no_grad()
def mnist_validation(dataloader: DataLoader,
                     model: torch.nn.Module):
    corrects = 0
    counts = 0
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        # forward + backward + optimize
        outputs = model(inputs)
        pred = outputs.argmax(1)
        corrects += (pred == labels).sum()
        counts += inputs.shape[0]

    return (corrects / counts).item()
