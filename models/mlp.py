import torch


class MLP(torch.nn.Module):
    def __init__(self, in_feature, hidden, layers, num_classes, dropout=0.):
        super().__init__()
        assert layers >= 2
        self.lins = torch.nn.ModuleList([torch.nn.Linear(in_feature, hidden),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(dropout)
                                         ])
        for l in range(layers - 2):
            self.lins.append(torch.nn.Linear(hidden, hidden))
            self.lins.append(torch.nn.ReLU())
            self.lins.append(torch.nn.Dropout(dropout))
        self.lins.append(torch.nn.Linear(hidden, num_classes))

    def reset_parameters(self):
        for l in self.lins:
            if isinstance(l, torch.nn.Linear):
                l.reset_parameters()

    def forward(self, x):
        for l in self.lins:
            x = l(x)
        return x.squeeze()
