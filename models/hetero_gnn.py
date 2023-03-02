from typing import Dict

import torch
from torch_geometric.typing import NodeType, EdgeType, Adj
from .my_sage_conv import MySAGEConv
from torch import Tensor, nn


class HeteroConv(nn.Module):
    def __init__(self, convs: Dict[EdgeType, nn.Module]):
        super().__init__()
        self.convs = nn.ModuleDict({'__'.join(k): v for k, v in convs.items()})

    def reset_parameters(self):
        for conv in self.convs.values():
            conv.reset_parameters()

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Dict[NodeType, Tensor]:
        out_dict = dict()
        for edge_type, edge_index in edge_index_dict.items():
            src, _, dst = edge_type
            str_edge_type = '__'.join(edge_type)
            conv = self.convs[str_edge_type]
            out = conv((x_dict[src], x_dict[dst]), edge_index)
            out_dict[dst] = out

        return out_dict

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(num_relations={len(self.convs)})'


class HeteroGNN(nn.Module):
    def __init__(self,
                 feature_in_channels: int,
                 aggr_in_channels: int,
                 hidden_channels: int,
                 out_channels: int,
                 num_layers: int):
        super().__init__()

        self.convs = nn.ModuleList()
        self.convs.append(HeteroConv({
            ('clients', 'to', 'aggregator'): MySAGEConv((feature_in_channels, aggr_in_channels), hidden_channels),
            ('aggregator', 'to', 'clients'): MySAGEConv((aggr_in_channels, feature_in_channels), hidden_channels),
        }))
        for _ in range(num_layers - 1):
            conv = HeteroConv({
                ('clients', 'to', 'aggregator'): MySAGEConv((hidden_channels, hidden_channels), hidden_channels),
                ('aggregator', 'to', 'clients'): MySAGEConv((hidden_channels, hidden_channels), hidden_channels),
            })
            self.convs.append(conv)

        self.lin = nn.Linear(hidden_channels, out_channels)
        self.nonlinear = nn.LeakyReLU(negative_slope=0.1, inplace=False)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: self.nonlinear(x) for key, x in x_dict.items()}
        return self.lin(x_dict['clients']).squeeze(1)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        self.lin.reset_parameters()
