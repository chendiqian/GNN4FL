from typing import Union, Tuple

from torch import Tensor
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_scatter import scatter


class MySAGEConv(MessagePassing):
    def __init__(self,
                 in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 normalize: bool = False,
                 bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'mean')
        super().__init__(**kwargs)

        self.aggr = kwargs['aggr']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self,
                x: Union[Tensor, Tuple[Tensor, Tensor]],
                edge_index: Tensor) -> Tensor:
        if isinstance(x, Tensor):
            x = (x, x)
        x_src, x_dst = x

        edge_index_src, edge_index_dst = edge_index

        msg = x_src[edge_index_src, :]
        msg = scatter(msg, edge_index_dst, dim=0, dim_size=x_dst.shape[0], reduce=self.aggr)

        out = self.lin_l(msg) + self.lin_r(x_dst)

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out
