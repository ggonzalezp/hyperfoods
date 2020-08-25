import torch
from torch.nn import Parameter
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops
from torch_scatter import scatter_max, scatter_add
from .message_passing import MessagePassing

from ..inits import glorot, zeros


def softmax(src, index, dim=0, num_nodes=None):
    num_nodes = index.max().item() + 1

    max = scatter_max(src, index, dim=dim, dim_size=num_nodes)[0][:, index]
    out = src - scatter_max(src, index, dim=dim, dim_size=num_nodes)[0][:,
                                                                        index]
    out = out.exp()
    out = out / (
        scatter_add(out, index, dim=dim, dim_size=num_nodes)[:, index] + 1e-16)

    return out


class GATConv(MessagePassing):
    def __init__(self,
                 in_channels,
                 out_channels,
                 heads=1,
                 concat=True,
                 negative_slope=0.2,
                 dropout=0,
                 bias=True,
                 **kwargs):
        super(GATConv, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        self.weight = Parameter(torch.Tensor(in_channels,
                                             heads * out_channels))
        self.att = Parameter(torch.Tensor(1, 1, heads, 2 * out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.att)
        zeros(self.bias)

    def forward(self, x, edge_index, size=None):
        """"""
        if size is None and torch.is_tensor(x):
            edge_index, _ = remove_self_loops(edge_index)
            edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(1))

        if torch.is_tensor(x):
            x = torch.matmul(x, self.weight)
        else:
            x = (None if x[0] is None else torch.matmul(x[0], self.weight),
                 None if x[1] is None else torch.matmul(x[1], self.weight))

        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        # Compute attention coefficients.
        bs = x_j.size(0)
        x_j = x_j.view(bs, -1, self.heads, self.out_channels)
        if x_i is None:
            alpha = (x_j * self.att[:, :, :, self.out_channels:]).sum(dim=-1)
        else:
            x_i = x_i.view(bs, -1, self.heads, self.out_channels)
            alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, edge_index_i, dim=1, num_nodes=size_i)

        # Sample attention coefficients stochastically.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(bs, -1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(aggr_out.size(0), -1,
                                     self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=2)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)
