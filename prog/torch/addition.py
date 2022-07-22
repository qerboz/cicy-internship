from typing import List, Optional, Union
import torch
from torch import Tensor
from torch_geometric.nn import global_add_pool,global_max_pool,global_mean_pool


class GlobalPooling(torch.nn.Module):
    r"""A global pooling module that wraps the usage of
    :meth:`~torch_geometric.nn.glob.global_add_pool`,
    :meth:`~torch_geometric.nn.glob.global_mean_pool` and
    :meth:`~torch_geometric.nn.glob.global_max_pool` into a single module.

    Args:
        aggr (string or List[str]): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
    """
    def __init__(self, aggr: Union[str, List[str]]):
        super().__init__()

        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        assert len(set(self.aggrs) | {'sum', 'add', 'mean', 'max'}) == 4

    def forward(self, x: Tensor, batch: Optional[Tensor],
                size: Optional[int] = None) -> Tensor:
        """"""
        xs: List[Tensor] = []

        for aggr in self.aggrs:
            if aggr == 'sum' or aggr == 'add':
                xs.append(global_add_pool(x, batch, size))
            elif aggr == 'mean':
                xs.append(global_mean_pool(x, batch, size))
            elif aggr == 'max':
                xs.append(global_max_pool(x, batch, size))

        return xs[0] if len(xs) == 1 else torch.cat(xs, dim=-1)


    def __repr__(self) -> str:
        aggr = self.aggrs[0] if len(self.aggrs) == 1 else self.aggrs
        return f'{self.__class__.__name__}(aggr={aggr})'
