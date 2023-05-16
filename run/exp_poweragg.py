from torch_geometric.nn import GraphConv
import torch
from torch import nn
from typing import Optional
from torch import Tensor
from run import run
from torch_geometric.nn.aggr import Aggregation, SoftmaxAggregation


class PowerMeanAggregationCustom(Aggregation):

	def __init__(self, p: float = 1.0, learn: bool = False):
		super().__init__()
		self._init_p = p
		self.learn = learn
		self.p = nn.Parameter(torch.Tensor(1)) if learn else p
		self.reset_parameters()

	def reset_parameters(self):
		if isinstance(self.p, Tensor):
			self.p.data.fill_(self._init_p)

	def forward(self, x: Tensor, index: Optional[Tensor] = None,
				ptr: Optional[Tensor] = None, dim_size: Optional[int] = None,
				dim: int = -2) -> Tensor:
		x = x.clamp(min=1e-6, max=100).pow(self.p)
		out = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
		out = out.clamp(min=1e-6, max=100).pow(1. / self.p)
		return out


def main():

	configs = {}

	configs["poweragg"] = {
			"layer": GraphConv,
			"num_layers": 4,
			"hidden_size": 64,
			"layer_kwargs": {"aggr": PowerMeanAggregationCustom},
			"agg_kwargs": {"learn": True},
		}

	configs["softmaxagg"] = {
		"layer": GraphConv,
		"num_layers": 4,
		"hidden_size": 64,
		"layer_kwargs": {"aggr": SoftmaxAggregation},
		"agg_kwargs": {"learn": True},
	}


	run(configs=configs, datasets=["PATTERN", "CLUSTER", "MNIST", "CIFAR10"],
		num_seeds=10, jobs_per_gpu=1, use_gpu=False, batch_size=32, single_thread=True,
		wandb_project="genagg", save=False)


if __name__ == '__main__':
	main()