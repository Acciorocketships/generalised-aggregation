from torch_geometric.nn import GraphConv
from torch_geometric.nn.aggr import *
import torch
from run import run


class ScaleAggregation(Aggregation):

	def __init__(self, aggr, scale=0):
		super().__init__()
		self.aggr = aggr
		self.scale = scale

	def get_n(self, x, index=None, ptr=None, dim_size=None, dim=-2):
		n = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce='sum')
		n[n==0] = 1
		return n

	def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
		n = self.get_n(x, index, ptr, dim_size, dim)
		y = self.aggr(x, index, ptr, dim_size, dim)
		out = y * (n ** self.scale)
		return out


class PNA(Aggregation):

	def __init__(self, in_channels, out_channels, aggr_list, **kwargs):
		super().__init__()
		self.num_aggr = len(aggr_list)
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.multi_aggr_channels = self.out_channels * self.num_aggr
		self.multi_aggr = MultiAggregation(aggrs=aggr_list, mode="cat", mode_kwargs={"in_channels": in_channels, "out_channels": out_channels}, **kwargs)
		self.encode = torch.nn.Linear(self.multi_aggr_channels, self.out_channels)

	def forward(self, x, index=None, ptr=None, dim_size=None, dim=-2):
		y = self.multi_aggr(x, index, ptr, dim_size, dim)
		out = self.encode(y)
		return out


def main():

	configs = {}

	configs["pna-scalers-cat"] = {
		"layer": GraphConv,
		"num_layers": 4,
		"hidden_size": 64,
		"agg_kwargs": lambda i: {"in_channels": 64, "out_channels": 64,
						"aggr_list": [ScaleAggregation(MeanAggregation(), scale=-1), ScaleAggregation(MinAggregation(), scale=-1), ScaleAggregation(MaxAggregation(), scale=-1), ScaleAggregation(StdAggregation(), scale=-1),
									 ScaleAggregation(MeanAggregation(), scale= 0), ScaleAggregation(MinAggregation(), scale= 0), ScaleAggregation(MaxAggregation(), scale= 0), ScaleAggregation(StdAggregation(), scale= 0),
									 ScaleAggregation(MeanAggregation(), scale= 1), ScaleAggregation(MinAggregation(), scale= 1), ScaleAggregation(MaxAggregation(), scale= 1), ScaleAggregation(StdAggregation(), scale= 1)]},
		"layer_kwargs": {"aggr": PNA},
	}

	run(configs=configs, datasets=["PATTERN", "CLUSTER", "MNIST", "CIFAR10"],
		num_seeds=10, jobs_per_gpu=1, use_gpu=True, batch_size=32, single_thread=True,
		wandb_project="genagg", save=False)


if __name__ == '__main__':
	main()