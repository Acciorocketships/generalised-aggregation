from torch_geometric.nn import GraphConv
from torch import nn
from run import run
from torch_geometric.nn.aggr import *


def main():

	configs = {}

	configs["sum"] = {
		"layer": GraphConv,
		"num_layers": 4,
		"hidden_size": 64,
		"agg_kwargs": {},
		"layer_kwargs": {"aggr": SumAggregation},
	}

	configs["max"] = {
		"layer": GraphConv,
		"num_layers": 4,
		"hidden_size": 64,
		"agg_kwargs": {},
		"layer_kwargs": {"aggr": MaxAggregation},
	}

	configs["mean"] = {
			"layer": GraphConv,
			"num_layers": 4,
			"hidden_size": 64,
			"agg_kwargs": {},
			"layer_kwargs": {"aggr": MeanAggregation},
		}
		

	run(configs=configs, datasets=["PATTERN", "CLUSTER", "MNIST", "CIFAR10"],
		num_seeds=10, jobs_per_gpu=1, use_gpu=True, batch_size=32, single_thread=True,
		wandb_project="genagg", save=False)


if __name__ == '__main__':
	main()