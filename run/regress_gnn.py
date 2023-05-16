import numpy as np
import wandb
import inspect
import torch
from torch_geometric.nn import GraphConv, Sequential, MessagePassing
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from scipy.sparse import coo_matrix
from exp_poweragg import PowerMeanAggregationCustom
from torch_geometric.nn.aggr import *
import gym
from gym import spaces

from genagg import GenAgg

project = "genagg-regress-gnn"
user = ""

def experiments():
	trials = {
		"genagg": [
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": GenAgg}, "aggr_kwargs": {"layer_sizes": (1, 2, 2, 4)}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))}
		],
		"mean": [
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": MeanAggregation}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))}
		],
		"poweragg": [
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": PowerMeanAggregationCustom}, "aggr_kwargs": {"learn": True}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))}
		],
		"softmaxagg": [
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "max", "mixfunc": lambda local, neighbours: np.max(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "min", "mixfunc": lambda local, neighbours: np.min(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "max-mag", "mixfunc": lambda local, neighbours: np.max(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "min-mag", "mixfunc": lambda local, neighbours: np.min(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "mean", "mixfunc": lambda local, neighbours: np.mean(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "sum", "mixfunc": lambda local, neighbours: np.sum(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "std", "mixfunc": lambda local, neighbours: np.std(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "2-norm", "mixfunc": lambda local, neighbours: np.linalg.norm(neighbours, axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "rms", "mixfunc": lambda local, neighbours: np.sqrt(np.mean(neighbours ** 2, axis=0))},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "prod-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "geom-mean-abs", "mixfunc": lambda local, neighbours: np.prod(np.abs(neighbours), axis=0) ** (1 / neighbours.shape[0])},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "harm-mean-abs", "mixfunc": lambda local, neighbours: neighbours.shape[0] / np.sum(1 / np.abs(neighbours), axis=0)},
			{"model_kwargs": {"aggr": SoftmaxAggregation}, "aggr_kwargs": {"learn": True}, "agg": "log-sum-exp", "mixfunc": lambda local, neighbours: np.log(np.sum(np.exp(neighbours), axis=0))}
		],
	}
	default = {
		"model_kwargs": {"aggr": GenAgg, "activation": torch.nn.Mish, "layer": GraphConv, "nlayers": 4},
		"aggr_kwargs": {},
		"layer_kwargs": {},
		"mixfunc": None,
		"agg": "n/a",
		"log_f": True,
		"train_iter": 10000,
		"epoch_size": 1024,
		"batch_size": 1024,
		"n_agents": 8,
		"obs_size": 1,
		"runs": 10,
	}
	n_runs = default["runs"]
	for r in range(n_runs):
		for name, trial in trials.items():
			if not isinstance(trial, list):
				trial = [trial]
			for cfg in trial:
				cfg = cfg.copy()
				config = default.copy()
				for key, val in config.items():
					if isinstance(val, dict):
						val.update(cfg.get(key, {}))
						if key in cfg:
							del cfg[key]
				config.update(cfg)
				config["name"] = name
				if r >= config.get("runs", n_runs):
					break
				del config["runs"]
				run(**config)


def run(
	mixfunc = lambda local, neighbours: np.min(neighbours, axis=0),
	name = "n/a",
	agg = "n/a",
	model_kwargs = {},
	aggr_kwargs = {},
	layer_kwargs = {},
	train_iter = 10000,
	epoch_size = 1024,
	batch_size = 64,
	n_agents = 16,
	obs_size = 1,
	log_f = False,
	):

	hidden_layers = []
	for k in range(model_kwargs["nlayers"]):
		aggr = model_kwargs["aggr"](**aggr_kwargs)
		layer_class = model_kwargs["layer"]
		activation = model_kwargs["activation"]

		layerk = layer_class(in_channels=obs_size, out_channels=obs_size, aggr=aggr, **layer_kwargs)

		hidden_layers.append((layerk, "x, edge_index -> x"))
		if k < model_kwargs["nlayers"]-1:
			hidden_layers.append((activation(), "x -> x"))
	model = Sequential("x, edge_index", hidden_layers)

	if torch.cuda.is_available():
		device = "cuda:0"
	else:
		device = "cpu"
	model = model.to(device)

	env_config = {
		"n_agents": n_agents,
		"obs_size": obs_size,
		"mixfunc": mixfunc,
	}

	env = PropEnv(**env_config)

	env_config.update({"agg": agg})

	if project is not None:
		wandb.init(
			entity=user,
			project=project,
			group=f"{name}",
			config=env_config,
		)

	optimiser = torch.optim.Adam(model.parameters())
	loss_func = torch.nn.MSELoss()

	if log_f:
		for layer in model:
			if isinstance(layer, MessagePassing) and isinstance(layer.aggr_module, GenAgg):
				layer.aggr_module.visualize = True

	for t in range(train_iter):

		data = generate_data(env=env, steps=epoch_size)

		A = data["adj"][:, -1]
		x = data["obs"][:, -1]
		y = data["y"][:, -1]

		dataobj = adj2geom(A=A, x=x, y=y).to(device)
		loader = DataLoader(dataobj, batch_size=batch_size)

		avg_loss = 0
		avg_corr = 0

		for batch in loader:

			y_preds = model(x=batch.x, edge_index=batch.edge_index)

			loss = loss_func(y_preds, batch.y.float())
			loss.backward()
			optimiser.step()
			optimiser.zero_grad()

			avg_loss += loss.item() * batch.num_graphs
			corr = np.corrcoef(
				y_preds.detach().cpu().numpy().reshape(-1), batch.y.detach().cpu().numpy().reshape(-1), rowvar=False
			)[0, 1]
			avg_corr += corr * batch.num_graphs

		if isinstance(model_kwargs["aggr"], GenAgg) and log_f and t % 64 == 0:
			v_mean = model[0].aggr_module.v_mean
			v_std = model[0].aggr_module.v_std
			v_f = model[0].aggr_module.f
			v_input = torch.linspace(v_mean - 2 * v_std, v_mean + 2 * v_std, 200, device=v_mean.device)
			with torch.no_grad():
				v_output = v_f.forward(v_input.unsqueeze(-1))
			v_input = v_input.detach().cpu().numpy()
			v_output = v_output.detach().cpu().numpy()
			fig = make_subplots()
			for i in range(v_output.shape[1]):
				trace = go.Scatter(x=v_input, y=v_output[:,i])
				fig.add_trace(trace)
			if project is not None:
				wandb.log({"f": fig}, commit=False)

		alpha = 0
		beta = 0
		param = 0
		if model_kwargs["aggr"] == GenAgg:
			alpha = model[0].aggr_module.a.data.item()
			beta = model[0].aggr_module.c.data.item()
		elif model_kwargs["aggr"] == PowerMeanAggregationCustom:
			param = model[0].aggr_module.p.data.item()
		elif model_kwargs["aggr"] == SoftmaxAggregation:
			param = model[0].aggr_module.t.data.item()
		avg_loss /= len(loader.dataset)
		avg_corr /= len(loader.dataset)
		if project is not None:
			wandb.log({
				"loss": avg_loss,
				"corr": avg_corr,
				"alpha": alpha,
				"beta": beta,
				"param": param,
			})

	if project is not None:
		wandb.finish()


def generate_data(env, steps):
	observations = [None] * steps
	adjacencies = [None] * steps
	ys = [None] * steps
	i = 0
	t = 0
	while t < steps:
		done = False
		obs_i = []
		adj_i = []
		y_i = []
		while done is False:
			action = np.stack(env.action_space.sample())
			obs, reward, done, info = env.step(action)
			obs_i.append(obs["obs"])
			adj_i.append(obs["adj"])
			y_i.append(obs["prop"])
			t += 1
		observations[i] = np.stack(obs_i)
		adjacencies[i] = np.stack(adj_i)
		ys[i] = np.stack(y_i)
		i += 1
		env.reset()
	observations = np.stack(observations[:i])
	adjacencies = np.stack(adjacencies[:i])
	ys = np.stack(ys[:i])
	return {"obs": observations, "adj": adjacencies, "y": ys}


class PropEnv(gym.Env):
	def __init__(self, **kwargs):
		self.n_agents = 16
		self.n_actions = 0
		self.episode_length = 1
		self.density = 0.3
		self.mixfunc = lambda local, neighbours: np.max(neighbours, axis=0)[0]
		self.obs_size = 1
		# p = 0: np.prod(neighbours) ** (1/len(neighbours))
		# p = -1: len(neighbours) / np.sum(1/neighbours, axis=0)
		# p = 1: np.mean(neighbours, axis=0)
		# p = -inf: np.min(neighbours, axis=0)
		# p = inf: np.max(neighbours, axis=0)

		self.set_params(kwargs)

		self.observation_space_single = spaces.Box(
			low=np.array([-np.inf] * self.obs_size), high=np.array([np.inf] * self.obs_size)
		)
		self.action_space_single = spaces.Box(
			low=0.0, high=1.0, shape=(self.n_actions,), dtype=np.float32
		)

		self.action_space = spaces.Tuple(
			[self.action_space_single for _ in range(self.n_agents)]
		)
		self.observation_space = spaces.Dict(
			{
				"obs": spaces.Tuple(
					[self.observation_space_single for _ in range(self.n_agents)]
				),
				"prop": spaces.Tuple(
					[self.observation_space_single for _ in range(self.n_agents)]
				),
				"adj": spaces.Tuple(
					[spaces.MultiBinary(self.n_agents) for _ in range(self.n_agents)]
				),
			}
		)

		PropEnv.reset(self)

	def set_params(self, params):
		for name, value in params.items():
			setattr(self, name, value)

	def mix_states(self, local_state):
		neighbour_states = self.adjacency[:, :, None] * local_state[None, :, :]
		new_state = np.array(
			[
				self.mixfunc(local_state[i], neighbour_states[i, self.adjacency[i], :])
				for i in range(local_state.shape[0])
			]
		)
		return new_state

	def step(self, action):
		self.t += 1
		state = self.state
		adj = self.adjacency
		prop = self.mix_states(state)
		self.state = prop
		return (
			{
				"obs": tuple(state.astype(np.float32)),
				"prop": tuple(prop.astype(np.float32)),
				"adj": tuple(adj.astype(np.byte)),
			},
			0,
			self.t >= self.episode_length,
			{},
		)

	def reset(self):
		self.t = 0
		self.state = np.random.randn(self.n_agents, self.obs_size)
		self.adjacency = self.gen_adjacency()
		obs = {
			"obs": tuple(self.state.astype(np.float32)),
			"prop": tuple(np.zeros((self.n_agents, 1), dtype=np.float32)),
			"adj": tuple(self.adjacency.astype(np.byte)),
		}
		return obs

	def gen_adjacency(self):
		orig_density = np.sqrt(self.density)
		while True:
			A = torch.rand(self.n_agents, self.n_agents) < orig_density
			torch.diagonal(A).fill_(0)
			A = A * A.T
			if torch.any(torch.sum(A, dim=0) == 0).item():
				continue
			return A.numpy()


def adj2geom(A, **kwargs):
    A = torch.as_tensor(A)
    for name, val in kwargs.items():
        kwargs[name] = torch.as_tensor(val)
    numdim = len(A.shape)
    numbatchdim = numdim - 2
    batchdimsize = int(np.prod(A.shape[:numbatchdim]))
    A = A.reshape(batchdimsize, *A.shape[numbatchdim:])
    for name, val in kwargs.items():
        kwargs[name] = val.reshape(batchdimsize, *val.shape[numbatchdim:])
    return dense_to_geometric(A=A, **kwargs)


def dense_to_geometric(A, **kwargs):
    edge_list = [from_scipy_sparse_matrix(coo_matrix(A[i])) for i in range(A.shape[0])]
    arg_slice = lambda i: {key: val[i] for key, val in kwargs.items()}
    data_list = [
        Data(edge_index=edge_index, edge_attr=edge_attr, **(arg_slice(i)))
        for i, (edge_index, edge_attr) in enumerate(edge_list)
    ]
    return Batch.from_data_list(data_list)


if __name__ == "__main__":
	experiments()
