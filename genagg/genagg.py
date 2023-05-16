import torch
from torch import nn
import inspect
from genagg.mlp_forrev import MLPForwardReverse
from torch_geometric.nn.aggr import Aggregation


class GenAgg(Aggregation):
	def __init__(
		self,
		f=None,
		layer_sizes=(1,2,2,4),
		std=True,
		scale=True,
		**kwargs,
	):
		super().__init__()
		self.visualize = False
		if f is None:
			self.f = MLPForwardReverse(
				layer_sizes=layer_sizes,
				**kwargs
			)
		elif inspect.isclass(f):
			self.f = f(**kwargs)
		else:
			self.f = f
		if isinstance(std, float):
			self.c = torch.tensor(std)
		elif std:
			self.c = nn.Parameter(torch.tensor(0.0))
		else:
			self.c = torch.tensor(0.0)
		if isinstance(scale, float):
			self.a = torch.tensor(scale)
		elif scale:
			self.a = nn.Parameter(torch.tensor(0.0))
		else:
			self.a = torch.tensor(0.0)


	def get_n(
		self,
		x,
		index=None,
		ptr=None,
		dim_size=None,
		dim=-2,
	):
		n = self.reduce(torch.ones_like(x), index, ptr, dim_size, dim, reduce='sum')
		n[n==0] = 1
		return n


	def forward(
		self,
		x,
		index=None,
		ptr=None,
		dim_size=None,
		dim=-2,
	):

		if isinstance(self.a, nn.Parameter):
			self.a.data = self.a.data.clamp(0, 1)
		if isinstance(self.c, nn.Parameter):
			self.c.data = self.c.data.clamp(0, 1)

		if isinstance(self.c, nn.Parameter) or self.c != 0:
			x_mean = self.reduce(x, index, ptr, dim_size, dim, reduce='mean')
			if index is None:
				x = x - self.c * x_mean.unsqueeze(dim)
			else:
				x = x - self.c * torch.index_select(input=x_mean, dim=dim, index=index)

		n = self.get_n(x=x, index=index, ptr=ptr, dim_size=dim_size, dim=dim)

		if self.visualize:
			self.v_std, self.v_mean = x.detach().std(), x.detach().mean()

		x = x.unsqueeze(-1)
		n = n.unsqueeze(-1)
		if dim < 0:
			dim -= 1

		y1 = self.f.forward(x)
		y2 = self.reduce(y1, index, ptr, dim_size, dim, reduce='mean')
		y3 = y2 * (n**self.a)
		z = self.f.reverse(y3)
		z = z.squeeze(-1)

		return z


if __name__ == "__main__":
	in_size = 8
	out_size = 2
	dim = 3
	x = torch.randn(in_size, dim) * 50
	index = torch.randint(low=0, high=out_size, size=(in_size,))
	agg = GenAgg()
	y = agg(x=x, index=index)
	print(x)
	print(y)