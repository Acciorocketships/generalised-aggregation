# GenAgg

## Installation
1. Install the requirements
```bash
pip install torch
pip install torch_scatter
pip install torch_geometric
pip install pandas
pip install gym
pip install wandb
```
2. Install the package
```bash
cd generalised-aggregation
pip install -e .
```

## Basic Usage
GenAgg uses the same sparse format as torch_scatter, allowing batched computation of aggregations with variable input sizes. To perform a normal, non-batched aggregation, set out_size to 1, and every element in index to 0.
```python
import torch
from genagg import GenAgg
in_size = 8 # number of input nodes
out_size = 2 # number of output nodes
dim = 3 # dimension of each node
x = torch.randn(in_size, dim) # state for each input element
index = torch.randint(low=0, high=out_size, size=(in_size,)) # the output node for each input node to aggregate into
agg = GenAgg() # initialise the aggregator
y = agg(x=x, index=index) # compute the output
```

## Experiments
Aggregator Regression:
```bash
python run/regress_agg.py
```
GNN Regression:
```bash
python run/regress_gnn.py
```
GNN Benchmarks:
```bash
python run/exp_genagg.py
python run/exp_aggr.py
python run/exp_poweragg.py
rpython un/exp_pna.py
```
