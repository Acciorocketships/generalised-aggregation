from setuptools import setup
from setuptools import find_packages

setup(
    name="genagg",
    version="0.0.1",
    packages=find_packages(),
    install_requires=["torch", "torch_scatter", "torch_geometric", "pandas", "gym", "wandb"],
    author="Anonymous",
    description="A Learnable, Generalised Aggregation Module",
)
