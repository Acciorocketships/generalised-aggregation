from setuptools import setup
from setuptools import find_packages

setup(
    name="genagg",
    version="2.0.0",
    packages=find_packages(),
    install_requires=["torch", "torch_geometric", "wandb", "scipy"],
    author="Ryan Kortvelesy",
    description="A Learnable, Generalised Aggregation Module",
)
