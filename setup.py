from setuptools import setup
from setuptools import find_packages

setup(
    name="genagg",
    version="2.0.2",
    packages=find_packages(),
    install_requires=["torch", "torch_geometric", "wandb", "scipy"],
    author="Ryan Kortvelesy",
    description="A learnable, generalised aggregation module for pytorch that parametrises the space of aggregation functions.",
    readme = "README.md",
)
