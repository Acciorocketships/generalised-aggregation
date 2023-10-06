from setuptools import setup
from setuptools import find_packages

with open("README.md") as file:
    long_desc = file.read()

long_desc = "# GenAgg\n" + "[Github](https://github.com/Acciorocketships/generalised-aggregation)\n" + long_desc[10:]

setup(
    name="genagg",
    version="2.0.3",
    packages=find_packages(),
    install_requires=["torch", "torch_geometric", "wandb", "scipy"],
    author="Ryan Kortvelesy",
    description="A learnable, generalised aggregation module for pytorch that parametrises the space of aggregation functions.",
    long_description=long_desc,
    long_description_content_type='text/markdown',
    readme = "README.md",
)