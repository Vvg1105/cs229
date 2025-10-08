
from setuptools import setup, find_packages

setup(
    name="cs229-ps1",
    version="0.1.0",
    description="CS229 Problem Set 1",
    packages=["src.linearclass", "src.imbalanced", "src.poisson"],
    python_requires=">=3.9",
)