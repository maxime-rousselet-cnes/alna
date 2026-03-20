"""
To install via pip install -e . in the root of the repository. This will make the ALNA package
available in the current environment.
"""

from setuptools import find_packages, setup

setup(
    name="alna",
    packages=find_packages(),
    version="0.0.1",
    description="Anelastic Love Number Algorithm",
    author="Maxime Rousselet",
)
