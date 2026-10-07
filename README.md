# PMAR - Particle-based Modelling and Analysis for MARine Decision Support

PMAR is open source and is programmed in Python. Particle trajectories are modelled using [OpenDrift](https://opendrift.github.io/). 

# Installation

To install PMAR, you can use either `conda` or `pip`. Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/index.html#) or [pip](https://pip.pypa.io/en/stable/) installed on your system.

## Using conda:
```
conda create -n pmar-env python=3.9
conda activate pmar-env
conda install -c conda-forge opendrift rioxarray geocube
pip install git+https://github.com/CNR-ISMAR/pmar.git
```

## Using pip only (in an existing environment):
```
pip install opendrift rioxarray geocube
pip install git+https://github.com/CNR-ISMAR/pmar.git
```
