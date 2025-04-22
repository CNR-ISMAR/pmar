# PMAR - Pressure models for MARine activities

**PMAR** is a software package for the modelling of anthropogenic **pressures** caused by **human activities** at sea. 

**Pressures** are substances which act as tracers and cause negative impacts when released in the marine environment. This may include marine litter, oil or other hazardous substances, chemicals, organic matter, etc.

**Human activities** such as maritime transport, fishing, oil & gas extraction, aquaculture, and more, can cause the release of such substances, thus becoming pressure **sources**. 

PMAR is open source and is programmed in Python. Particle trajectories are modelled using [OpenDrift](https://opendrift.github.io/). 

![Alt text](/images/pmar_intro.png)

# Installation

To install PMAR, you can use either `conda` or `pip`. Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/index.html#) or [pip](https://pip.pypa.io/en/stable/) installed on your system.

## Using conda:
```
conda create -n pmar-env python=3.9
conda activate pmar-env
conda install -c conda-forge opendrift rioxarray
pip install git+https://github.com/CNR-ISMAR/pmar.git
```

## Using pip only (in an existing environment):
```
pip install opendrift rioxarray
pip install git+https://github.com/CNR-ISMAR/pmar.git
```
