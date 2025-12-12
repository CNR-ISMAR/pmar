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
conda install -c conda-forge opendrift rioxarray geocube
pip install git+https://github.com/CNR-ISMAR/pmar.git
```

## Using pip only (in an existing environment):
```
pip install opendrift rioxarray geocube
pip install git+https://github.com/CNR-ISMAR/pmar.git
```

# Citation
If you use PMAR in your research, please cite the following paper: 

> Bosi, S., Raffaetà, A., Simeoni, M., Bobchev, N., Berov, D., Barbanti, A., & Menegon, S. (2025). PMAR: A Lagrangian approach to the modelling of anthropogenic pressures for marine management. Environmental Modelling & Software, 106822. doi: https://doi.org/10.1016/j.envsoft.2025.106822

BibTeX:
```
@article{bosi2025pmar,
  title={PMAR: A Lagrangian approach to the modelling of anthropogenic pressures for marine management},
  author={Bosi, Sofia and Raffaet{\`a}, Alessandra and Simeoni, Marta and Bobchev, Nikola and Berov, Dimitar and Barbanti, Andrea and Menegon, Stefano},
  journal={Environmental Modelling \& Software},
  pages={106822},
  year={2025},
  publisher={Elsevier},
  doi={https://doi.org/10.1016/j.envsoft.2025.106822}
}
```
