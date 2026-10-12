<<<<<<< HEAD
# PMAR - Particle-based Modelling and Analysis for MARine Decision Support

PMAR is open source and is programmed in Python. Particle trajectories are modelled using [OpenDrift](https://opendrift.github.io/). 
=======
# PMAR - Particle-based modelling and analysis for MARine decision support

**PMAR** is a software package for the analysis of tracer dispersal from human activities at sea in support of Maritime Spatial Planning.

PMAR is open source and is programmed in Python. Particle trajectories are simulated using [OpenDrift](https://opendrift.github.io/). 
>>>>>>> 6de6a02c6ea807412f434555595622ce9b2cd72e

# Installation

To install PMAR, you can use either `conda` or `pip`. Make sure you have [Conda](https://docs.conda.io/projects/conda/en/latest/index.html#) or [pip](https://pip.pypa.io/en/stable/) installed on your system.

## Using conda:
```
conda create -n pmar-env python=3.11
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
