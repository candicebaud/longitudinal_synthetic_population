# Longitudinal Synthetic Population Framework

This repository contains the code, documentation, examples, and graph-generation scripts for a flexible framework to generate longitudinal synthetic populations.

The framework is designed to help researchers generate synthetic populations based on individual life trajectories. Instead of creating independent cross-sectional populations for each year, the approach generates time-independent life trajectories and maps them to time-dependent population states. Trajectories can be sampled purely from prior specifications (data-free), or updated against one or several observed datasets using a Bayesian data-integration scheme.

The repository is structured to be reused and extended by other researchers. It includes the source code of the framework, examples showing how to define and sample models from priors and from posteriors, documentation, and notebooks to analyze the generated populations and produce graphs.

This framework is presented in more detail in the following papers:

- Baud, C., & Bierlaire, M. (2026). *A flexible and realistic synthetic panel population generation.* In Proceedings of the 2026 IEEE 29th International Conference on Intelligent Transportation Systems (ITSC). (In press)
- Baud, C., & Bierlaire, M. (2026). *From Priors to Data: A Flexible Bayesian Framework for Panel Synthetic Population Generation.* In Proceedings of the 2026 hEART Conference. (In press)

## Repository content

The repository contains the following main folders:


```text
longitudinal_synthetic_population/
│
├── framework/                # Source code defining the synthetic population framework (priors + posterior updating)
├── examples/                 # Examples showing how to use the framework
│   ├── prior_models/         # Sampling populations from priors only
│   └── posterior_models/     # Sampling populations from priors updated with observed data
├── graphs/                   # Notebooks and scripts to analyze outputs and produce graphs
│   ├── prior_graphs/
│   └── posterior_graphs/
└── documentation.pdf         # Full documentation of the framework and code
```

The `framework` folder contains the core implementation. The `examples` folder contains examples to help users become familiar with both the prior-only and the data-integrated (posterior) workflows. The `graphs` folder contains example notebooks to analyze results and generate figures.

## Main idea

The framework is based on a trajectory representation of individuals. Each individual is described by a set of life-course dimensions, such as existence, education, employment, or other demographic and behavioral characteristics. Each dimension is composed of events. Events have durations, start dates, attributes, and temporal constraints. Once individual trajectories are generated, they can be mapped to specific years to obtain cross-sectional synthetic populations, making it possible to generate coherent longitudinal data where the same individual can be followed consistently over time.

The framework supports two complementary sampling regimes:

- **Sampling from priors**, where trajectories are generated purely from a specified probabilistic model, without any observed data.
- **Sampling from the posterior**, where a prior model is combined with one or several observed datasets through a Bayesian data-integration scheme (Metropolis-Hastings updating of the population), so that the resulting synthetic population is consistent with the data.

## Features

The framework supports:

- definition of life-course dimensions and of the events within each dimension;
- single-time and multiple-time events;
- event-specific attributes;
- temporal constraints on event start ages and durations;
- linear constraints (equality or inequality) between durations of different events, possibly across dimensions;
- independent, joint, or automatic sampling of dimensions;
- prior sampling without observed data, with convergence diagnostics across multiple chains;
- Bayesian updating of a prior population against one or several observed datasets (`BirthKernel` / `BirthPopUpdater` for the Existence dimension, `DimKernel` / `JointDimPopUpdater` for the other dimensions), including support for weighted observations;
- sampling of only the individuals concerned by the observed data, or of individuals from the prior for the rest of the population;
- checkpointed sampling for long-running jobs (e.g., on HPC), so simulations can resume without restarting from scratch;
- population-level generation with parallelized sampling;
- export of generated individuals and populations to dataframes.

The framework includes classes for defining probabilistic models, attributes, events, dimensions, constraints, individuals, and populations, as well as the storage classes used to hold sampled results.

## Installation

Clone the repository:

```bash
git clone https://github.com/candicebaud/longitudinal_synthetic_population.git
cd longitudinal_synthetic_population
```

The project is implemented in Python. It has been tested with Python 3.11.7 on the [SCITAS](https://scitas-doc.epfl.ch/) supercomputer and with Python 3.13.7 on a local machine. Python 3.11.7 or newer is recommended.

Install the main dependencies with:

```bash
pip install numpy pandas pycddlib
```

The main external dependencies are:
```text
numpy
pandas
pycddlib
scipy
matplotlib
```

**A particular point of attention concerns `pycddlib`.** The package provides the `cdd` module, used to compute polytope vertices from systems of linear inequalities. Two versions of the package are supported:

- the older API (version 2.1.7), used on SCITAS;
- the newer 3.x API, used on local machines.

Since `pycddlib` 3.x introduced backward-incompatible changes in the construction of matrices and polyhedra, two implementations of the affected functions are provided. The `framework` folder targets `pycddlib` 3.x; if you are running an older `pycddlib` (e.g. 2.1.7, as on SCITAS), use the corresponding files in the `hpc_friendly` folder instead.

### Python path

The source code of the framework is located in the `framework/` directory.  
Before running the examples, add this directory to your Python path.

From the root of the repository, run:

```bash
export PYTHONPATH="$PWD/framework:$PYTHONPATH"
cd examples/prior_models
python sample_priors.py
```

## Code organization
The code files follow a naming convention based on their first word:
```text
classes_*      Storage classes
constraints_*  Constraint definitions
spec_*         Model specification classes
har_*          Hit-and-run algorithms
model_*        Model definitions
sample_*       Files used to sample a model
run_*          Run files, especially for HPC usage
```
This convention is used to make the framework easier to navigate and extend.

## Documentation

A detailed code documentation is available as `documentation.pdf` in the repository. It describes the installation, notation conventions, the classes used to define and sample from priors (`AttributeSpec`, `EventSpec`, `DimensionSpec`, `IndividualSpec`, `PopulationSpec`, `ExistenceDimensionSpec`, etc.), the classes used to update a population against observed data (`BirthKernel`, `BirthPopUpdater`, `DimKernel`, `JointDimPopUpdater`), and the storage classes (`Attribute`, `Event`, `Trajectory`, `Individual`, `Population`).

## Examples

The `examples` folder contains two sets of examples:

- `prior_models/`: sampling a population from priors only (`model_priors`, `sample_priors`, `run_priors`).
- `posterior_models/`: sampling a population updated against observed data, split into an Existence update (`sample_existence_update_2010_2015_sampling`, `run_existence_update_2010_2015_sampling`) and an update of the other life-trajectory dimensions (`sample_alive_full_2010_2015_sampling`, `run_full_alive_2010_2015_sampling`), using two synthetic example datasets (`2010_sample`, `2015_sample`) that mimic the structure of the confidential data used in the papers without being extracts of it.

The `graphs` folder mirrors this structure with `prior_graphs/` and `posterior_graphs/` notebooks to reproduce the figures from the papers, either on the provided example dataframes or on your own generated populations.

## Citation

If you use this repository in academic work, please cite the first release of the repository and, when relevant, the associated publications.

### Software repository

```bibtex
@software{baud_longitudinal_synthetic_population_v1_0_0,
  author  = {Baud, Candice},
  title   = {{Longitudinal Synthetic Population Framework}},
  year    = {2026},
  version = {v1.0.0},
  url     = {https://github.com/candicebaud/longitudinal_synthetic_population/releases/tag/v1.0.0}
}
```

If the release is archived on Zenodo, please cite the Zenodo DOI associated with the first release instead.

### Associated publications

```bibtex
@inproceedings{baud_bierlaire_2026_flexible_realistic,
  author    = {Baud, Candice and Bierlaire, Michel},
  title     = {{A Flexible and Realistic Synthetic Panel Population Generation}},
  booktitle = {Proceedings of the 2026 IEEE 29th International Conference on Intelligent Transportation Systems},
  year      = {2026},
  note      = {In press}
}

@inproceedings{baud_bierlaire_2026_priors_to_data,
  author    = {Baud, Candice and Bierlaire, Michel},
  title     = {{From Priors to Data: A Flexible Bayesian Framework for Panel Synthetic Population Generation}},
  booktitle = {Proceedings of the 2026 hEART Conference},
  year      = {2026},
  note      = {In press}
}
```


## Author
Candice Baud  
EPFL


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.














