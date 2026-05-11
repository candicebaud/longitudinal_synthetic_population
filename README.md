# Longitudinal Synthetic Population Framework

This repository contains the code, documentation, examples, and graph-generation scripts for a flexible framework to generate longitudinal synthetic populations.

The framework is designed to help researchers generate synthetic populations based on individual life trajectories. Instead of creating independent cross-sectional populations for each year, the approach generates time-independent life trajectories and maps them to time-dependent population states.

The repository is structured to be reused and extended by other researchers. It includes the source code of the framework, examples showing how to define and sample models, documentation, and notebooks to analyze the generated populations and produce graphs.

## Repository content

The repository contains the following main folders:

```text
longitudinal_synthetic_population/
│
├── framework/      # Source code defining the synthetic population framework
├── examples/       # Examples showing how to use the framework
├── graphs/         # Notebooks and scripts to analyze outputs and produce graphs
└── documentation/  # Documentation of the framework and code
```

The `framework` folder contains the core implementation. The `examples` folder contains examples to help users become familiar with the framework. In particular, the `prior model` example shows how to sample a population from prior models. The `graphs` folder contains example notebooks to analyze results and generate figures.

## Main idea
The framework is based on a trajectory representation of individuals. Each individual is described by a set of life-course dimensions, such as existence, education, employment, or other demographic and behavioral characteristics.
Each dimension is composed of events. Events have durations, start dates, attributes, and temporal constraints. Once individual trajectories are generated, they can be mapped to specific years to obtain cross-sectional synthetic populations.
This makes it possible to generate coherent longitudinal data, where the same individual can be followed consistently over time.

## Features
The framework supports:
- definition of life-course dimensions;
- definition of events within each dimension;
- single-time and multiple-time events;
- event-specific attributes;
- temporal constraints on event start ages and durations;
- linear constraints between durations of different events;
- independent or joint sampling of dimensions;
- prior sampling without observed data;
- population-level generation;
- parallelized population sampling;
- export of generated individuals and populations to dataframes.

The framework includes classes for defining probabilistic models, attributes, events, dimensions, constraints, individuals, and populations.

## Installation
Clone the repository:
```bash
git clone https://github.com/candicebaud/longitudinal_synthetic_population.git
cd longitudinal_synthetic_population
```

The project is implemented in Python. It has been tested with Python 3.11.7 on the SCITAS supercomputer and with Python 3.13.7 on a local machine. Python 3.11.7 or newer is recommended.
Install the main dependencies with:
```bash
pip install numpy pandas pycddlib
```

The main external dependencies are:
```text
numpy
pandas
pycddlib
```

A particular point of attention concerns `pycddlib`. The package provides the `cdd` module, which is used to compute polytope vertices from systems of linear inequalities. The repository supports both the older `pycddlib` API, used on SCITAS with version 2.1.7, and the newer 3.x API. Since `pycddlib` 3.x introduced backward-incompatible changes, users should select the implementation corresponding to their installed version.

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
A detailed code documentation is available in the repository. It describes the installation, notation conventions, and the main classes used to define and sample synthetic populations.

## Citation
If you use this repository in academic work, please cite the associated paper or repository.

```bibtex
@misc{baud_longitudinal_synthetic_population,
  author       = {Baud, Candice},
  title        = {Longitudinal Synthetic Population Framework},
  year         = {2026},
  howpublished = {\url{https://github.com/candicebaud/longitudinal_synthetic_population}}
}
```

Please update this citation once the associated paper or official reference is available.

## Author
Candice Baud  
EPFL


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.














