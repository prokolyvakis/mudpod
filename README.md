# mud-pod: A Multivariate Unimodality Test

![Build Status](https://github.com/prokolyvakis/mudpod/actions/workflows/test_workflow.yml/badge.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16614-b31b1b.svg)](https://arxiv.org/abs/2311.16614)

This package offers tools to analyze the unimodality of data sampled from multivariate distributions lying in the Euclidean Space. To read an independent explanation and summary of this paper, please refer to the write-up by AIModels.fyi [here](https://www.aimodels.fyi/papers/arxiv/multivariate-unimodality-test-harnessing-dip-statistic-mahalanobis).

## Features

- The `mud-pod` test: A multivariate unimodality test.
- The `dip-means` clustering algorithm: A wrapper of `k-means` that also detects the numbers of clusters.

## Installation

To install `mudpod`, you can use [`pdm`](https://pdm-project.org/latest/), which is a modern packaging tool that manages your Python packages without the need for creating a virtualenv in a traditional sense.

### Prerequisites

Ensure you have `pdm` installed on your system. If not, install it using the following command:

```bash
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
```

### Project Setup

Please run:

```bash
pdm install -G core
```

**Note**: If you want to run the tests or the experiments, please install the additional dependencies, i.e., `test` and `exps`, respectively, using the following command:

```bash
pdm install -G GROUP_NAME
```

## References

If you find this code useful in your research, please cite:

```
@misc{kolyvakis2023multivariate,
      title={A Multivariate Unimodality Test Harnenssing the Dip Statistic of Mahalanobis Distances Over Random Projections}, 
      author={Prodromos Kolyvakis and Aristidis Likas},
      year={2023},
      eprint={2311.16614},
      archivePrefix={arXiv},
      primaryClass={stat.ME}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
