# mud-pod: A Multivariate Unimodality Test

![Build Status](https://github.com/prokolyvakis/mudpod/actions/workflows/test_workflow.yml/badge.svg)
[![arXiv](https://img.shields.io/badge/arXiv-2311.16614-b31b1b.svg)](https://arxiv.org/abs/2311.16614)
[![UAI 2025](https://img.shields.io/badge/Accepted-UAI_2025-brightgreen)](https://openreview.net/pdf?id=f3KdHTMabv)

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
@InProceedings{pmlr-v286-kolyvakis25a,
  title = 	 {A Multivariate Unimodality Test Harnessing the Dip Statistic of Mahalanobis Distances Over Random Projections},
  author =       {Kolyvakis, Prodromos and Likas, Aristidis},
  booktitle = 	 {Proceedings of the Forty-first Conference on Uncertainty in Artificial Intelligence},
  pages = 	 {2255--2268},
  year = 	 {2025},
  editor = 	 {Chiappa, Silvia and Magliacane, Sara},
  volume = 	 {286},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--25 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://raw.githubusercontent.com/mlresearch/v286/main/assets/kolyvakis25a/kolyvakis25a.pdf},
  url = 	 {https://proceedings.mlr.press/v286/kolyvakis25a.html},
  abstract = 	 {Unimodality, pivotal in statistical analysis, offers insights into dataset structures and drives sophisticated analytical procedures. While unimodality’s confirmation is straightforward for one-dimensional data using methods like Silverman’s approach and Hartigans’ dip statistic, its generalization to higher dimensions remains challenging. By extrapolating one-dimensional unimodality principles to multi-dimensional spaces through linear random projections and leveraging point-to-point distancing, our method, rooted in $\alpha$-unimodality assumptions, presents a novel multivariate unimodality test named $\textit{mud-pod}$. Both theoretical and empirical studies confirm the efficacy of our method in unimodality assessment of multidimensional datasets as well as in estimating the number of clusters.}
}
```

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.
