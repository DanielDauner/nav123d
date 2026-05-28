<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/nav123d_white.svg" width="500">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo/nav123d_black.svg" width="500">
    <img alt="Logo" src="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg" width="500">
  </picture>
  <h2 align="center">Cross-Dataset Reimplementation of NAVSIM</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2406.15349">Paper</a> | <a href="https://youtu.be/Q4q29fpXnx8">Video</a> | <a href="https://danieldauner.github.io/nav123d/">Documentation</a></h3>
  <h4 align="center"><i>"Because good things come in 123D."</i></h4>
</h1>

> 🏗️ **Under construction.** Expect things to change, break, and improve as we build it out. Comments, suggestions, and contributions are very welcome!

`nav123d` reimplements [NAVSIM](https://github.com/autonomousvision/navsim) — data-driven, non-reactive autonomous-vehicle simulation and benchmarking — on top of the [123D](https://github.com/kesai-labs/py123d) unified data framework, with similar agents, metrics, and training/evaluation pipeline run across datasets. It ships baseline agents (constant-velocity, ego-status MLP, PDM, TransFuser) together with training and evaluation entry points.

## Installation
Editable pip install (Python 3.9 - 3.12):
```sh
pip install -e .
```
... or in a fresh conda environment (Python 3.9 - 3.12)

```sh
conda create -n nav123d python=3.12 -y
conda activate nav123d
pip install uv
uv pip install -e .[dev]
```

> [!WARNING]
> We have not verified that metric values are identical across Python versions. If you intend to compare or report results, make sure all runs use the same Python version.

See the [installation docs](docs/installation.md) for dataset setup and environment variables.

<p align="right">(<a href="#top">back to top</a>)</p>

## Citation

All assets and code in this repository are under the [Apache 2.0 license](./LICENSE) unless specified otherwise.

```BibTeX
@inproceedings{Dauner2024NEURIPS,
	title = {NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking},
	author = {Daniel Dauner and Marcel Hallgarten and Tianyu Li and Xinshuo Weng and Zhiyu Huang and Zetong Yang and Hongyang Li and Igor Gilitschenski and Boris Ivanovic and Marco Pavone and Andreas Geiger and Kashyap Chitta},
	booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
	year = {2024},
}
```

```BibTeX
@inproceedings{Dauner2026ARXIV,
  title={123D: Unifying Multi-Modal Autonomous Driving Data at Scale},
  author={Dauner, Daniel and Charraut, Valentin and Berle, Bastian and Li, Tianyu and Nguyen, Long and Wang, Jiabao and Jing, Changhui and Igl, Maximilian and Caesar, Holger and Ivanovic, Boris and Geiger, Andreas and Chitta, Kashyap},
  journal={arXiv preprint arXiv:2605.08084},
  year={2026}
}
```

<p align="right">(<a href="#top">back to top</a>)</p>
