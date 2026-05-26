<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo/nav123d_white.svg" width="500">
    <source media="(prefers-color-scheme: light)" srcset="assets/logo/nav123d_black.svg" width="500">
    <img alt="Logo" src="https://kesai.eu/py123d/_static/123D_logo_transparent_black.svg" width="500">
  </picture>
  <h2 align="center">Cross-Dataset Reimplementation of NAVSIM</h2>
  <h2 align="center">"Because good things come in three." - Daniel</h2>
  <h3 align="center"><a href="https://arxiv.org/abs/2605.08084">Paper</a> | <a href="https://youtu.be/Q4q29fpXnx8">Video</a> | <a href="https://kesai.eu/py123d/">Documentation</a></h3>
</h1>

</div>


The main branch contains the code for NAVSIM v2, used in the 2025 NAVSIM challenge. <b style='color:red;'>For NAVSIM v1, as well as its `navtest` leaderboard, which are also part of this repository, please check the [v1.1 branch](https://github.com/autonomousvision/navsim/tree/v1.1).</b>

<br/>

> [**NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking**](https://arxiv.org/abs/2406.15349)
>
> [Daniel Dauner](https://danieldauner.github.io/)<sup>1,2</sup>, [Marcel Hallgarten](https://mh0797.github.io/)<sup>1,5</sup>, [Tianyu Li](https://github.com/sephyli)<sup>3</sup>, [Xinshuo Weng](https://xinshuoweng.com/)<sup>4</sup>, [Zhiyu Huang](https://mczhi.github.io/)<sup>4,6</sup>, [Zetong Yang](https://scholar.google.com/citations?user=oPiZSVYAAAAJ)<sup>3</sup>,\
> [Hongyang Li](https://lihongyang.info/)<sup>3</sup>, [Igor Gilitschenski](https://www.gilitschenski.org/igor/)<sup>7,8</sup>, [Boris Ivanovic](https://www.borisivanovic.com/)<sup>4</sup>, [Marco Pavone](https://web.stanford.edu/~pavone/)<sup>4,9</sup>, [Andreas Geiger](https://www.cvlibs.net/)<sup>1,2</sup>, and [Kashyap Chitta](https://kashyap7x.github.io/)<sup>1,2</sup>  <br>
>
> <sup>1</sup>University of Tübingen, <sup>2</sup>Tübingen AI Center, <sup>3</sup>OpenDriveLab at Shanghai AI Lab, <sup>4</sup>NVIDIA Research\
> <sup>5</sup>Robert Bosch GmbH, <sup>6</sup>Nanyang Technological University, <sup>7</sup>University of Toronto, <sup>8</sup>Vector Institute, <sup>9</sup>Stanford University
>
> Advances in Neural Information Processing Systems (NeurIPS), 2024 \
> Track on Datasets and Benchmarks
<br/>


![image](https://github.com/user-attachments/assets/5966e9ca-5fcb-4cad-a686-eb8ef2bf943e)

## Table of Contents

1. [Highlights](#highlight)
2. [Getting started](#gettingstarted)
3. [Changelog](#changelog)
4. [License and citation](#licenseandcitation)
5. [Other resources](#otherresources)

## Getting started <a name="gettingstarted"></a>

- [Download and installation](docs/install.md)
- [Understanding and creating agents](docs/agents.md)
- [Understanding the data format and classes](docs/cache.md)
- [Dataset splits vs. filtered training / test splits](docs/splits.md)
- [Understanding the Extended PDM Score](docs/metrics.md)
- [Understanding the traffic simulation](docs/traffic_agents.md)
- [Submitting to the Leaderboard](docs/submission.md)

<p align="right">(<a href="#top">back to top</a>)</p>


## Citation <a name="licenseandcitation"></a>

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

## Other resources <a name="otherresources"></a>

- [SLEDGE](https://github.com/autonomousvision/sledge) | [tuPlan garage](https://github.com/autonomousvision/tuplan_garage) | [CARLA garage](https://github.com/autonomousvision/carla_garage) | [Survey on E2EAD](https://github.com/OpenDriveLab/End-to-end-Autonomous-Driving)
- [PlanT](https://github.com/autonomousvision/plant) | [KING](https://github.com/autonomousvision/king) | [TransFuser](https://github.com/autonomousvision/transfuser) | [NEAT](https://github.com/autonomousvision/neat)

<p align="right">(<a href="#top">back to top</a>)</p>
