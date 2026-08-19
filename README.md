# Gaussian-KAN

Core implementation for the paper:

**[Making Gaussian Kolmogorov–Arnold networks reliable and accurate](https://doi.org/10.1016/j.neucom.2026.134771)**  
Amir Noorizadegan, Sifan Wang and Leevan Ling


This repository contains the main Gaussian KAN model and the first-layer conditioning diagnostic used in the paper.

The full experimental pipeline is not included. The purpose of this repository is to provide the essential model code needed to reproduce the Gaussian basis construction, fixed-center architecture, and first-layer conditioning analysis.

## Paper

The Gaussian scale parameter \(\epsilon\) strongly affects the behavior of Gaussian KANs. In this work, we study how \(\epsilon\) influences first-layer feature geometry, conditioning, and approximation accuracy.

The main practical finding is that, for a Gaussian KAN with $G$ uniformly distributed centers, the interval

$$
\epsilon \in
\left[
\frac{1}{G-1},
\frac{2}{G-1}
\right]
$$

provides a stable and effective operating range for the standard shared-center Gaussian KAN considered in the paper.

This range is used as a practical design rule, not as a universal optimality theorem.

## Repository contents

```text
Gaussian-KAN/
│
├── model_Gaussian.py      # core Gaussian KAN model
├── diagnostics.py         # first-layer conditioning diagnostic
└── README.md
```

## Citation

If you use this code, please cite:

```bibtex
@article{Amir2026GaussianKAN,
  title   = {Scaling of Gaussian Kolmogorov--Arnold Networks},
  author  = {Noorizadegan, Amir and Wang, Sifan},
  journal = {arXiv preprint arXiv:2604.21174},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.21174}
}
```
