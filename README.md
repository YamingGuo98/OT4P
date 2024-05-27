# OT4P: Unlocking Effective Orthogonal Group Path for Permutation Relaxation

The repository is a PyTorch implementation of OT4P (Orthogonal Group-based Transformation for Permutation Relaxation), a differentiable transformation for relaxing permutation matrices onto the orthogonal group. We provide a minimal example demonstrating the use of OT4P. The rest of the code from the paper is being prepared.

## Requirements

We use Python version `3.10`, PyTorch version `2.01`, and CUDA version `11.7`. Our method also relies on repository [torch-linear-assignment](https://github.com/ivan-chai/torch-linear-assignment), which provides batch computation of linear assignment problems on GPUs.

## Minimal example

We provide a minimal example (`example.py` or `example.ipynb`) demonstrating the use of OT4P.  Given $X$ and $Y = PXP^{\top}$, where $P$ is a permutation matrix, the objective is to find the true permutation matrix $P$. This problem can be defined as:

$$
\min_P ||PXP^{\top} - Y||^2.
$$

We use OT4P to solve this problem from three different perspectives:

1. Deterministic Optimization;
2. Stochastic Optimization;
3. Constrained Optimization.

## Finding mode connectivity

It is being prepared...

## Inferring neuron identities

It is being prepared...