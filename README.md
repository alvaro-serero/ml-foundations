# ML Foundations

A hands-on portfolio of machine learning fundamentals by [Alvaro Serero](https://github.com/alvaro-serero).

This repository collects focused Python exercises that build core ML ideas from first principles before applying the same ideas with NumPy and PyTorch. Each file is intentionally small and self-contained: it implements one concept, typically includes a compact example, and can be read without navigating a larger application.

## Current focus

Currently focusing on PyTorch, neural networks, optimization, and deep learning fundamentals.

## What is covered

| Area | Exercises | Examples |
| --- | ---: | --- |
| [Calculus](calculus/) | 3 | Polynomial derivatives, partial derivatives, chain rule |
| [Linear algebra](linear_algebra/) | 20 | Matrix operations, norms, eigenvalues, basis transformations |
| [Probability and statistics](statistics/) | 4 | Descriptive statistics, covariance, empirical PMFs, dice statistics |
| [Machine learning](machine_learning/) | 21 | Regression, metrics, preprocessing, batching, gradient descent |
| [Deep learning](deep_learning/) | 10 | Activations, backpropagation, convolution, KL divergence |
| [PyTorch](pytorch/) | 24 | Autograd, layers, training steps, normalization, convolution |

## Selected implementations

- [Linear regression with gradient descent](machine_learning/linear_regression_using_gradient_descent.py) — fits model weights using an explicit optimization loop.
- [Gradient descent variants](machine_learning/gradient_descent_variants_with_mse_loss.py) — compares batch, stochastic, and mini-batch updates.
- [Single-neuron backpropagation](deep_learning/single_neuron_backpropagation.py) — implements forward and backward passes with NumPy.
- [2D convolution from scratch](deep_learning/simple_convolutional_2d_layer.py) — applies padding and stride without a deep-learning framework.
- [Convolution via `unfold`](pytorch/conv2d_implementation_with_unfold.py) — reconstructs a PyTorch convolution from tensor operations.
- [Layer normalization from scratch](pytorch/layer_norm_from_scratch.py) — mirrors the behavior of `torch.nn.LayerNorm`.

## Running the exercises

### With `uv`

```bash
git clone https://github.com/alvaro-serero/ml-foundations.git
cd ml-foundations
uv sync
uv run python deep_learning/single_neuron_backpropagation.py
```

### With `pip`

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install numpy torch
python deep_learning/single_neuron_backpropagation.py
```

Python 3.12 or newer is required. Most exercises use only the standard library or NumPy; files in [`pytorch/`](pytorch/) require PyTorch.

## Running the tests

```bash
uv run pytest
```

The focused test suite checks selected from-scratch implementations against known outputs and PyTorch reference behavior.

## Repository conventions

- One concept per file, named for the exercise it demonstrates.
- Implementations favor clarity and explicit math over abstraction.
- Many files include small executable examples, so they can be run directly.
- Alternative implementations are retained when they illustrate a different API or learning step.

This is a learning portfolio rather than a production library. The exercises are preserved as individual implementations so the progression from mathematical foundations to framework-based models remains visible.

Many exercises are based on problems completed on DeepML and are implemented here as part of my independent study.
