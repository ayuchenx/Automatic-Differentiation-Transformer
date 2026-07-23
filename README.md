# Automatic-Differentiation-Transformer

UCSD CSE234 (Winter 2025) — Programming Assignment 1

A from-scratch automatic differentiation (autodiff) engine built on a computational graph, used to construct and train a Transformer-style sequence classifier on MNIST — plus a set of fused GPU-style kernels (matmul+layernorm, matmul+softmax) built on top of it.

## Overview

This project has three parts:

1. **Autodiff engine** (`auto_diff.py`) — a reverse-mode automatic differentiation library. Computation is expressed as a graph of `Node`s connected by `Op`s; the engine supports forward evaluation and backpropagation (`gradients`) via topological sort over the graph.
2. **Fused operators** (`fused_ops.py`) — custom ops that fuse two operations into a single kernel (e.g. matmul + layernorm, matmul + softmax) to reduce intermediate memory reads/writes, along with a short write-up of the design (`part3.txt`).
3. **Transformer model** (`transformer.py`) — a single-layer Transformer (self-attention + feed-forward) built entirely out of `auto_diff` ops, trained with plain SGD to classify MNIST digits (each image treated as a sequence of rows).

## Repository structure

```
.
├── auto_diff.py             # Core autodiff engine: Node, Op, graph ops, topological_sort, Evaluator, gradients
├── fused_ops.py             # Fused ops: MatMulLayerNormOp, MatMulSoftmaxOp
├── transformer.py           # Transformer forward pass, softmax loss, SGD training loop on MNIST
├── part3.txt                # Write-up on the intuition/design behind the fused operators
├── CSE234_PA1.ipynb         # Notebook version / driver for the assignment
├── tests/
│   ├── test_auto_diff_node_forward.py
│   ├── test_auto_diff_node_backward.py
│   ├── test_auto_diff_graph_forward.py
│   ├── test_auto_diff_graph_backward.py
│   ├── test_fused_ops.py
│   └── test_fused_ops_perf.py
└── README.md
```

## Core components

### `auto_diff.py`
- `Node` / `Variable` — nodes in the computational graph.
- `Op` and its subclasses — the supported operations, including `AddOp`, `MulOp`, `SubOp`, `DivOp`, `MatMulOp`, `TransposeOp`, `BroadcastOp`, `ExpandAsOp` (2D/3D), `SumOp`, `MeanOp`, `SoftmaxOp`, `LayerNormOp`, `ReLUOp`, `LogOp`, `SqrtOp`, `PowerOp`, `GreaterThanOp`, `ZerosLikeOp`/`OnesLikeOp`, and constant-operand variants (`AddByConstOp`, `MulByConstOp`, `DivByConstOp`).
- `topological_sort(nodes)` — orders graph nodes for evaluation.
- `Evaluator` — runs the forward pass over a graph given input values.
- `gradients(output_node, nodes)` — computes gradients of `output_node` with respect to a list of `nodes` via reverse-mode autodiff.

### `fused_ops.py`
- `MatMulLayerNormOp` — fuses matrix multiplication with layer normalization into one op.
- `MatMulSoftmaxOp` — fuses matrix multiplication with softmax into one op.

These reduce the need to materialize intermediate results between matmul and the following normalization/activation step (see `part3.txt` for the reasoning and possible future optimizations, such as automatic operator fusion).

### `transformer.py`
- `transformer(X, nodes, model_dim, seq_length, eps, batch_size, num_classes)` — builds the computational graph for a single Transformer layer (query/key/value/output projections `Wq, Wk, Wv, Wo`, feed-forward weights `W1, W2, b1, b2`) followed by sequence classification.
- `softmax_loss(Z, y_one_hot, batch_size)` — softmax cross-entropy loss node.
- `sgd_epoch(...)` — runs one epoch of mini-batch SGD over the training data.
- `train_model()` — end-to-end training script: loads MNIST (each 28×28 image treated as a length-28 sequence of 28-dim rows), builds the forward/backward graph, and trains with SGD.

## Requirements

- Python 3
- `torch`
- `torchvision`
- `numpy`
- `scikit-learn`

Install with:

```bash
pip install torch torchvision numpy scikit-learn
```

## Usage
Run JupyterNotebook for test cases. 

Train the Transformer classifier on MNIST:

```python
python transformer.py
```

Note that the accuracy isn't very high, due to it being a single layer and training on 20 epochs. 

## Notes

This was built as coursework for UCSD CSE234 (Winter 2025), Programming Assignment 1, and is intended for educational purposes — implementing autodiff, backpropagation, and operator fusion from first principles rather than relying on an existing deep learning framework's autograd.
