# VBA: Vector Bundle Attention for Intrinsically Geometric Representation

Shenglei Fang, Xianfang Sun, You Zhou

This is code for paper "VBA: Vector Bundle Attention for Intrinsically Geometric Representation" in ICML 2026.

## Abstract

Learning from geometrically structured data is central to applications in biology, physics, and computer vision. In many tasks, meaningful comparisons depend on how features are aligned in space. Graph Neural Networks capture local structure but are constrained by message passing. Transformers model long-range dependencies but largely ignore geometry. We introduce the Vector Bundle Attention Transformer (VBA-Transformer), a framework that redefines attention as an intrinsic geometric operator. Each token couples a base manifold coordinate with a fiber feature vector, following vector bundle theory. A principled parallel transport mechanism aligns fiber features across local coordinate systems before similarity is computed. This embeds geometry directly into the attention operator. Unlike prior methods that inject geometry as an external bias or positional encoding, VBA integrates geometry natively inside attention. On challenging single-cell RNA sequencing benchmarks, VBA achieves state-of-the-art accuracy, outperforming Transformer baselines by over 3--5\%. On spatial transcriptomics, it demonstrates superior clustering performance. On 3D point clouds, it achieves competitive accuracy, validating broad generalization across domains. Beyond empirical gains, we provide theoretical analysis of invariance and perturbation stability. We also demonstrate robust transport behavior empirically. Together, these results establish intrinsic geometric alignment as a powerful principle for scalable representation learning.


<p align="center">
  <img src="images/VBA%20(7)%20(1)_00.png" width="48%" />
</p>

<p align="center">
  <em>Conceptual illustration of the attention mechanism in VBA.</em>
</p>

## Installation

Create a Python environment:

```bash

conda create -n vba python=3.10
conda activate vba

```

Install PyTorch:

```bash

pip install torch torchvision torchaudio

```

## Quick Start

Run the smoke test:

```bash

python vba.py

```

Example with explicit arguments:

```bash

python vba.py \
  --batch 2 \
  --tokens 32 \
  --dim 128 \
  --depth 2 \
  --heads 4 \
  --base_dim 2 \
  --fiber_dim 16 \
  --bundles 4 \
  --k 8 \
  --steps 5

```

## Minimal Usage

```bash

import torch
from vba import VBAEncoder

model = VBAEncoder(
    dim=128,
    depth=2,
    heads=4,
    base_dim=2,
    fiber_dim=16,
    num_bundles=4,
    dropout=0.1,
    k_neighbors=8,
)

x = torch.randn(2, 32, 128)
y = model(x)

print(y.shape)

```


## Citation

