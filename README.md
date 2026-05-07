# VBA: Vector Bundle Attention for Intrinsically Geometric Representation

Shenglei Fang, Xianfang Sun, You Zhou

This is code for paper "VBA: Vector Bundle Attention for Intrinsically Geometric Representation" in ICML 2026.

## Abstract

Learning from geometrically structured data is central to applications in biology, physics, and computer vision. In many tasks, meaningful comparisons depend on how features are aligned in space. Graph Neural Networks capture local structure but are constrained by message passing. Transformers model long-range dependencies but largely ignore geometry. We introduce the Vector Bundle Attention Transformer (VBA-Transformer), a framework that redefines attention as an intrinsic geometric operator. Each token couples a base manifold coordinate with a fiber feature vector, following vector bundle theory. A principled parallel transport mechanism aligns fiber features across local coordinate systems before similarity is computed. This embeds geometry directly into the attention operator. Unlike prior methods that inject geometry as an external bias or positional encoding, VBA integrates geometry natively inside attention. On challenging single-cell RNA sequencing benchmarks, VBA achieves state-of-the-art accuracy, outperforming Transformer baselines by over 3--5\%. On spatial transcriptomics, it demonstrates superior clustering performance. On 3D point clouds, it achieves competitive accuracy, validating broad generalization across domains. Beyond empirical gains, we provide theoretical analysis of invariance and perturbation stability. We also demonstrate robust transport behavior empirically. Together, these results establish intrinsic geometric alignment as a powerful principle for scalable representation learning.

## Requirements
