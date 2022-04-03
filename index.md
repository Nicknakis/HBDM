# Hierarchical Block Distance Model (HBDM)

Python 3.8.3 and Pytorch 1.9.0 implementation of the Hierarchical Block Distance Model (HBDM).

## Description

We propose a novel graph representation learning method named the Hierarchical Block Distance Model (HBDM). It extracts node embeddings imposing a multiscale block-structure that accounts for homophily and transitivity properties across the levels of the inferred hierarchy. Notably, the HBDM naturally accommodates unipartite, directed, and bipartite networks whereas the hierarchy is designed to ensure linearithmic time and space complexity enabling the analysis of very large-scale networks. We evaluate the performance of our approach on massive networks consisting of millions of nodes. Our HBDM framework significantly outperforming recent scalable approaches in downstream tasks providing superior performance even using ultra low-dimensional embeddings at the same time facilitating direct and hierarchical-aware network visualization and interpretation.

### Code
An implementation of the project in Python and Pytorch can be reached at our [Github repository](https://github.com/Nicknakis/HBDM).

### Installation
BLA BLA

### Learning hierarchical and multi-scale graph representations with HBDM
BLA BLA

### References
N. Nakis, A. Celikkanat, S. Lehmann and M. Mørup, [A Hierarchical Block Distance Model for Ultra Low-Dimensional Graph Representations](https://openreview.net/pdf?id=U-GB_gONqbo), Under Review.
