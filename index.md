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

### A Millon Node Unipartite Example with [Flixster](http://konect.cc/networks/flixster/) Network
| <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/flixster-1.png?raw=true"   />   |
|:---:|
| Dendrogram - Binary Logarithm over the Sum of Euclidean Distances |

| <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/flixster_l=0-min.png?"  alt="drawing"  width="150"  />   | <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/flixster_l=2-min.png?raw=true"  alt="drawing"  width="150" />  | <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/flixster_l=4-min.png?raw=true"  alt="drawing"  width="150"  />  | <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/flixster_l=7-min.png?raw=true"  alt="drawing"  width="150"  />  | <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/flixsterscatter_re.png?raw=true"  alt="drawing"  width="250"  />  |
|:---:|:---:|:---:|:---:|:---:|
| L=1 | L=3| L=5 | L=8 | 2-D Embedding Space |


### A Bipartite Example with [GitHub](http://konect.cc/networks/github/) Network 
| <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/github-1.png?raw=true"   />   |
|:---:|
| Dendrogram - Binary Logarithm over the Sum of Euclidean Distances |

| <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/github_l=0.png?raw=true"  alt="drawing"  width="220"  />   | <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/github_l=2.png?raw=true"  alt="drawing"  width="220"  />  | <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/github_l=5.png?raw=true"  alt="drawing"  width="220"  />  |  <img src="https://github.com/Nicknakis/HBDM/blob/gh-pages/docs/assets/githubscatter_re.png?raw=true"  alt="drawing"  width="250" />  |
|:---:|:---:|:---:|:---:|
| L=1 | L=3| L=6 | 2-D Embedding Space |


### References
N. Nakis, A. Celikkanat, S. Lehmann and M. Mørup, [A Hierarchical Block Distance Model for Ultra Low-Dimensional Graph Representations](https://openreview.net/pdf?id=U-GB_gONqbo), Under Review.
