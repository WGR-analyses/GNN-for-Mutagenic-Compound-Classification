# GNN for Mutagenic Compound Classification

This project implements custom Graph Neural Network (GNN) layers in PyTorch to classify molecules as mutagenic (y=1) or non‑mutagenic (y=0). It is centered on a single Jupyter notebook: GNN-for-mutagenic-compound.ipynb.

## Overview

- Task: graph-level classification of molecules using atom (node) and bond (edge) features.
- Challenge: each molecule has a variable number of nodes; the model must be permutation-invariant and size-invariant.
- Approach: stack one or more custom convolution layers, apply global pooling (mean or max) to obtain a fixed-size graph embedding, then predict with an MLP head.
- Architectures implemented:
  - GraphConv (neighbor aggregation)
  - GraphSAGEConv (concatenate node and aggregated neighbors)
  - AttentionConv (GAT-style learned attention over neighbors)

## Repository Contents

- GNN-for-mutagenic-compound.ipynb
  - Data loading and exploration (per-graph records)
  - Custom layers: GraphConv, GraphSAGEConv, AttentionConv
  - Global pooling and MLP classifier
  - Train/validation split and evaluation scaffold
- Homework_2_Grossrieder_wanchai.pdf
  - Written report describing design rationale, two strategies for incorporating edge features, experimental setup, and validation results.

## Data Format

Each graph (molecule) is represented with the following fields (as seen in the notebook’s exploratory cells):

- node_feat: [num_nodes, in_node_features]
- edge_index: edge list (or adjacency available as edge_adj)
- edge_attr: [num_edges, in_edge_features] (not used in node-only baseline)
- edge_adj: [num_nodes, num_nodes] dense adjacency
- y: graph label (1 for mutagenic, 0 for non‑mutagenic)
- num_nodes, num_edges: metadata

## Methods

### Model Structure

- Convolutional block:
  - One of GraphConv, GraphSAGEConv, or AttentionConv updates node embeddings using adjacency-defined neighborhoods.
  - Aggregation choices:
    - GraphConv, GraphSAGEConv: mean or max
    - AttentionConv: sum (default), mean, or max
- Global pooling:
  - Node embeddings are reduced to a fixed-size graph embedding via global max or mean pooling.
- Classifier:
  - A small MLP maps the pooled graph embedding to logits for binary classification.

This design ensures invariance to node ordering and robustness to differing graph sizes.

### Custom Convolution Layers

- GraphConv:
  - Aggregates neighbor features (mean or max) and applies a linear transform + activation.
- GraphSAGEConv:
  - Aggregates neighbor features, concatenates with the node’s own features, then applies a linear transform + activation.
- AttentionConv:
  - Learns attention coefficients over neighbors using a small feedforward scoring function on concatenated node–neighbor pairs, normalizes with softmax, and aggregates messages (default sum). Supports activation on the aggregated output.

## Incorporating Edge Features (Two Strategies)

The project explores two approaches to use edge attributes beyond node-only baselines, as detailed in the PDF:

1) Concatenate edge-derived features to node features:
   - For each node, derive or aggregate relevant edge features and append them to the node feature vector.
   - Pros: minimal changes to the convolution, pooling, and training loop; treats augmented features as node inputs.

2) “Edge-as-node” reformulation:
   - Construct an edge–edge adjacency where two edges are connected if they share a node.
   - Convolve edge features analogously to node features and pool edge embeddings.
   - Combine pooled node and pooled edge embeddings before the MLP.
   - Pros: preserves edge-specific structure via a dedicated adjacency; requires modifying the GNN module and training loop to handle both node and edge streams.

## Experimental Setup and Results

- Configuration:
  - Single convolution layer + single-layer MLP.
  - Global max pooling.
  - Aggregation:
    - GraphConv/GraphSAGEConv: max aggregation
    - AttentionConv: sum aggregation
- Learning rates tested: 1e-4, 5e-4, 1e-3
- Observations:
  - Reported validation accuracies across methods and LR settings were generally low or inconsistent.
  - Examples cited include accuracies such as 65.5%, 72.4%, and 79.3% under different configurations, with several settings collapsing to ~34.4% or even 0% for AttentionConv at 1e-3.
  - Results appeared unstable and not clearly aligned with expectations (e.g., attention not outperforming basic baselines). The report notes possible implementation issues and the simplicity of the tested models as contributing factors.

Interpretation:
- The minimal depth (one conv + one MLP layer), aggregation choices, and potential coding issues likely limited performance and stability.
- A thorough hyperparameter sweep and multi-layer variants were deemed out of scope for the current time constraints.

## How to Run

1) Set up environment
- Python 3.8+
- pip install:
  - torch torchvision torchaudio
  - numpy pandas scikit-learn matplotlib

2) Launch notebook
- jupyter notebook GNN-for-mutagenic-compound.ipynb
- Execute cells in order, ensuring data structures are correctly loaded and converted to tensors.

3) Training
- Choose layer type (GraphConv/GraphSAGEConv/AttentionConv), aggregation, and pooling.
- Set hyperparameters (hidden size, learning rate, epochs, batch handling if added).
- Run training and validation cells, monitor accuracy and loss.

## Extending the Project

- Depth: stack multiple convolution layers and a deeper MLP.
- Edge utilization:
  - Implement and compare both edge strategies thoroughly.
  - Integrate edge_attr directly into message passing.
- Efficiency and scalability:
  - Vectorize operations or adopt PyTorch Geometric/DGL for batched multi-graph training.
- Evaluation:
  - Add metrics (AUC, F1), cross-validation, early stopping, and LR schedulers.
- Reproducibility:
  - Set random seeds, log runs, and pin dependency versions in requirements.txt.

## Troubleshooting

- Device mismatch: ensure tensors and modules are on the same device (CPU/GPU).
- Shape errors: confirm node_feat matches in_features and adj is [N, N].
- Isolated nodes: verify graceful handling in all layers.
- Unstable training: lower learning rate, add gradient clipping, and verify attention score computations.
