# Graph Neural Network (GNN) from Scratch

## Project Overview
This project implements the core mathematical components of a Graph Neural Network (GNN) and Graph Attention Mechanism using pure **NumPy**. 

The goal of this project is to demonstrate an understanding of the underlying linear algebra and matrix operations that power modern GNN frameworks (like PyTorch Geometric) by building them from the ground up.

## Features
* **Graph Representation:** Handling Adjacency Matrices ($A$) and Node Feature Matrices ($X$).
* **Walk Calculation:** Algorithmically computing graph walks using matrix powers ($A^k$).
* **GNN Forward Pass:** A custom implementation of a 3-layer GNN using Matrix-Vector multiplication for neighbor aggregation.
* **Graph Attention:** A "Self-Attention" mechanism applied to graph structures, including masking and softmax normalization.

## Technical Details
The implementation avoids high-level GNN libraries to focus on the math:
* **Language:** Python 3
* **Libraries:** NumPy (Computation), NetworkX & Matplotlib (Visualization)

## Usage
To run the simulation with a randomly generated graph:

```bash
python gnn_implementation.py
