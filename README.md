# Practical 2: GNNs for 3D Molecular Data — Invariant & Equivariant GNNs

> **Course:** Graph Representation Learning  
> **Focus:** Graph Neural Networks (GNNs) for molecular property prediction with 3D geometric information  
> **Framework:** PyTorch Geometric (PyG)  
> **Dataset:** QM9 (Quantum Mechanics dataset 9)

---

## Overview

This practical explores **Geometric Deep Learning** principles by developing Graph Neural Networks that respect the invariances and symmetries present in 3D molecular data. Starting from a standard Message Passing Neural Network (MPNN) baseline, we progressively build more geometrically principled architectures: the Graph Isomorphism Network (GIN), a naive coordinate-augmented GIN, and finally an **E(3)-invariant MPNN** that leverages 3D structural information in a theoretically sound way.

The central task is **molecular property prediction** — predicting the [electric dipole moment](https://en.wikipedia.org/wiki/Electric_dipole_moment) of drug-like molecules using their graph structure, atom features, and 3D coordinates.

---

## Learning Objectives

- Gain proficiency with **PyTorch Geometric's** `MessagePassing` base class and `Data` objects
- Implement and extend the **Graph Isomorphism Network (GIN)** for graphs with edge features
- Understand how to incorporate **3D geometric information** into GNN message passing
- Learn the importance of **geometric invariance** (rotation & translation invariance under E(3)) for molecular property prediction
- Design **E(3)-invariant GNN layers** using pairwise distances instead of raw coordinates

---

## Repository Structure

```
practical2/
├── Practical_2.ipynb    # Main Jupyter notebook with all tasks and solutions
└── README.md            # This file
```

---

## Requirements

The notebook installs dependencies inline, but the core requirements are:

| Package | Version | Purpose |
|---------|---------|---------|
| `torch` | 2.0.0+cu118 | Deep learning framework |
| `torch-geometric` | 2.3.0 | Graph neural network library |
| `torch_scatter` | 2.1.2 | Sparse aggregation operations |
| `torch_sparse` | 0.6.18 | Sparse tensor operations |
| `torch_cluster` | 1.6.3 | Graph clustering utilities |
| `torch_spline_conv` | 1.2.2 | Spline-based convolutions |
| `rdkit` | latest | Molecular chemistry toolkit |
| `py3Dmol` | latest | 3D molecular visualization |
| `numpy` | <2 | Numerical computing |
| `pandas` | latest | Results tracking |
| `seaborn` | latest | Training curves visualization |
| `scipy` | latest | Orthogonal rotation matrices |

---

## Part 0: Setup & Baseline

### Dataset: QM9

- **130,831** small organic molecules with up to 9 heavy atoms
- **19 regression targets** (we predict target #0: electric dipole moment)
- Each molecule is represented as a fully-connected graph with:
  - **Node features** `x` ∈ ℝⁿˣ¹¹: atom type (one-hot), atomic number, aromaticity, hybridization, number of hydrogens
  - **Edge features** `edge_attr` ∈ ℝᵉˣ⁴: bond type (single, double, triple, aromatic); zero for non-bonded pairs
  - **3D coordinates** `pos` ∈ ℝⁿˣ³: atom positions in 3D space
  - **Target** `y` ∈ ℝ: scalar dipole moment (normalized to mean=0, std=1)

### Data Splits

| Split | Size | Purpose |
|-------|------|---------|
| Train | 1,000 | Model training |
| Validation | 1,000 | Hyperparameter tuning & early stopping |
| Test | 1,000 | Final evaluation |

### Baseline Model: MPNNModel

A 4-layer Message Passing Neural Network with:
- Node feature projection: 11 → 64 dimensions
- `MPNNLayer`: message MLP ψ(hᵢ, hⱼ, eᵢⱼ) → aggregate → update MLP φ(hᵢ, mᵢ)
- Residual connections after each layer
- Global mean pooling + linear prediction head
- Trained with MSE loss, Adam optimizer, ReduceLROnPlateau scheduler

**Baseline Test MAE:** ~0.585

---

## Part 1: Graph Isomorphism Network (30 Points)

### Task 1.1 — Standard GIN Layer (12 pts)

Implements the GIN update rule (Xu et al., 2019):

```
h_u^(t) = MLP( (1 + ε^(t)) · h_u^(t-1) + Σ_{v ∈ N(u)} h_v^(t-1) )
```

Key implementation details:
- Learnable scalar parameter `eps`
- Messages are simply `ReLU(h_j)` — neighbor features passed through activation
- Sum aggregation (to match the Weisfeiler-Lehman test's expressiveness)
- Post-aggregation MLP with BatchNorm and ReLU

### Task 1.2 — GIN with Edge Features: GINEConv (12 pts)

Extends GIN to handle the 4-dimensional edge features in QM9:

```
m_vu = ReLU( h_v + proj(e_vu) )
```

- Adds an `edge_proj` network mapping edge_dim → emb_dim
- Incorporates projected edge features into messages via addition
- Uses the same `(1 + ε) · h + sum(messages)` aggregation pattern

### Task 1.3 — Theoretical Analysis (6 pts)

Discussion questions covering:
1. **Role of ε:** Controls self-contribution weight. When ε=0, the node is treated exactly like its neighbors (closed neighborhood sum). A learnable ε helps distinguish a node from identical-feature neighbors.
2. **Message function:** In `GINConv`, messages ignore destination features and edge attributes: `m_vu = ReLU(h_v)`.
3. **Limitation on QM9:** Standard GIN ignores `edge_attr`, discarding bond-type information. In fully-connected graphs, it cannot distinguish real bonds from zero-padded non-bonds.

**GINModel Test MAE:** ~0.588 (comparable to MPNN baseline)

---

## Part 2: Message Passing with 3D Coordinates (25 Points)

### Task 2.1 — Naive Coordinate GIN: `CoordGINModel` (12.5 pts)

A first attempt at using 3D coordinates by **concatenating** them to node features:

```python
h = self.lin_in(torch.cat([data.x, data.pos], dim=-1))  # (n, 14) → (n, 64)
```

- Input projection adapted from 11 → 14 dimensions (node features + 3D coords)
- Uses `GINEConv` layers for message passing

### Task 2.2 — Sanity Checks (12.5 pts)

Performs permutation invariance/equivariance unit tests:
- `CoordGINModel`: permutation invariant ✅
- `GINEConv`: permutation equivariant ✅

**CoordGINModel Test MAE:** ~0.788 (⚠️ **worse than baseline!**)

> **Key insight:** Simply concatenating raw coordinates hurts performance because coordinates are **not** invariant to 3D rotations and translations. The model must learn this symmetry from limited data, which it fails to do.

---

## Part 3: Invariance to 3D Symmetries (15 Points)

### Core Concept: E(3) Invariance

Molecular properties are **intrinsic** — they do not depend on the coordinate frame. A molecule rotated or translated in space has the same dipole moment. Therefore, GNNs for molecular property prediction should be **invariant** to the Euclidean group E(3):

```
F(H, XQ^T + 1t^T, A) = F(H, X, A)    for all Q ∈ O(3), t ∈ ℝ³
```

### Task 3.1 — Mathematical Formalization (5 pts)

Expressed invariance using matrix notation:
- `H` ∈ ℝⁿˣᵈ: node features
- `X` ∈ ℝⁿˣ³: node coordinates
- `A` ∈ ℝⁿˣⁿ: adjacency matrix
- `F`: GNN layer that must satisfy the invariance condition above

### Task 3.2 — Why Architectural Invariance > Data Augmentation (5 pts)

**Data augmentation** (training with random rotations/translations) is inferior because:
- Increases effective training cost by an SO(3) × ℝ³ factor
- Wastes model capacity memorizing symmetry instead of learning chemistry
- Provides **no guarantee** for unseen orientations at test time

**Architectural invariance** is exact (to floating-point precision), free at inference, and composes cleanly with permutation equivariance.

### Task 3.3 — Unit Test for E(3) Invariance (5 pts)

Implemented `rot_trans_invariance_unit_test` applying random rigid motions:
```python
Q = random_orthogonal_matrix(dim=3)
t = torch.rand(3)
data.pos = data.pos @ Q.T + t
```

**Result:** `CoordGINModel` fails the E(3) invariance test (returns `False`) ✅

---

## Part 4: E(3)-Invariant Message Passing (30 Points)

### Task 4.1 — `InvariantMPNNLayer` (10 pts)

The key insight: **pairwise distances** between atoms are invariant under E(3):

```
|| (Q·x_i + t) - (Q·x_j + t) || = || Q·(x_i - x_j) || = || x_i - x_j ||
```

The `InvariantMPNNLayer` constructs messages using:
1. Source and destination node features `h_i`, `h_j`
2. Edge features `edge_attr`
3. **Pairwise Euclidean distance** `||pos_i - pos_j||` (a scalar)

```python
dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)   # (e, 1)
msg = torch.cat([h_i, h_j, edge_attr, dist], dim=-1)     # (e, 2d + d_e + 1)
return self.mlp_msg(msg)
```

**Design choices:**
- Uses MPNN-style message construction (not GIN) — more natural for geometric features
- Passes `pos` through `propagate()` to enable `pos_i`, `pos_j` indexing
- Distance is a **scalar** — rotationally and translationally invariant by construction

The accompanying `InvariantMPNNModel` stacks 4 such layers with residual connections.

### Task 4.2 — Proof of E(3) Invariance (10 pts)

**Update equations:**
```
d_uv = ||x_u - x_v||₂
m_uv = ψ(h_u, h_v, e_uv, d_uv)
m_u = ⊕_{v ∈ N(u)} m_uv
h_u^(t) = φ(h_u^(t-1), m_u)
```

**Proof sketch:**
1. Under any rigid motion, `d_uv` is unchanged (shown via Q^T·Q = I)
2. Since ψ's arguments are unchanged, `m_uv` is unchanged
3. Aggregation and update are deterministic functions of unchanged quantities
4. By induction, all layer outputs are invariant
5. Global mean pooling and linear head preserve invariance

### Task 4.3 — Sanity Checks (5 pts)

| Property | `InvariantMPNNLayer` | `InvariantMPNNModel` |
|----------|---------------------|----------------------|
| E(3) Invariant | ✅ True | ✅ True |
| Permutation Equivariant (layer) | ✅ | — |
| Permutation Invariant (model) | — | ✅ |

### Task 4.4 — Training & Evaluation (5 pts)

**InvariantMPNNModel Test MAE:** ~0.399

---

## Results Summary

### Model Comparison

| Model | Best Val MAE | Test MAE | Uses Coords? | Geometrically Principled? | Parameters |
|-------|-------------|----------|-------------|---------------------------|------------|
| `MPNNModel` | 0.721 | **0.585** | ❌ No | — | 103,233 |
| `GINModel` | 0.717 | **0.588** | ❌ No | — | 69,957 |
| `CoordGINModel` | 0.895 | **0.788** | ✅ Yes (naive concat) | ❌ No | 70,149 |
| `InvariantMPNNModel` | **0.437** | **0.399** | ✅ Yes (pairwise distances) | ✅ E(3) invariant | 103,489 |

### Key Takeaways

1. **Naive coordinate usage hurts:** `CoordGINModel` is ~35% worse than the baseline that ignores coordinates entirely. Raw coordinates introduce frame-dependent noise that the model cannot disentangle from limited data.

2. **Geometric invariance is crucial:** By encoding E(3) invariance directly into the architecture via pairwise distances, `InvariantMPNNModel` achieves:
   - **32% improvement** over `MPNNModel`
   - **49% improvement** over `CoordGINModel`
   - Consistently lower validation error throughout training

3. **Symmetry-aware architectures generalize better:** Architectural invariance provides exact guarantees, eliminates the need for data augmentation, and frees model capacity to learn the actual chemistry rather than coordinate frame artifacts.

---

## Architecture Details

### InvariantMPNNLayer

```
Input:  h ∈ ℝⁿˣᵈ, pos ∈ ℝⁿˣ³, edge_index, edge_attr ∈ ℝᵉˣᵈᵉ

Message construction (per edge):
  dist_ij = ||pos_i - pos_j||₂                    # scalar, E(3)-invariant
  m_ij = MLP_msg([h_i, h_j, edge_attr, dist_ij])  # ℝ²ᵈ⁺ᵈᵉ⁺¹ → ℝᵈ

Aggregation:
  m_i = Σ_{j ∈ N(i)} m_ij                         # sum aggregation

Update:
  h_i' = MLP_upd([h_i, m_i])                      # ℝ²ᵈ → ℝᵈ

Output: updated node features h' ∈ ℝⁿˣᵈ
```

### Full InvariantMPNNModel Pipeline

```
data.x (n, 11) ──→ Linear ──→ h⁰ (n, 64)
                            ↓
         ┌────────────────────────────────────────────┐
         │  InvariantMPNNLayer × 4 (with residuals)  │
         │  each: h, pos, edge_index, edge_attr       │
         └────────────────────────────────────────────┘
                            ↓
                    h^L (n, 64)
                            ↓
                global_mean_pool ──→ h_G (batch_size, 64)
                            ↓
                  Linear ──→ ŷ (batch_size, 1)
```

---

## Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Epochs | 100 |
| Batch size | 32 |
| Hidden dimension | 64 |
| Number of layers | 4 |
| Optimizer | Adam |
| Initial learning rate | 0.01 |
| LR scheduler | ReduceLROnPlateau (factor=0.95, patience=5, min_lr=1e-5) |
| Loss | MSE Loss |
| Evaluation metric | Mean Absolute Error (MAE) |
| Device | CUDA if available, else CPU |
| Random seed | 0 (for reproducibility) |

---

## References

- **GIN:** Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). *How Powerful are Graph Neural Networks?* ICLR 2019.
- **QM9 Dataset:** Ramakrishnan, R., Dral, P. O., Rupp, M., & von Lilienfeld, O. A. (2014). *Quantum chemistry structures and properties of 134 kilo molecules.* Scientific Data.
- **PyTorch Geometric:** Fey, M., & Lenssen, J. E. (2019). *Fast Graph Representation Learning with PyTorch Geometric.*
- **Geometric Deep Learning:** Bronstein, M. M., Bruna, J., Cohen, T., & Veličković, P. (2021). *Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges.*

---

## Acknowledgments

Based on a notebook originally created by:
- **Chaitanya K. Joshi**
- **Charlie Harris**
- **Ramon Viñas Torné**

---

## License

This practical is for educational purposes as part of a Geometric Deep Learning course.
