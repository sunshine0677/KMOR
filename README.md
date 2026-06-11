```markdown
# Replication Package for Paper: KMOR

This repository contains the code, datasets, and experimental scripts for our paper **"KMOR: Finding Top-k Multi-constraint Optimal Routes over On-demand Transportation Networks"**.

## Environment

- Python 3.11
- Required packages: `matplotlib`, `numpy`
- Install: `pip install matplotlib numpy`

## 📁 Datasets

| Dataset | Description | Source | Files in This Repository |
| :--- | :--- | :--- | :--- |
| **OL** (Oldenburg) | Road network of Oldenburg City | [Spatial Dataset Repository](https://users.cs.utah.edu/~lifeifei/SpatialDataset.htm) | `OL.cedge.txt` |
| **TG** (San Joaquin) | Road network of San Joaquin County | [Spatial Dataset Repository](https://users.cs.utah.edu/~lifeifei/SpatialDataset.htm) | `TG.cedge.txt` |
| **CD** (Chengdu) | Road network of Chengdu City | OpenStreetMap via osmnx | `cd_edges.txt` |
| **NY** (New York City) | Road network of New York City (all five boroughs) | OpenStreetMap via osmnx | `ny_edges.txt` |

### Dataset Statistics

| Dataset | Nodes | Edges |
|---------|-------|-------|
| OL | 6,105 | 7,035 |
| TG | 18,263 | 23,874 |
| CD | 159,438 | 397,482 |
| NY | 448,229 | 1,370,396 |

### File Formats

Edge files (space-separated):
```
edge_id start_node end_node weight
```

Request files (space-separated):
```
name pickup_node dropoff_node demand request_time wait_limit
```

## 🚀 Algorithms

### Kprun-search: Dominance-Based Pruning

Finds top-k multi-constraint optimal routes using dominance-based pruning.

- **Purpose**: Efficiently prunes redundant partial routes during graph exploration
- **Key Features**:
  - Dominance relation for eliminating equivalent or inferior partial routes
  - Multi-constraint feasibility checking (time windows, capacity, detour limits)
  - Guarantees completeness while reducing the search space

### DPKMOR: Dynamic Programming Acceleration

Accelerates route construction by replacing the basic insertion operation with a dynamic programming-based mechanism.

- **Purpose**: Efficient exact algorithm for top-k multi-constraint route discovery
- **Key Features**:
  - Unifies time-window, capacity, and detour constraint checking into a single DP pass
  - Reuses intermediate feasibility states to avoid redundant computation
  - Reduces insertion complexity from O(n³) to O(n²)
  - Builds on Kprun-search's pruning framework

## 🔧 Experiments

### 1. Comparison with Baselines

Compares DPKMOR against three insertion-based baselines (Greedy, Nearest-Insertion, Best-Insertion Solomon).

```bash
python paper_experiment_suite.py --workdir . --only effectiveness
```

Output:
- `paper_results/effectiveness_results.csv`
- `paper_results/effectiveness_cost_per_served.pdf`

### 2. Value of Multi-Route Retrieval

Compares k=1 versus k=5 to demonstrate the benefit of returning multiple non-dominated routes.

```bash
python paper_experiment_suite.py --workdir . --only ablation
```

Output:
- `paper_results/ablation_results.csv`
- `paper_results/ablation_search_time_s.pdf`
- `paper_results/ablation_examined_partial_routes.pdf`

### 3. Parameter Sensitivity Analysis

Evaluates sensitivity to request pool size (|R|), route count (k), vehicle capacity (Kv), and delivery deadline (er).

```bash
python parameter_sensitivity.py --workdir . --only run
python parameter_sensitivity.py --workdir . --only plot
```

Output:
- `paper_results/parameter_sensitivity_results.csv`
- `paper_results/parameter_sensitivity_summary.csv`
- `paper_results/sensitivity_R_search_time.pdf`
- `paper_results/sensitivity_R_examined_partial_routes.pdf`
- `paper_results/sensitivity_k_search_time.pdf`
- `paper_results/sensitivity_k_examined_partial_routes.pdf`
- `paper_results/sensitivity_deadline_search_time.pdf`
- `paper_results/sensitivity_deadline_examined_partial_routes.pdf`
- `paper_results/sensitivity_capacity_search_time.pdf`
- `paper_results/sensitivity_capacity_examined_partial_routes.pdf`

### 4. Ablation Study

Isolates the contributions of dominance pruning and dynamic programming.

```bash
python paper_experiment_suite.py --workdir . --only ablation
```

Output:
- `paper_results/ablation_results.csv`
- `paper_results/ablation_search_time_s.pdf`
- `paper_results/ablation_examined_partial_routes.pdf`

### Run All Experiments

```bash
python parameter_sensitivity.py --workdir . --only run
python paper_experiment_suite.py --workdir . --only all
```

### Regenerate Figures Only

```bash
python parameter_sensitivity.py --workdir . --only plot
python paper_experiment_suite.py --workdir . --only plot
```

## ⚙️ Key Parameters

| Parameter | Default | Range |
|-----------|---------|-------|
| |R| (request pool size) | 3000 | 1000–5000 |
| k (routes) | 30 | 10–50 |
| er (deadline) | 10 min | 5–25 min |
| Kv (capacity) | 4 | 3–20 |
| Detour ratio | 1.4 | — |
| Max states | 100,000 | — |
| Trials | 10 | — |

## 📊 Results

All figures are saved as PDF files in `paper_results/`. Detailed analysis is provided in Section VI of the paper.
```
