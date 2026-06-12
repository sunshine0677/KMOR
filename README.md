# Replication Package for Paper: KMOR

This repository contains the code, datasets, and experimental scripts for our paper **"KMOR: Finding Top-k Multi-constraint Optimal Routes over On-demand Transportation Networks"**.

The package is designed to support the reproducibility of all experimental results reported in the paper, including parameter sensitivity analysis, effectiveness evaluation, ablation studies, and multi-route diversity validation.

---

## Environment

* Python 3.11
* Required packages: `matplotlib`, `numpy`

Install dependencies:

```bash
pip install matplotlib numpy
```

---

## Project Structure

```text
.
├── Dpkmor.py                      # Core algorithm library: graph structures, request model, RideSharePlanner
├── parameter_sensitivity.py       # Experiment 1: parameter sensitivity analysis
├── paper_experiment_suite.py      # Experiment 2: effectiveness and ablation studies
├── kmor_experiments.py            # Experimental utilities
├── kmor_effectiveness.py          # Baseline comparison module
├── prepare_paper_datasets.py      # Dataset and request generation
├── paper_datasets.json            # Dataset configuration for effectiveness experiments
├── paper_ablation_datasets.json   # Dataset configuration for ablation experiments
├── OL.cedge.txt                   # Oldenburg road network
├── TG.cedge.txt                   # San Joaquin road network
├── CD.edges.txt                   # Chengdu road network
├── NY.edges.txt                   # New York City road network
├── requests_ol.txt                # OL requests for effectiveness experiments
├── requests_tg.txt                # TG requests for effectiveness experiments
├── requests_ny.txt                # NY requests for effectiveness experiments
├── requests_cd.txt                # CD requests for effectiveness experiments
├── requests_ol_ablation.txt       # OL requests for ablation experiments
├── requests_tg_ablation.txt       # TG requests for ablation experiments
├── requests_ny_ablation.txt       # NY requests for ablation experiments
├── requests_cd_ablation.txt       # CD requests for ablation experiments
├── paper_results/                 # Output directory (generated automatically)
└── README.md
```

---

## Datasets

To facilitate reproducibility, all road network datasets used in the paper are included in this repository.

| Dataset                | Description                        | Source                     | File           |
| ---------------------- | ---------------------------------- | -------------------------- | -------------- |
| **OL (Oldenburg)**     | Road network of Oldenburg City     | Spatial Dataset Repository | `OL.cedge.txt` |
| **TG (San Joaquin)**   | Road network of San Joaquin County | Spatial Dataset Repository | `TG.cedge.txt` |
| **CD (Chengdu)**       | Road network of Chengdu City       | OpenStreetMap (via osmnx)  | `CD.edges.txt` |
| **NY (New York City)** | Road network of New York City      | OpenStreetMap (via osmnx)  | `NY.edges.zip` |

### Dataset Format

Each road network dataset is represented as an edge list.

* `*.cedge.txt` / `*.edges.txt`

  * Contains directed road segments.
  * Each record specifies a source node, destination node, and edge weight.
  * Edge weights correspond to travel distance along the road network.

The datasets are used to construct the on-demand transportation networks evaluated in the paper.

---

## Experimental Platform

All experiments were conducted on:

* CPU: Intel Core i7-12700H
* Memory: 16 GB RAM
* Operating System: Windows 11
* Python: 3.11

Each experiment was repeated 10 times, and the reported results correspond to the average values.

---

## Experimental Overview

The experimental evaluation consists of four parts. All scripts can be executed independently or as part of the complete evaluation pipeline.

### 1. Parameter Sensitivity Analysis

**Objective**

Evaluate the sensitivity of DPKMOR to:

* Request pool size (|R|)
* Number of returned routes (k)
* Vehicle capacity (Kv)
* Delivery deadline (er)

**Script**

```bash
parameter_sensitivity.py
```

**Commands**

```bash
# Run all parameter sensitivity experiments
python parameter_sensitivity.py --workdir . --only run

# Generate plots from existing results
python parameter_sensitivity.py --workdir . --only plot

# Complete workflow
python parameter_sensitivity.py --workdir . --only all
```

**Outputs**

* `paper_results/parameter_sensitivity_results.csv`
* `paper_results/parameter_sensitivity_summary.csv`
* Sensitivity figures in PDF format

---

### 2. Effectiveness Evaluation

**Objective**

Compare DPKMOR against three insertion-based baselines:

* Greedy
* Nearest-Insertion
* Best-Insertion Solomon

**Script**

```bash
paper_experiment_suite.py
```

**Command**

```bash
python paper_experiment_suite.py --workdir . --only effectiveness
```

**Outputs**

* `paper_results/effectiveness_results.csv`
* `paper_results/effectiveness_cost_per_served.pdf`

---

### 3. Ablation Study

**Objective**

Analyze the contribution of the two core components:

* Dominance-based pruning
* Dynamic programming acceleration

The following variants are compared:

* Full DPKMOR
* NoPruning
* NoDP

**Command**

```bash
python paper_experiment_suite.py --workdir . --only ablation
```

**Outputs**

* `paper_results/ablation_results.csv`
* `paper_results/ablation_search_time_s.pdf`
* `paper_results/ablation_examined_partial_routes.pdf`

---

### 4. Multi-route Diversity Validation

**Objective**

Compare route sets generated with:

* k = 1
* k = 5

to demonstrate the practical value of returning multiple non-dominated routes.

**Outputs**

* Diversity statistics stored in `ablation_results.csv`
* Diversity table reported in the paper

---

## Running All Experiments

### Step 1: Generate Requests and Configurations

```bash
python prepare_paper_datasets.py --workdir .
```

### Step 2: Run All Experiments

```bash
python parameter_sensitivity.py --workdir . --only run

python paper_experiment_suite.py --workdir . --only all
```

### Step 3 (Optional): Regenerate Figures

```bash
python parameter_sensitivity.py --workdir . --only plot

python paper_experiment_suite.py --workdir . --only plot
```

---

## Core Algorithms

### Kprun-search: Dominance-Based Pruning

**Purpose**

Reduce the exponential growth of the search space by pruning dominated partial routes.

**Key Features**

* Dominance-based route pruning
* Multi-constraint feasibility checking
* Completeness-preserving search reduction
* Efficient management of candidate route expansions

---

### DPKMOR: Dynamic Programming Acceleration

**Purpose**

Accelerate route construction using dynamic programming, reducing insertion complexity from O(n³) to O(n²).

**Key Features**

* Unified handling of:

  * Time-window constraints
  * Capacity constraints
  * Detour constraints
* Reuse of intermediate feasibility states
* Significant reduction of redundant computations
* Built upon the Kprun-search framework

---

## Experimental Parameters

The parameter sensitivity experiments vary one parameter at a time while keeping the others fixed at their default values.

| Parameter | Tested Values |
|------------|------------|
| Request Pool Size (|R|) | 1000, 2000, 3000, 4000, 5000 |
| Number of Returned Routes (k) | 10, 20, 30, 40, 50 |
| Delivery Deadline er (min) | 5, 10, 15, 20, 25 |
| Vehicle Capacity Kv | 3, 4, 6, 10, 20 |

Default settings:

- |R| = 3000
- k = 30
- er = 10 min
- Kv = 4
---

## Evaluation Metrics

The following metrics are used throughout the experiments:

* **Search Time (s)**
  Total query processing time.

* **Examined Partial Routes**
  Number of intermediate route states explored during search.

* **Service Cost**
  Total travel cost of the returned route set.

* **Served Requests**
  Number of successfully served transportation requests.

* **Cost per Served Request**
  Average travel cost per served request.

These metrics are used to evaluate efficiency, effectiveness, scalability, and route diversity.

---

## Results

All experimental figures are automatically generated in PDF format and stored under:

```text
paper_results/
```

Detailed analysis and discussion of the results can be found in **Section VI** of the paper.

---

## Data Sources

* **OL (Oldenburg)** and **TG (San Joaquin)** are obtained from the Spatial Dataset Repository.
* **CD (Chengdu)** and **NY (New York City)** are extracted from OpenStreetMap using the `osmnx` library.
* Edge weights are computed from actual road lengths.
* Request sets are automatically generated by `prepare_paper_datasets.py` using shortest-path references on the corresponding road networks.

---
