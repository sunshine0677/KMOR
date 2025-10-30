# Replication Package for Paper: KMOR
This repository contains the datasets used in our paper **"KMOR: Finding Top-k Multi-constraint Optimal Routes over On-demand Transportation Networks"** to ensure the reproducibility of our work, as some original data links have become inaccessible.
## 📁 Datasets
| Dataset | Description | Original Source | Files in This Repository |
| :--- | :--- | :--- | :--- |
| **CD**<br>(Chengdu) | Road network of Chengdu City | [GAIA Initiative](https://outreach.didichuxing.com/research/opendata/)<br>*[Link inactive, last accessed: 2021-10-09]* ||
| **TG**<br>(San Joaquin) | Road network of San Joaquin County | [Spatial Dataset Repository](https://users.cs.utah.edu/~lifeifei/SpatialDataset.htm)<br> | `TG.cedge.txt` |
| **OL**<br>(Oldenburg) | Road network of Oldenburg City | [Spatial Dataset Repository](https://users.cs.utah.edu/~lifeifei/SpatialDataset.htm)<br> | `OL.cedge.txt` |
| **NY**<br>(New York) | Road network of New York City | [Spatial Dataset Repository](http://www.cs.utah.edu/~lifeifei/SpatialDataset.html)<br>*[Link inactive, last accessed: 2021-10-09]* ||

## 🔧 Usage

The datasets are provided in their original format. Each road network dataset typically consists of a file:
- A **`*edge`** file containing edge information with start and end nodes.

## 🚀 Algorithms

### K-PURN Search (`kpurn_search`)
- **Purpose**: Finds promising route networks for multi-constraint routing
- **Key Features**:
  - Efficient pruning of suboptimal routes
  - Multi-constraint optimization
  - Scalable network exploration

### DP-KMOR (`dpkmor`) 
- **Purpose**: Dynamic programming approach for Top-k Multi-constraint Optimal Routes
- **Key Features**:
  - Exact algorithm for optimal route discovery
  - Handles multiple constraints simultaneously
  - Returns top-k diverse optimal routes


