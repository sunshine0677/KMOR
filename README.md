# KMOR: Top-k Multi-Constraint Route Planning over On-Demand Transportation Networks

This repository provides the source code, experimental configurations, and available replication materials for the manuscript **“Top-k Multi-Constraint Route Planning over On-Demand Transportation Networks”**, submitted to **IEEE ICDE 2027**.

> **Release status:** The code and a synthetic smoke-test instance are available, together with selected archived experimental summaries. The original fixed query banks and per-seed dynamic event traces are not included in this release. Road-network and request files have been staged locally, but redistribution permissions must be verified before public upload. Consequently, this repository **does not yet support exact replay of every experiment in the submitted manuscript**.

## 1. Overview

In on-demand transportation networks, a vehicle may serve several pickup–delivery requests subject to capacity, pickup-deadline, passenger-detour, and route-completion constraints. **KMOR** retrieves up to `k` distinct, non-dominated feasible vehicle routes, considering service coverage and route cost. Different concrete routes can have equal objective values while serving different requests or following different pickup–delivery orders.

The proposed algorithms are:

- **KPrun-Search:** Exact Top-k search with constraint-aware feasibility checks and safe coverage–cost pruning.
- **DPKMOR:** Resource-aware state sharing that avoids redundant expansions while retaining distinct histories for concrete-route reconstruction.
- **EI-DPKMOR:** Event-driven incremental maintenance for request arrivals, request cancellations, and road travel-time updates, with completeness certification and exact recovery when needed.

The package also contains Basic Search, an Exhaustive Search reference for small instances, Full-DPKMOR recomputation, and **adapted** KineticTree-TopK, TD-TopK-Insertion, and RHLS-TopK baselines. These adaptations are not unmodified releases of the original methods.

## 2. Environment

The submitted experiments report the following environment:

| Component | Configuration |
|---|---|
| Operating system | 64-bit Windows |
| Python | 3.11.7 |
| Processor | Intel Core i5-1135G7 |
| Memory | 16 GB RAM |
| Execution | Sequential workloads; single-threaded search |

Install the Python dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` pins direct dependencies but is not a complete lockfile for every transitive package. Runtime measurements can vary with hardware and background system load.

## 3. Quick Start

The included synthetic graph and request file allow a basic run without downloading an external road network.

**Run the lightweight tests:**

```bash
python -B -m unittest test_static_correctness_paths test_prepare_dimacs_datasets
```

**Compute Top-k routes on the synthetic example:**

```bash
python -B Dpkmor.py --input edges_demo.txt --ignore-first-col --source 0 --target 99 --requests-file requests_with_time_demo.txt --param k=2 capacity=4 time_budget=500 detour_ratio=1.4 max_states=1000
```

During package preparation, the three lightweight tests passed, and the example returned two routes. This is a **smoke test**, not a reproduction of a submitted paper experiment.

## 4. Road Networks and Data Availability

The experiments use eight road networks. The figures below are the statistics reported in the submitted manuscript (Table II).

| Dataset | Network | Nodes | Edges / arcs |
|---|---|---:|---:|
| OL | Oldenburg | 6,105 | 7,035 |
| TG | San Joaquin | 18,263 | 23,874 |
| CD | Chengdu | 159,438 | 397,482 |
| BAY | San Francisco Bay Area | 321,270 | 800,172 |
| COL | Colorado | 435,666 | 1,057,066 |
| NY | New York City | 448,229 | 1,370,396 |
| FLA | Florida | 1,070,376 | 2,712,798 |
| NW | Northwest USA | 1,207,945 | 2,840,208 |

The manuscript attributes **OL** and **TG** to the Spatial Dataset Repository; **CD** and **NY** to OpenStreetMap data processed with OSMnx; and **BAY**, **COL**, **FLA**, and **NW** to the [Ninth DIMACS Implementation Challenge USA road-network datasets](https://www.diag.uniroma1.it/challenge9/data/USA-road-d). DIMACS `USA-road-d` weights represent distances, not measured traffic travel times. The corresponding preprocessing and travel-time configuration are therefore needed to reproduce the routing workloads.

Dataset configuration is recorded in `paper_datasets_8.json`, and `prepare_dimacs_datasets.py` provides the local DIMACS preprocessing utility.

### Road files and checksums

The local data preparation package uses the following filenames. **Their appearance below does not mean that they are included in the public release.** Consult `DATA_FILES.json` for expected relative paths and hashes, if that manifest is provided.

| Dataset | Local road-file name |
|---|---|
| OL | `OL.cedge.txt` |
| TG | `TG.cedge.txt` |
| CD | `CD.edges.txt` |
| NY | `NY.edges.txt.gz` |
| BAY | `BAY.edges.txt` |
| COL | `COL.edges.txt.gz` |
| FLA | `FLA.edges.txt.gz` |
| NW | `NW.edges.txt.gz` |

Some local road files were compressed for packaging. **If these files are lawfully available in your local copy**, the helper can verify the staged inputs and restore the compressed edge files:

```bash
python -B restore_road_data.py --verify-only
python -B restore_road_data.py
```

Do not publicly upload third-party datasets solely because they exist in the local release directory. Source provenance and redistribution permissions must be checked for each dataset. `DATA_FILES.json` and `UPLOAD_FILES.json` provide file metadata and checksums; neither constitutes a license or establishes source provenance.

**Not included in the current replication release:** the fixed static query banks, some per-query reference results, and the original per-seed dynamic request/event traces. The portable manifests in `replication_manifests/` describe their configurations and locations but **do not regenerate the historical inputs**.

## 5. Algorithm Implementations

| Method | Role | Main file |
|---|---|---|
| Basic Search | Static search reference | `Dpkmor.py` |
| Exhaustive Search | Exact small-instance reference | `static_correctness_experiment.py` |
| KPrun-Search | Constraint-aware static Top-k search | `Dpkmor.py` |
| DPKMOR | Resource-aware state sharing and concrete-route recovery | `Dpkmor.py` |
| Full-DPKMOR | Exact recomputation after an event | `dynamic_kmor.py` |
| EI-DPKMOR | Certified event-driven maintenance | `dynamic_kmor.py` |
| KineticTree-TopK | Adapted dynamic baseline | `dynamic_kinetic_tree.py` |
| TD-TopK-Insertion | Adapted dynamic baseline | `dynamic_td_insertion.py` |
| RHLS-TopK | Adapted dynamic baseline | `dynamic_rhls.py` |

The legacy filename `Kpurn_search.py` has been retained for compatibility. The main static experimental entry points are `static_correctness_experiment.py` and `static_threeway_efficiency.py`.

## 6. Reproducing the Paper Experiments

The commands below document **archived experimental configurations**. Except for the Quick Start, they are **not guaranteed to run from a fresh public clone**, because the original fixed input banks and some reference outputs are missing. Supply the original inputs at the paths specified by the configuration files before running them. Write newly generated results to separate directories, rather than overwriting archived summaries. `--timeout` denotes a per-workload limit, not total batch runtime.

### 6.1 Static Top-k exactness — Table III

The submitted evaluation uses **720 fixed queries** over eight networks and `k ∈ {1, 3, 5}`, for **2,160 query–k configurations**. KPrun-Search and DPKMOR are compared against Exhaustive Search. The comparison checks objective values, served-request sets, event orders, and complete road paths.

```bash
python -B static_correctness_experiment.py --workdir . --config paper_datasets_8.json --output-dir reproduction_runs/static_correctness --query-bank paper_results_correctness_8_final/correctness_queries.jsonl --datasets OL,TG,CD,NY,BAY,COL,FLA,NW --request-sizes 8,10,12 --k-values 1,3,5 --queries-per-size 30 --capacity 4 --detour-ratio 1.4 --dp-mode bundled --max-states 2000000 --exact-max-states 5000000 --timeout 3600 --reuse-query-bank
```

The fixed query bank `paper_results_correctness_8_final/correctness_queries.jsonl` is **not included**. The historical summary `paper_results_correctness_8_final/correctness_summary.csv` is included as an archived result, not as a substitute for the missing inputs. Empirical agreement does not by itself constitute a mathematical proof of exactness.

### 6.2 Static search efficiency — Table IV

The experiment compares Basic Search, KPrun-Search, and DPKMOR on **240 paired query IDs** (30 per network). It reports search time and expanded states; separately recorded preprocessing is excluded from `search_time`.

```bash
python -B static_threeway_efficiency.py --workdir . --config paper_datasets_8.json --output-dir reproduction_runs/static_threeway --query-bank paper_results_kprun_bundled_final/efficiency_queries.jsonl,paper_results_static8_final/efficiency/efficiency_queries.jsonl --reference-results paper_results_kprun_bundled_dag_r25_final/kprun_efficiency_results.csv,paper_results_static8_final/efficiency/kprun_efficiency_results.csv --reference-query-only --datasets OL,TG,CD,NY,BAY,COL,FLA,NW --queries-per-dataset 30 --capacity 4 --k 5 --detour-ratio 1.4 --max-states 2000000 --timeout 3600
```

The referenced query banks and per-query reference CSVs are **not included**. `paper_results_threeway_r25_8_final/threeway_efficiency_summary.csv` is an archived summary. Do not combine absolute timing measurements from separate experimental batches as if they came from one synchronized run.

### 6.3 Dynamic maintenance and baselines — Table V and Fig. 5

The submitted dynamic evaluation contains **2,370 distinct EI-DPKMOR events** across eight networks:

| Event type | Count |
|---|---:|
| Request arrival | 960 |
| Request cancellation | 130 |
| Edge-weight update | 1,280 |
| **Total** | **2,370** |

The archived workload consists of 1,187 events for OL/TG/CD/NY and 1,183 events for BAY/COL/FLA/NW. Rows for multiple algorithms on the same event are **method–event records**, not additional independent events.

**Old-four network batch:**

```bash
python -B dynamic_paired_experiment.py --workdir . --manifest replication_manifests/dynamic_old4.json --output-dir reproduction_runs/dynamic_old4 --datasets OL,TG,CD,NY --scenarios baseline --seeds 1,2,3,4,5,6,7,8,9,10 --capacity 4 --k 5 --detour-ratio 1.4 --max-states 30000 --timeout 3600 --variants full --online-strategies incremental --paired-order alternating --resume
```

For the new-four network batch, use `replication_manifests/dynamic_new4.json`, select `BAY,COL,FLA,NW`, set `--timeout 10800`, and select `--online-strategies incremental,kinetic_tree,td_topk_insertion,rhls_topk`.

The submitted manuscript reports **1.241×–2.591×** speedups of EI-DPKMOR over paired Full-DPKMOR recomputation, with identical concrete Top-k results across the evaluated events. The old-four and new-four workloads were executed in separate batches: use each algorithm's paired Full-DPKMOR reference rather than comparing unmatched absolute times across batches. Dynamic event latency excludes process startup and initial preprocessing.

Selected archived summaries are stored at:

- `paper_results_dynamic_certificate_final/dynamic_paired_summary_results.csv`
- `paper_results_dynamic_dimacs_external_final/dynamic_paired_summary_results.csv`

The event manifests are included, but the fixed request and event trace files they reference are **not included**.

### 6.4 Event-specific processing — Table VI and Fig. 6

This analysis separates request arrivals, cancellations, and edge updates. Fig. 6 reports speedup against paired Full-DPKMOR; Table VI reports per-event averages of:

- **Expanded states:** Full-DPKMOR versus EI-DPKMOR.
- **IT:** Insertion transitions tested.
- **PS:** Prefix transitions saved.
- **AR:** Affected routes.

The complete per-event records are required to regenerate the submitted event-specific statistics and plots; summary CSVs alone may not contain sufficient detail.

### 6.5 Completeness certification — Section VII-E

The `IncrementalOnly` diagnostic uses the old-four event traces with `--variants incremental_only` and a separate output directory. It disables local reoptimization and full recovery; **it is not a certificate-only ablation**.

The submitted manuscript reports **24 concrete Top-k mismatches among 1,187 IncrementalOnly events**: 15 with different objective frontiers and 9 with identical objective frontiers but different concrete route sets. Across 2,370 EI-DPKMOR events, **three** required full DPKMOR recovery.

A `COMPLETE` execution status does not itself establish agreement with Full-DPKMOR. Report `TIMEOUT`, `STATE_LIMIT`, `INCOMPLETE`, and invalid outputs separately. The original diagnostic archive is not included in this release.

### 6.6 Sensitivity analysis — Fig. 7

The submitted Fig. 7 uses **OL, TG, CD, and NY** and varies:

| Parameter | Values |
|---|---|
| Request arrivals | `6, 12, 18` |
| Top-k size | `1, 3, 5, 10` |
| Traffic episodes | `4, 8, 12` |
| Traffic factor | `1.2, 1.5, 2.0` |

The figure generator is `plot_dynamic_sensitivity_2x2.py`. Recreating the submitted plots requires the full paired event CSVs and metadata, which are **not included**. Additional unfinished sensitivity workloads on BAY/COL/FLA/NW are separate from the submitted Fig. 7 and should not be substituted for it.


### 6.7 Submitted Paper Tables and Figures

The following table provides a mapping between the experimental results reported in the submitted manuscript and their corresponding implementations and supporting materials.

| Paper Item | Experiment | Implementation / Materials |
|---|---|---|
| Table II | Statistics of Eight Road Networks | `paper_datasets_8.json` |
| Table III | Exactness of Static Top-k Computation | `static_correctness_experiment.py` |
| Table IV | Efficiency of Static Top-k Computation | `static_threeway_efficiency.py` |
| Table V | Performance Comparison of Dynamic Algorithms | `dynamic_paired_experiment.py` |
| Table VI | Event-Specific Processing Statistics | Archived dynamic event records |
| Fig. 5 | Overall Speedup of EI-DPKMOR over Full-DPKMOR | `plot_dynamic_overall_speedup_eight.py` |
| Fig. 6 | Event-Specific Speedup of EI-DPKMOR over Full-DPKMOR | `plot_dynamic_event_type_grouped_eight.py` |
| Fig. 7 | Sensitivity of EI-DPKMOR Performance | `plot_dynamic_sensitivity_2x2.py` |

The repository provides algorithm implementations, experimental configurations, and selected archived results. Reproducing the original experiments requires the corresponding fixed workloads and datasets, some of which are not included in the current release. Archived results are preserved separately from newly generated experimental outputs.


## 7. Repository Layout

The code retains its original flat module layout to preserve existing imports and experiment entry points. The principal files are:

```text
KMOR-Replication/
├── README.md
├── requirements.txt
├── Dpkmor.py
├── Kpurn_search.py
├── dynamic_kmor.py
├── static_correctness_experiment.py
├── static_threeway_efficiency.py
├── dynamic_paired_experiment.py
├── dynamic_kinetic_tree.py
├── dynamic_td_insertion.py
├── dynamic_rhls.py
├── generate_dynamic_scenarios.py
├── prepare_dimacs_datasets.py
├── restore_road_data.py
├── replication_validate.py
├── plot_dynamic_*.py
├── test_*.py
├── paper_datasets_8.json
├── replication_manifests/
├── edges_demo.txt
├── requests_with_time_demo.txt
├── DATA_FILES.json
├── UPLOAD_FILES.json
└── paper_results_*/       # Selected archived summary CSVs
```

Large original road/request files, the fixed query banks, raw event traces, and checkpoints may be held outside the publicly uploaded files. Check the actual GitHub file tree and the release inventory before assuming any optional data file is available. `UPLOAD_FILES.json` should describe only the files actually included in the public release.

## 8. Reproducibility Notes

- **Exact inputs:** Replaying the submitted workload requires the original vehicle states, fixed queries, requests, road networks, dynamic events, seeds, and parameter configurations. Newly generated workloads with the same parameters are new experiments, not the exact historical inputs.
- **Concrete Top-k agreement:** Matching coverage and cost is insufficient. Compare selected requests, pickup–delivery event sequences, complete road trajectories, and the prescribed Top-k ordering.
- **Resource limits:** Distinguish successful certified results from timeouts, state limits, incomplete executions, and invalid outputs.
- **Timing:** Static `search_time` excludes separately measured preprocessing; dynamic event timing begins from an already-loaded state. Runtime values may change on other hardware.
- **Archival provenance:** `UPLOAD_FILES.json` and `DATA_FILES.json` record checksums, not historical Git commits, source permissions, or experimental provenance.
- **Third-party materials:** Do not redistribute datasets or incorporated third-party material without confirming the applicable permissions.

## 9. Citation

If this code or experimental framework is useful in your research, please reference the associated manuscript:

**Top-k Multi-Constraint Route Planning over On-Demand Transportation Networks.** Manuscript submitted to IEEE ICDE 2027.

A complete bibliographic citation will be added when a public publication record is available. No DOI, acceptance decision, or publication date is implied by this repository.

## 10. License and Redistribution

No repository-wide license is currently asserted. The code and any third-party datasets or implementations may be subject to different ownership and redistribution terms. Please confirm permission before publicly distributing the staged road/request files or reusing third-party material.

---

This repository is intended to support transparent verification of the submitted KMOR experiments while clearly distinguishing archived results from new reproductions.
