# IceFrog Simulator and Experiment Scripts (simulator-artifact)

This repository contains a simulator and experiment scripts for deadline-aware, elastic scheduling of distributed training on multi-GPU clusters. It is designed to reproduce and analyze:
- DeadlineMeet (ADAPT‑DM) scheduling: utility maximization (goodput × deadline pressure × elastic weights) with feasible bin-packing;
- Restart avoidance: suppress unnecessary restarts when allocation changes are minor;
- Collaborative Resource Redistribution (CRR): small, urgency- and marginal-gain-guided resource transfers between jobs;
- DDF (Deadline Debt Fairness): an independent, lightweight, online “deadline-debt water-filling” scheduler.

It also includes experiment runners, traces, analysis utilities, and test scripts to help you quickly reproduce results and extend the work.


## Environment

We recommend using Conda (Python 3.8). This folder provides both `environment.yaml` and `requirements.txt`.

```bash
# 1) Create and activate the environment (name is "integrated-scheduler" in environment.yaml)
conda env create -f environment.yaml
conda activate integrated-scheduler

# 2) Optionally (or additionally) install Python dependencies
pip install -r requirements.txt
```

Notes:
- `environment.yaml` pins Python=3.8 and a validated set of packages (mip, numpy, pandas, torch, etc.);
- GPU/torch is not required if you only run log analysis and plotting;
- If reusing an existing environment, keep core versions close to those in the YAML (notably numpy/scipy/pandas/mip).


## Quick start

Batch runners and plotting helpers are provided. Results are stored under `N_results/main/...` (N = 0, 1, 8, 9, ...).

```bash
# Run the main experiments (see script contents for details)
bash exp/main/bash_run_main.sh

# Re-run missing/failed parts if needed
bash exp/main/run_missing_results.sh
```

Common entry points (pick what you need):
- `simple_integrated.py`: DeadlineMeet/ADAPT‑DM main scheduling;
- `simple_integrated_smart_crr.py`: enable CRR on top of the main logic;
- `simple_pollux.py`, `simple_integrated_complex.py`, `simple_icefrog_backup.py`: baselines/variants;
- `demo_crr.py`: CRR demo/validation;
- `simulator.py`: common entry/args and orchestration;
- `exp/main/plot_workload_simple_lucid.py`: a plotting example.

Tip: arguments differ slightly across scripts. Skim the headers and `simulator.py` to choose appropriate flags.


## Result layout

A typical result tree (example: `8_results/main`; other `N_results` follow the same pattern):

```
8_results/
  main/
    workload-1.0/
    workload-2.0/
    workload-3.0/
    workload-4.0/
      <strategy-name>/
        workload-1.log
        workload-2.log
        workload-3.log
        summary.json
```

Conventions:
- `workload-x.0` denotes different workload groups/scales;
- Each strategy folder contains JSON-lines logs for 1/2/3 combinations (`workload-*.log`) and an overall `summary.json`;
- A log line may be a JSON object or an array; tasks reside under the `submitted_jobs` field.


## Analysis tools and scripts

These utilities can be run standalone to analyze existing results.

### 1) Average task overdue (seconds)

File: `tools/overdue_per_task.py`

Definition: overdue = max(0, completion_time - deadline). Only completed tasks are included.

Usage:
```bash
python3 tools/overdue_per_task.py [path/to/workload-*.log]
```
- If no argument is provided, the script uses its built-in `DEFAULT_LOG_PATH`;
- Output: prints a single numeric value to stdout (average overdue in seconds, fixed decimals);
- Robust to JSON-lines where each line is either an object or an array of objects.

### 2) Max JCT per workload

File: `tools/max_jct_per_workload.py`

Usage:
```bash
python3 tools/max_jct_per_workload.py [path/to/summary.json]
```
- If omitted, uses the built-in `DEFAULT_SUMMARY_PATH` in the script;
- Output format: `workload-X\tmax_jct=<value>\tjob=<job_name>`.

### 3) Restart counts (with combo granularity)

File: `analyze_restarts.py`

Function: scans `8_results/main` across the four `workload-*.0` groups, and for each strategy and each combo (`workload-1/2/3`), computes the maximum restart count per job and summarizes the stats.

Usage:
```bash
python3 analyze_restarts.py
```
Outputs:
- Detailed CSV: `restart_analysis_detailed_8results.csv`
- Summary CSV: `restart_analysis_summary_8results.csv`
- Console: per-strategy and per-combo summaries (mean/median/std/restart-rate, etc.).

If your results are under a different folder than `8_results/main`, tweak the default `results_dir` in the script.


## Data and traces

- Workload definitions: `workloads-*/workload-*.csv`;
- Traces: `traces/`;
- Example weights/devices: `weights/`;
- CRR documentation: `docs/CRR_Guide.md`.


## Algorithm overview (as implemented)

- ADAPT‑DM (DeadlineMeet main logic)
  - Upper layer: 0‑1 utility maximization over discrete candidates (goodput × deadline pressure × elastic weights), solved with CBC (mip) in seconds;
  - Lower layer: node-level greedy bin-packing for feasibility and stability (non-preemptible first, capacity trimming, rollback on failure).

- Restart avoidance
  - Keep the current allocation when changes are small or the job is close to its deadline to reduce restart/migration losses.

- CRR (Collaborative Resource Redistribution)
  - Iterative small-step transfers based on urgency and marginal utility to boost overall on-time rate;
  - After each round, validate and fix capacity violations.

- DDF (Deadline Debt Fairness; can run independently)
  - Introduces “deadline debt” and uses water-filling to allocate GPUs to the most indebted jobs first, then balances by marginal gains;
  - Online, lightweight, interpretable; roughly O(G·log|J|).

For full mathematical details and execution flow, refer to the accompanying document in the same workspace: `../words/words.txt` (if available).


## Tests and validation

Run the built-in tests directly:

```bash
# CRR comprehensive test (basic/edge/metrics)
python3 test_crr.py --test all

# Other checks (read script headers for details)
python3 test_violation_logic.py
python3 test_workload_violations.py
python3 test_random_violations.py
python3 test_type_safety.py
```


## Troubleshooting (FAQ)

- Conda env creation is slow or fails: try `conda env create -f environment.yaml` first, then `pip install -r requirements.txt`; if needed, pin older numpy/pandas for compatibility.
- Script can’t find your result files: check the `DEFAULT_*` paths at the top of each tool, or pass explicit file paths via CLI arguments.
- JSON-lines parsing errors: logs may have lines that are either objects or arrays; `tools/overdue_per_task.py` handles both. If errors persist, ensure the file isn’t truncated or malformed.
