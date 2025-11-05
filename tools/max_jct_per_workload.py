#!/usr/bin/env python3
import json
import sys
from pathlib import Path


def load_json(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def max_jct_per_workload(summary: dict):
    jcts = summary.get('jcts', {})
    results = {}
    for workload, jobs in jcts.items():
        if not isinstance(jobs, dict) or not jobs:
            continue
        # find max jct and corresponding job
        job, max_jct = max(jobs.items(), key=lambda kv: kv[1])
        results[workload] = (job, max_jct)
    return results


DEFAULT_SUMMARY_PATH = Path(
    #"/home/lab/simulator-artifact/11_results/main/workload-4.0/simple_icefrog-deadline-avoid-restart/summary.json"
    "/home/lab/simulator-artifact/9_results/main/workload-4.0/simple_icefrog-deadline-avoid-restart/summary.json"
)


def main(argv):
    # If a path is provided, use it; otherwise fall back to the built-in default
    if len(argv) >= 2:
        summary_path = Path(argv[1])
    else:
        summary_path = DEFAULT_SUMMARY_PATH
        print(f"Using built-in default: {summary_path}", file=sys.stderr)
    if not summary_path.exists():
        print(f"File not found: {summary_path}", file=sys.stderr)
        return 2

    try:
        summary = load_json(summary_path)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}", file=sys.stderr)
        return 2

    results = max_jct_per_workload(summary)
    if not results:
        print("No workloads found in 'jcts' section.")
        return 0

    # Pretty print results
    for workload in sorted(results.keys()):
        job, value = results[workload]
        print(f"{workload}\tmax_jct={value}\tjob={job}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
