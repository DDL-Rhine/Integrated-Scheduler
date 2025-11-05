#!/usr/bin/env python3
"""
Compute average overdue time across completed tasks from a simulator workload log.

Definition: overdue = max(0, completion_time - deadline)
Only tasks with a non-null completion_time are included.

Usage:
    python3 tools/overdue_per_task.py [path/to/workload-*.log]

If no log path is provided, a built-in default path will be used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple, Any

import argparse


# Built-in default to the current file the user is working with
DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parent.parent
    / "7_results/main/workload-4.0/simple_icefrog-batch-fixed/workload-3.log"
)

# Project root directory (two levels above this script: simulator-artifact/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_overdues(log_path: Path) -> Dict[str, Tuple[int, float, float]]:
    """
    Parse the log and return a mapping:
      name -> (overdue_int, completion_time, deadline)
    Only records jobs the first time they appear with completion_time != None.
    """
    results: Dict[str, Tuple[int, float, float]] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # Skip any malformed lines
                continue
            # Some lines may be a dict with key 'submitted_jobs'; others might be a list of such dicts
            records: list[dict[str, Any]] = []
            if isinstance(rec, dict):
                records.append(rec)
            elif isinstance(rec, list):
                records.extend([x for x in rec if isinstance(x, dict)])
            else:
                continue

            # Aggregate all submitted_jobs arrays from the collected records
            jobs: list[dict[str, Any]] = []
            for r in records:
                maybe = r.get("submitted_jobs")
                if isinstance(maybe, list):
                    jobs.extend([j for j in maybe if isinstance(j, dict)])
            if not isinstance(jobs, list):
                continue
            for job in jobs:
                try:
                    name = job.get("name")
                    completion_time = job.get("completion_time")
                    deadline = job.get("deadline")
                except AttributeError:
                    continue
                if name is None or completion_time is None or deadline is None:
                    # Skip if incomplete fields or not yet finished
                    continue
                if name in results:
                    # Already recorded (completion/deadline are stable across lines)
                    continue
                try:
                    overdue_val = max(0.0, float(completion_time) - float(deadline))
                except (TypeError, ValueError):
                    continue
                # Report overdue as integer seconds (floor toward zero by int())
                results[name] = (int(overdue_val), float(completion_time), float(deadline))
    return results


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Compute average overdue time from workload log")
    parser.add_argument("log", nargs="?", help="Path to workload-*.log; defaults to built-in")
    args = parser.parse_args(argv[1:])

    if args.log:
        log_path = Path(args.log).expanduser().resolve()
    else:
        log_path = DEFAULT_LOG_PATH
        print(f"[info] Using built-in default log: {log_path}", file=sys.stderr)

    if not log_path.exists():
        print(f"[error] Log file not found: {log_path}", file=sys.stderr)
        return 2

    results = load_overdues(log_path)

    # Compute and print average overdue (seconds; 11 decimals)
    total_overdue = sum(v[0] for v in results.values())
    count = len(results)
    avg = (total_overdue / count) if count > 0 else 0
    # Print only the numeric value (or include a label if preferred)
    print(f"{avg:.11f}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
