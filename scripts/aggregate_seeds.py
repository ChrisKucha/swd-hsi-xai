"""
aggregate_seeds.py

Aggregate *_summary.json metric files across several seed output trees
produced by run_all.py, matching runs by relative path.

Usage
-----
    python aggregate_seeds.py outputs outputs_seed101 outputs_seed202 --out seed_summary

Writes
------
    {out}_long.csv   one row per (run, seed_root, metric)
    {out}_wide.csv   one row per (run, metric) with mean, sd, n, and a
                     formatted "mean ± sd" string for manuscript tables
"""

import argparse
import csv
import json
import math
from pathlib import Path


def collect(root: Path) -> dict:
    """Return {relative_run_key: {metric: value}} for one output tree."""
    runs = {}
    for path in sorted(root.rglob("*_summary.json")):
        rel = str(path.relative_to(root))
        try:
            with open(path) as fh:
                data = json.load(fh)
        except Exception as exc:
            print(f"  [WARN] could not read {path}: {exc}")
            continue
        metrics = {}
        def walk(obj, prefix=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(v, f"{prefix}{k}." if isinstance(v, dict) else f"{prefix}{k}")
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                metrics[prefix] = float(obj)
        walk(data)
        runs[rel] = metrics
    return runs


def mean_sd(values):
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, float("nan")
    var = sum((v - m) ** 2 for v in values) / (n - 1)
    return m, math.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="+", help="Output directories, one per seed")
    ap.add_argument("--out", default="seed_summary", help="Output file prefix")
    args = ap.parse_args()

    roots = [Path(r) for r in args.roots]
    for r in roots:
        if not r.exists():
            ap.error(f"Directory not found: {r}")

    per_root = {str(r): collect(r) for r in roots}
    for name, runs in per_root.items():
        print(f"  {name}: {len(runs)} summary files")

    # Union of run keys
    all_keys = sorted(set().union(*[set(v) for v in per_root.values()]))

    long_rows, wide_rows = [], []
    for key in all_keys:
        present = {name: runs[key] for name, runs in per_root.items() if key in runs}
        if len(present) < len(roots):
            missing = [n for n in per_root if key not in per_root[n]]
            print(f"  [WARN] {key} missing in: {', '.join(missing)}")
        metric_names = sorted(set().union(*[set(m) for m in present.values()]))
        for metric in metric_names:
            values = []
            for name, metrics in present.items():
                if metric in metrics:
                    long_rows.append({"run": key, "seed_root": name,
                                      "metric": metric, "value": metrics[metric]})
                    values.append(metrics[metric])
            if values:
                m, sd = mean_sd(values)
                # Percent-style metrics get 2 decimals, others 4
                digits = 2 if abs(m) > 1 else 4
                sd_str = "nan" if math.isnan(sd) else f"{sd:.{digits}f}"
                wide_rows.append({
                    "run": key, "metric": metric, "n": len(values),
                    "mean": f"{m:.{digits}f}", "sd": sd_str,
                    "formatted": f"{m:.{digits}f} ± {sd_str}",
                })

    long_path = f"{args.out}_long.csv"
    wide_path = f"{args.out}_wide.csv"
    with open(long_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "seed_root", "metric", "value"])
        w.writeheader(); w.writerows(long_rows)
    with open(wide_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "metric", "n", "mean", "sd", "formatted"])
        w.writeheader(); w.writerows(wide_rows)

    print(f"\n  Wrote {long_path} ({len(long_rows)} rows)")
    print(f"  Wrote {wide_path} ({len(wide_rows)} rows)")


if __name__ == "__main__":
    main()
