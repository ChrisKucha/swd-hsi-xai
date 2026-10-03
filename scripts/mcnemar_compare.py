"""
mcnemar_compare.py

Exact McNemar's test between two model runs evaluated on the SAME test set,
using the per-sample *_results.csv files written by evaluate.py.

Usage
-----
    python mcnemar_compare.py runA_results.csv runB_results.csv

The test uses only the discordant pairs (samples one model gets right and the
other gets wrong) with an exact two-sided binomial p-value, which is valid at
any sample size and needs no continuity correction.

Safety check. The script refuses to run unless both files contain the same
number of rows with an identical true_label sequence, which guards against
comparing runs evaluated on different test sets or orderings.
"""

import argparse
import csv
import math
import sys


def load(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"Empty file: {path}")
    try:
        true = [int(float(r["true_label"])) for r in rows]
        pred = [int(float(r["pred_label"])) for r in rows]
    except KeyError:
        sys.exit(f"{path} lacks true_label/pred_label columns "
                 f"(found: {list(rows[0].keys())[:8]}...)")
    return true, pred


def exact_mcnemar_p(b, c):
    """Two-sided exact binomial test on discordant counts b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided p = 2 * P(X <= k) under Binomial(n, 0.5), capped at 1
    cdf = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2.0 * cdf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--label_a", default="Model A")
    ap.add_argument("--label_b", default="Model B")
    args = ap.parse_args()

    true_a, pred_a = load(args.file_a)
    true_b, pred_b = load(args.file_b)

    if len(true_a) != len(true_b):
        sys.exit(f"Row count mismatch: {len(true_a)} vs {len(true_b)}. "
                 "These runs were not evaluated on the same test set.")
    if true_a != true_b:
        sys.exit("true_label sequences differ between the two files. "
                 "Same test set and same evaluation order are required "
                 "for a paired test. Aborting.")

    n = len(true_a)
    correct_a = [p == t for p, t in zip(pred_a, true_a)]
    correct_b = [p == t for p, t in zip(pred_b, true_b)]

    both = sum(a and b for a, b in zip(correct_a, correct_b))
    neither = sum((not a) and (not b) for a, b in zip(correct_a, correct_b))
    b_count = sum(a and (not b) for a, b in zip(correct_a, correct_b))  # A right, B wrong
    c_count = sum((not a) and b for a, b in zip(correct_a, correct_b))  # A wrong, B right

    acc_a = sum(correct_a) / n
    acc_b = sum(correct_b) / n
    p = exact_mcnemar_p(b_count, c_count)

    print(f"\n  Paired test on {n} test samples")
    print(f"  {args.label_a}: accuracy {acc_a:.4f}   ({args.file_a})")
    print(f"  {args.label_b}: accuracy {acc_b:.4f}   ({args.file_b})")
    print(f"\n  Contingency of correctness")
    print(f"    both correct        : {both}")
    print(f"    only {args.label_a:>8} correct: {b_count}")
    print(f"    only {args.label_b:>8} correct: {c_count}")
    print(f"    neither correct     : {neither}")
    print(f"\n  Exact McNemar two-sided p = {p:.4g}")
    if p < 0.05:
        print("  The accuracy difference is statistically significant at 0.05.")
    else:
        print("  The accuracy difference is NOT statistically significant at 0.05.")


if __name__ == "__main__":
    main()
