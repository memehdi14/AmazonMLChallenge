"""Compare submission files with a reference (default: the HPC v11 twin) by country.

usage: python scripts/fr_diff.py <sub dir or tsv> [...] [--ref output/v14_v11chk/matching_results.tsv]
Prints matches per country vs the reference and the share of S1 whose match set changed.
"""
import argparse
import glob
import os

import pandas as pd


def load(p):
    if os.path.isdir(p):
        p = sorted(glob.glob(os.path.join(p, "matching_results.tsv*")))[0]
    s = pd.read_csv(p, sep="\t", dtype=str, keep_default_na=False)
    col = s.columns[1]
    return {a: set(x for x in b.split(",") if x) for a, b in zip(s["source1_entity_id"], s[col])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subs", nargs="+")
    ap.add_argument("--ref", default="output/v14_v11chk/matching_results.tsv")
    args = ap.parse_args()
    t = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", dtype=str, keep_default_na=False,
                    usecols=["entity_id", "country"])
    country = dict(zip(t["entity_id"], t["country"]))
    ref = load(args.ref)
    by = {c: [s for s in ref if country[s] == c] for c in sorted(set(country.values()))}
    base = {c: sum(len(ref[s]) for s in ids) for c, ids in by.items()}
    print("reference " + " | ".join(f"{c} {n:,}" for c, n in base.items()))
    for p in args.subs:
        sub = load(p)
        parts = []
        for c, ids in by.items():
            n = sum(len(sub[s]) for s in ids)
            ch = sum(sub[s] != ref[s] for s in ids) / len(ids)
            parts.append(f"{c} {n:,} ({n - base[c]:+,}, {ch:.2%} S1 changed)")
        print(f"{p}: " + " | ".join(parts))


if __name__ == "__main__":
    main()
