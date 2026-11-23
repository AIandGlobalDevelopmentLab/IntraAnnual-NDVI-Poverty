"""Write the spatial fold of every cluster for publication: cluster_id, group and fold only.

The folds file of notebook iwi/03 (train_clusters_folds.csv) also holds the DHS coordinates and wealth values,
which must not be redistributed. This script keeps only the identifier, the spatial group (used by the cluster
bootstrap) and the fold, so that registered DHS users can rebuild the exact splits.

Usage: python tools/export_folds.py /path/to/data/train_clusters_folds.csv [results/folds.csv]
"""
import sys
import pandas as pd

src = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else 'results/folds.csv'
keep = ['cluster_id', 'group', 'fold']
df = pd.read_csv(src, usecols=lambda c: c in keep, dtype={'cluster_id': str})
missing = [c for c in keep if c not in df]
assert not missing, f'{src} has no column {missing}'
df[keep].sort_values('cluster_id').to_csv(out, index=False)
print(f'{len(df):,} clusters, {df["group"].nunique():,} spatial groups, folds {sorted(df["fold"].unique())} -> {out}')
