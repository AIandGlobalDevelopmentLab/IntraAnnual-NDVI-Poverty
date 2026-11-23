"""Bootstrap intervals for the summaries over architectures reported in the manuscript.

Reads eval_runs.csv and eval_test_predictions.csv.gz written by notebook 16 and writes
summary_bootstrap.csv next to them. Same design as notebooks 16-18: 1,000 paired Poisson(1)
resamples of the spatial groups (row 0 = the point estimate), seed 20261117.

Usage: python summary_bootstrap.py [evaluation folder]   (default: $IWI_EVAL_DIR, else results/iwi)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('IWI_EVAL_DIR', REPO / 'results' / 'iwi'))
SEED, B = 20261117, 1000
NINE = ['resnet18', 'resnet34', 'resnet50', 'swin', 'unet', 'vit', 'satclip', 'mosaiks', 'olmoearth']
EIGHT = [m for m in NINE if m != 'mosaiks']          # the newer architectures with pretrained weights
BASELINES = ['vgg16bn', 'mlp']
COMBOS = ['L+NL', 'L+NL+wavelet', 'L+NL+DFT', 'L+NL+peak:pixel11/wavelet6', 'L+NL+TT:cyclic']

runs = pd.read_csv(E / 'eval_runs.csv', dtype={'folds': str, 'notebook': str}, keep_default_na=False)
runs = runs[(runs['used'].astype(str).str.lower() == 'true') & (runs['family'] == 'ten_models')]
cols = {'iwi', 'group', 'area'} | set(runs['run_id'])
wide = pd.read_csv(E / 'eval_test_predictions.csv.gz', usecols=lambda c: c in cols)
y = wide['iwi'].to_numpy(float)
g = pd.factorize(wide['group'])[0]
G = g.max() + 1
area = wide['area'].fillna('').to_numpy()
R = {(r.model, r.combo, r.start): r.run_id for r in runs.itertuples()}
keys = list(R)
idx = {k: i for i, k in enumerate(keys)}
P = np.column_stack([wide[R[k]].to_numpy(float) for k in keys])
assert np.isfinite(P).all(), 'missing predictions'
E2 = (y[:, None] - P) ** 2
rng = np.random.default_rng(SEED)
W = np.vstack([np.ones((1, G)), rng.poisson(1.0, size=(B, G)).astype(float)])


def r2_matrix(mask):
    """R2 of every run in every resample, on the clusters in mask: (B + 1) x runs."""
    gg = g[mask]
    S = np.zeros((G, len(keys)))
    np.add.at(S, gg, E2[mask])
    n = np.bincount(gg, minlength=G).astype(float)
    sy = np.bincount(gg, weights=y[mask], minlength=G)
    syy = np.bincount(gg, weights=y[mask] ** 2, minlength=G)
    N, SY, SYY = W @ n, W @ sy, W @ syy
    return 1 - (W @ S) / (SYY - SY * SY / N)[:, None]


def ci(v):
    return float(v[0]), float(np.quantile(v[1:], 0.025)), float(np.quantile(v[1:], 0.975)), float((v[1:] > 0).mean())


out = []
for subset, mask in [('all', np.ones(len(y), bool)), ('rural', area == 'rural'), ('urban', area == 'urban')]:
    R2 = r2_matrix(mask)

    def r(m, c, s):
        return R2[:, idx[(m, c, s)]]

    def add(start, combo, quantity, v):
        out.append((subset, start, combo, quantity, *ci(v)))

    for s in ['pretrained', 'scratch']:
        nine = {c: np.mean([r(m, c, s) for m in NINE], axis=0) for c in COMBOS}
        for c in COMBOS:
            add(s, c, 'nine: mean R2', nine[c])
            for b in BASELINES:
                add(s, c, f'nine minus {b}, same input', nine[c] - r(b, c, s))
                add(s, c, f'nine minus {b} on L+NL', nine[c] - r(b, 'L+NL', s))
            if c == 'L+NL':
                continue
            gain9 = nine[c] - nine['L+NL']
            add(s, c, 'nine: mean gain over L+NL', gain9)
            if c != 'L+NL+wavelet':
                add(s, c, 'nine: minus wavelet', nine[c] - nine['L+NL+wavelet'])
            for b in BASELINES:
                own = r(b, c, s) - r(b, 'L+NL', s)
                add(s, c, f'{b}: gain over L+NL', own)
                add(s, c, f'{b} gain minus nine mean gain', own - gain9)
        gain9_4 = np.mean([nine[c] - nine['L+NL'] for c in COMBOS[1:]], axis=0)
        add(s, 'mean of 4 inputs', 'nine: mean gain over L+NL', gain9_4)
        for b in BASELINES:
            own4 = np.mean([r(b, c, s) - r(b, 'L+NL', s) for c in COMBOS[1:]], axis=0)
            add(s, 'mean of 4 inputs', f'{b}: gain over L+NL', own4)
            add(s, 'mean of 4 inputs', f'{b} gain minus nine mean gain', own4 - gain9_4)
        add(s, 'L+NL', 'satclip minus resnet18', r('satclip', 'L+NL', s) - r('resnet18', 'L+NL', s))
        add(s, 'L+NL', 'vgg16bn minus mlp', r('vgg16bn', 'L+NL', s) - r('mlp', 'L+NL', s))
    for c in COMBOS:
        add('pretrained minus scratch', c, 'eight: mean', np.mean([r(m, c, 'pretrained') - r(m, c, 'scratch') for m in EIGHT], axis=0))
        add('pretrained minus scratch', c, 'vgg16bn', r('vgg16bn', c, 'pretrained') - r('vgg16bn', c, 'scratch'))

df = pd.DataFrame(out, columns=['subset', 'start', 'combo', 'quantity', 'estimate', 'low', 'high', 'P(>0)'])
df.to_csv(E / 'summary_bootstrap.csv', index=False, float_format='%.6f')
print(f'{len(df)} rows -> {E / "summary_bootstrap.csv"}')
