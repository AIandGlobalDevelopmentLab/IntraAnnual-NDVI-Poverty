"""Bootstrap intervals for the summaries over architectures of the NL-target study.

Reads eval_runs.csv and eval_test_predictions.csv.gz written by 16_Evaluate_All_Models and writes
summary_bootstrap.csv next to them. Same design as the IWI study's summary_bootstrap.py and notebooks 16-18:
1,000 paired Poisson(1) resamples of the spatial groups (row 0 = the point estimate), seed 20261117.
Differences from the IWI script: the target is `nl`, the baseline input is Landsat ('L'), the subsets also
include the two NL periods (DMSP, VIIRS), and runs with fewer than five test folds are allowed: every
comparison of two runs uses the clusters both runs predict, and means over architectures average these
paired differences.

Usage: python summary_bootstrap.py [evaluation folder]   (default: $NL_EVAL_DIR, else results/nl)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('NL_EVAL_DIR', REPO / 'results' / 'nl'))
SEED, B = 20261117, 1000
TARGET, BASE = 'nl', 'L'
NINE = ['resnet18', 'resnet34', 'resnet50', 'swin', 'unet', 'vit', 'satclip', 'mosaiks', 'olmoearth']
EIGHT = [m for m in NINE if m != 'mosaiks']          # the newer architectures with pretrained weights
BASELINES = ['vgg16bn', 'mlp']
COMBOS = ['L', 'L+wavelet', 'L+DFT', 'L+peak:pixel11/wavelet6', 'L+TT:cyclic']

runs = pd.read_csv(E / 'eval_runs.csv', dtype={'folds': str, 'notebook': str}, keep_default_na=False)
runs = runs[(runs['used'].astype(str).str.lower() == 'true') & (runs['family'] == 'ten_models')]
cols = {TARGET, 'group', 'area', 'nl_sensor'} | set(runs['run_id'])
wide = pd.read_csv(E / 'eval_test_predictions.csv.gz', usecols=lambda c: c in cols)
y = wide[TARGET].to_numpy(float)
g = pd.factorize(wide['group'])[0]
G = g.max() + 1
area = wide['area'].fillna('').astype(str).to_numpy()
period = wide['nl_sensor'].fillna('').astype(str).to_numpy()
R = {(r.model, r.combo, r.start): r.run_id for r in runs.itertuples()}
P = {k: wide[v].to_numpy(float) for k, v in R.items()}
rng = np.random.default_rng(SEED)
W = np.vstack([np.ones((1, G)), rng.poisson(1.0, size=(B, G)).astype(float)])
_cache = {}


def r2_on(k, mask):
    """R2 of run k in every resample, on the clusters of mask (cached by run and mask)."""
    key = (k, mask.tobytes().__hash__(), int(mask.sum()))
    if key not in _cache:
        gg, yy = g[mask], y[mask]
        e2 = (yy - P[k][mask]) ** 2
        n = np.bincount(gg, minlength=G).astype(float)
        sy = np.bincount(gg, weights=yy, minlength=G)
        syy = np.bincount(gg, weights=yy * yy, minlength=G)
        sse = np.bincount(gg, weights=e2, minlength=G)
        N, SY, SYY, SSE = W @ n, W @ sy, W @ syy, W @ sse
        _cache[key] = 1 - SSE / (SYY - SY * SY / N)
    return _cache[key]


def diff(a, b, subset):
    """R2(a) - R2(b) in every resample, on the clusters of the subset that both runs predict."""
    m = subset & np.isfinite(P[a]) & np.isfinite(P[b]) & np.isfinite(y)
    return r2_on(a, m) - r2_on(b, m)


def level(a, subset):
    m = subset & np.isfinite(P[a]) & np.isfinite(y)
    return r2_on(a, m)


def ci(v):
    return float(v[0]), float(np.quantile(v[1:], 0.025)), float(np.quantile(v[1:], 0.975)), float((v[1:] > 0).mean())


out = []
SUBSETS = [('all', np.ones(len(y), bool)), ('rural', area == 'rural'), ('urban', area == 'urban'),
           ('DMSP', period == 'DMSP'), ('VIIRS', period == 'VIIRS')]
for subset, mask in SUBSETS:
    def add(start, combo, quantity, v):
        out.append((subset, start, combo, quantity, *ci(v)))

    for s in ['pretrained', 'scratch']:
        for c in COMBOS:
            add(s, c, 'nine: mean R2', np.mean([level((m, c, s), mask) for m in NINE], axis=0))
            for b in BASELINES:
                add(s, c, f'nine minus {b}, same input',
                    np.mean([diff((m, c, s), (b, c, s), mask) for m in NINE], axis=0))
                add(s, c, f'nine minus {b} on L', np.mean([diff((m, c, s), (b, BASE, s), mask) for m in NINE], axis=0))
        gains = {}
        for c in COMBOS[1:]:
            gains[c] = np.mean([diff((m, c, s), (m, BASE, s), mask) for m in NINE], axis=0)
            add(s, c, 'nine: mean gain over L', gains[c])
            if c != 'L+wavelet':
                add(s, c, 'nine: minus wavelet', np.mean([diff((m, c, s), (m, 'L+wavelet', s), mask) for m in NINE], axis=0))
            for b in BASELINES:
                own = diff((b, c, s), (b, BASE, s), mask)
                add(s, c, f'{b}: gain over L', own)
                add(s, c, f'{b} gain minus nine mean gain', own - gains[c])
        g4 = np.mean([gains[c] for c in COMBOS[1:]], axis=0)
        add(s, 'mean of 4 inputs', 'nine: mean gain over L', g4)
        for b in BASELINES:
            own4 = np.mean([diff((b, c, s), (b, BASE, s), mask) for c in COMBOS[1:]], axis=0)
            add(s, 'mean of 4 inputs', f'{b}: gain over L', own4)
            add(s, 'mean of 4 inputs', f'{b} gain minus nine mean gain', own4 - g4)
        add(s, BASE, 'satclip minus resnet18', diff(('satclip', BASE, s), ('resnet18', BASE, s), mask))
        add(s, BASE, 'vgg16bn minus mlp', diff(('vgg16bn', BASE, s), ('mlp', BASE, s), mask))
    for c in COMBOS:
        add('pretrained minus scratch', c, 'eight: mean',
            np.mean([diff((m, c, 'pretrained'), (m, c, 'scratch'), mask) for m in EIGHT], axis=0))
        add('pretrained minus scratch', c, 'vgg16bn', diff(('vgg16bn', c, 'pretrained'), ('vgg16bn', c, 'scratch'), mask))

df = pd.DataFrame(out, columns=['subset', 'start', 'combo', 'quantity', 'estimate', 'low', 'high', 'P(>0)'])
df.to_csv(E / 'summary_bootstrap.csv', index=False, float_format='%.6f')
partial = runs[runs['folds'] != 'ABCDE']
print(f'{len(df)} rows -> {E / "summary_bootstrap.csv"}; runs with fewer folds:',
      dict(zip(partial['run_id'], partial['folds'])))
