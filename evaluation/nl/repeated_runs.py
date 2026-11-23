"""Run-to-run variation of the NL-target study, from configurations that were trained twice.

The Landsat ('L') and Landsat + wavelet rows of notebooks 07 (ResNet-18) and 08 (ViT-S/16) repeat the
resnet18 and vit runs of notebooks 12 and 11: same bands, network, recipe, seed and folds. This script
compares the two copies: identical predictions or not, R2 of each, the difference with a 95 % interval
(1,000 paired Poisson(1) resamples of the spatial groups, seed 20261117), the correlation of the predictions,
the wavelet gain measured twice and the R2 of each copy in every test fold. Writes repeated_runs.csv.

Usage: python repeated_runs.py [evaluation folder]   (default: $NL_EVAL_DIR, else results/nl)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('NL_EVAL_DIR', REPO / 'results' / 'nl'))
PAIRS = [('resnet18', 'landsat_resnet18'), ('vit', 'landsat_vits16_finetune')]
cols = {'nl', 'group', 'fold'} | {f'{fam}|{m}|{c}|{s}' for a, b in PAIRS for fam, m in (('ten_models', a), ('single', b))
                                  for c in ('L', 'L+wavelet') for s in ('pretrained', 'scratch')}
P = pd.read_csv(E / 'eval_test_predictions.csv.gz', usecols=lambda c: c in cols)
y = P['nl'].to_numpy(float)
g = pd.factorize(P['group'])[0]
W = np.random.default_rng(20261117).poisson(1.0, size=(1000, g.max() + 1)).astype(float)[:, g]
sst_w = (W * (y[None, :] - ((W @ y) / W.sum(1))[:, None]) ** 2).sum(1)

def r2(p, m=slice(None)):
    return 1 - ((y[m] - p[m]) ** 2).sum() / ((y[m] - y[m].mean()) ** 2).sum()

def r2w(p):
    return 1 - (W @ (y - p) ** 2) / sst_w

def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)

rows = []
for main, single in PAIRS:
    for start in ['pretrained', 'scratch']:
        a = {c: P[f'ten_models|{main}|{c}|{start}'].to_numpy(float) for c in ('L', 'L+wavelet')}
        b = {c: P[f'single|{single}|{c}|{start}'].to_numpy(float) for c in ('L', 'L+wavelet')}
        for c in ('L', 'L+wavelet'):
            lo, hi = ci(r2w(b[c]) - r2w(a[c]))
            row = dict(model=main, start=start, combo=c, identical=bool(np.allclose(a[c], b[c], atol=1e-6)),
                       r2_main=r2(a[c]), r2_repeat=r2(b[c]), diff=r2(b[c]) - r2(a[c]), diff_low=lo, diff_high=hi,
                       corr=np.corrcoef(a[c], b[c])[0, 1])
            for f in sorted(P['fold'].unique()):
                m = (P['fold'] == f).to_numpy()
                row[f'fold {f} main'], row[f'fold {f} repeat'] = r2(a[c], m), r2(b[c], m)
            rows.append(row)
        ga, gb = r2w(a['L+wavelet']) - r2w(a['L']), r2w(b['L+wavelet']) - r2w(b['L'])
        lo, hi = ci(gb - ga)
        rows.append(dict(model=main, start=start, combo='wavelet gain', r2_main=r2(a['L+wavelet']) - r2(a['L']),
                         r2_repeat=r2(b['L+wavelet']) - r2(b['L']),
                         diff=(r2(b['L+wavelet']) - r2(b['L'])) - (r2(a['L+wavelet']) - r2(a['L'])),
                         diff_low=lo, diff_high=hi))
out = pd.DataFrame(rows)
out.to_csv(E / 'repeated_runs.csv', index=False, float_format='%.6f')
print(out.to_string(index=False, float_format=lambda v: '%.3f' % v))
