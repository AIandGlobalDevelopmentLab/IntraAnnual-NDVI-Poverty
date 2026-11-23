"""Mean change in MAE of the nine newer architectures for each NDVI encoding (NL-target study, WaveletNL).

Reads eval_test_predictions.csv.gz written by 16_Evaluate_All_Models and writes mae_bootstrap.csv next to it:
for each start, encoding and subset (all, rural, urban, DMSP, VIIRS) the mean over the nine architectures of
MAE(encoding) - MAE(Landsat), its mean relative change, and a 95 % interval from 1,000 paired Poisson(1)
resamples of the spatial groups (seed 20261117, as in summary_bootstrap.py). Each architecture is scored on
the clusters that both of its runs predict (the temporal-transformer runs of swin and olmoearth lack folds).

Usage: python mae_bootstrap.py [evaluation folder]   (default: $NL_EVAL_DIR, else results/nl)
"""
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
E = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(os.environ.get('NL_EVAL_DIR', REPO / 'results' / 'nl'))
NINE = ['resnet18', 'resnet34', 'resnet50', 'swin', 'unet', 'vit', 'satclip', 'mosaiks', 'olmoearth']
ENC = ['L+wavelet', 'L+DFT', 'L+peak:pixel11/wavelet6', 'L+TT:cyclic']
cols = {'nl', 'group', 'rural', 'nl_sensor'} | {f'ten_models|{m}|{c}|{s}' for m in NINE for c in ENC + ['L']
                                               for s in ('pretrained', 'scratch')}
P = pd.read_csv(E / 'eval_test_predictions.csv.gz', usecols=lambda c: c in cols)
y = P['nl'].to_numpy(float)
g = pd.factorize(P['group'])[0]
W = np.random.default_rng(20261117).poisson(1.0, size=(1000, g.max() + 1)).astype(float)[:, g]
rural = P['rural'].astype(str).str.lower().eq('true').to_numpy()
subsets = {'all': np.ones(len(P), bool), 'rural': rural, 'urban': ~rural,
           'DMSP': (P['nl_sensor'] == 'DMSP').to_numpy(), 'VIIRS': (P['nl_sensor'] == 'VIIRS').to_numpy()}
rows = []
for start in ['pretrained', 'scratch']:
    for enc in ENC:
        for name, sub in subsets.items():
            est, rel, boot = [], [], []
            for m in NINE:
                a = P[f'ten_models|{m}|{enc}|{start}'].to_numpy(float)
                b = P[f'ten_models|{m}|L|{start}'].to_numpy(float)
                k = sub & np.isfinite(a) & np.isfinite(b)
                ea, eb = np.abs(y[k] - a[k]), np.abs(y[k] - b[k])
                est.append(ea.mean() - eb.mean())
                rel.append(est[-1] / eb.mean())
                w = W[:, k]
                boot.append((w @ ea - w @ eb) / w.sum(1))
            boot = np.mean(boot, axis=0)
            rows.append(dict(start=start, combo=enc, subset=name, dMAE=np.mean(est), rel_dMAE=np.mean(rel),
                             low=np.percentile(boot, 2.5), high=np.percentile(boot, 97.5)))
out = pd.DataFrame(rows)
out.to_csv(E / 'mae_bootstrap.csv', index=False, float_format='%.6f')
print(out.to_string(index=False, float_format=lambda v: '%.4f' % v))
