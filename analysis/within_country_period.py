import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
from subset_gains import OUT
sys.path.insert(0, str(HERE))
from gainlib import Gains
from subset_gains import NINE
from scipy.stats import spearmanr
D = pd.read_pickle(OUT / 'D.pkl')
res = []
# (a) within-country period comparison, IWI
G = Gains(D, 'iwi')
for c, g in D.groupby('country'):
    nd, nv = (g.nl_sensor == 'DMSP').sum(), (g.nl_sensor == 'VIIRS').sum()
    if nd < 150 or nv < 150: continue
    for start in ('scratch', 'pretrained'):
        row = dict(country=c, start=start, n_dmsp=nd, n_viirs=nv)
        for p in ('DMSP', 'VIIRS'):
            m = ((D.country == c) & (D.nl_sensor == p)).to_numpy()
            r = G.nine(start, m)
            row[f'dR2_{p}'], row[f'rel_{p}'], row[f'se_{p}'] = r['dR2'], r['rel'], r['rel_se']
        res.append(row)
W = pd.DataFrame(res)
pd.set_option('display.width', 200)
print(W.to_string(index=False, float_format=lambda v: '%.3f' % v))
for start in ('scratch', 'pretrained'):
    w = W[W.start == start]
    d = w.rel_DMSP - w.rel_VIIRS
    print(start, 'countries', len(w), 'rel MSE change DMSP minus VIIRS: median %.3f; DMSP better in %d of %d' % (d.median(), (d < 0).sum(), len(w)),
          '| median rel DMSP %.3f VIIRS %.3f' % (w.rel_DMSP.median(), w.rel_VIIRS.median()))
W.to_csv(OUT / 'within_country_period.csv', index=False)
