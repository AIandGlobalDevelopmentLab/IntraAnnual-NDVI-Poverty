"""Shared helpers: nine-model mean gains of the NDVI encodings for IWI in any subset of clusters, with paired bootstrap."""
import numpy as np, pandas as pd
from subset_gains import load, NINE, BASEL, ENC, B, SEED

class Gains:
    def __init__(self, D, target='iwi'):
        self.D, self.target = D, target
        self.y = D[target].to_numpy(float)
        self.base = 'L+NL'                              # reference input: Landsat + nighttime lights
        g = pd.factorize(D['group'])[0]; self.g, self.G = g, g.max() + 1
        rng = np.random.default_rng(SEED)
        self.Wg = np.vstack([np.ones((1, self.G)), rng.poisson(1.0, size=(B, self.G))]).astype(np.float64)
        self.P = {}
        for start in ('scratch', 'pretrained'):
            for m in NINE + BASEL:
                for c in [self.base] + [f'{self.base}+{e}' for e in ENC]:
                    self.P[(start, m, c)] = D[f'ten_models|{m}|{c}|{start}'].to_numpy(float)

    def _w(self, v, k):
        return self.Wg @ np.bincount(self.g[k], weights=v, minlength=self.G)

    def model(self, start, m, e, mask):
        """dR2, relative dMSE, base R2 (row 0 = estimate, rows 1.. = bootstrap) for one model and encoding."""
        a, b = self.P[(start, m, f'{self.base}+{e}')], self.P[(start, m, self.base)]
        k = mask & np.isfinite(a) & np.isfinite(b)
        y = self.y[k]
        A, Bb = self._w((y - a[k]) ** 2, k), self._w((y - b[k]) ** 2, k)
        sw, sy, syy = self._w(np.ones(k.sum()), k), self._w(y, k), self._w(y ** 2, k)
        sst = syy - sy ** 2 / sw
        return (Bb - A) / sst, (A - Bb) / Bb, 1 - Bb / sst, k.sum()

    def nine(self, start, mask, encs=None):
        encs = encs or ENC
        d, r, b = [], [], []
        for m in NINE:
            dd, rr, bb = [], [], None
            for e in encs:
                x = self.model(start, m, e, mask); dd.append(x[0]); rr.append(x[1]); bb = x[2]
            d.append(np.mean(dd, 0)); r.append(np.mean(rr, 0)); b.append(bb)
        d, r, b = np.mean(d, 0), np.mean(r, 0), np.mean(b, 0)
        pct = lambda v: (np.nanpercentile(v[1:], 2.5), np.nanpercentile(v[1:], 97.5))
        return dict(n=int(mask.sum()), base_r2=b[0], dR2=d[0], dR2_ci=pct(d), dR2_se=np.nanstd(d[1:]),
                    rel=r[0], rel_ci=pct(r), rel_se=np.nanstd(r[1:]), _d=d, _r=r)

    def baseline(self, start, bl, mask, encs=None):
        encs = encs or ENC
        d = np.mean([self.model(start, bl, e, mask)[0] for e in encs], 0)
        r = np.mean([self.model(start, bl, e, mask)[1] for e in encs], 0)
        return dict(dR2=d[0], dR2_ci=(np.nanpercentile(d[1:], 2.5), np.nanpercentile(d[1:], 97.5)), rel=r[0])
