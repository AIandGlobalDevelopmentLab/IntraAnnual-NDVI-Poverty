"""Where does intra-annual NDVI help? Gains of the NDVI encodings in subsets of clusters (IWI and NL experiments).

For each target (iwi, nl), start, encoding (and the mean of the four), and subset of clusters, computes
- base R2 of the nine newer architectures (mean over models, within the subset),
- nine-model mean dR2 within the subset, with a 95% interval (1,000 paired Poisson(1) resamples of spatial groups),
- nine-model mean relative change in MSE (dMSE / MSE of the reference input), with interval,
- dR2 of VGG-16-BN and of the MLP, with intervals.
Subsets: all, rural/urban, sensor period, period x area, Koppen climate group, NL tercile within period (lights),
and every country. Writes subset_gains.csv, subset_gains_clusters.csv and D.pkl (used by the other scripts) to $ANALYSIS_OUT
(default results/analysis). The last two hold cluster-level DHS data: keep them out of the repository.
Usage: python subset_gains.py   (needs numpy, pandas, scipy and kgcpy: pip install kgcpy)
"""
import sys, numpy as np, pandas as pd, kgcpy
from pathlib import Path
import os
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
# evaluation folders written by notebook 16 of each experiment, and the folder for the outputs of these scripts
IWI_EVAL = Path(os.environ.get('IWI_EVAL_DIR', REPO / 'results' / 'iwi'))
NL_EVAL = Path(os.environ.get('NL_EVAL_DIR', REPO / 'results' / 'nl'))
OUT = Path(os.environ.get('ANALYSIS_OUT', REPO / 'results' / 'analysis')); OUT.mkdir(parents=True, exist_ok=True)

IWI = IWI_EVAL / 'eval_test_predictions.csv.gz'
NL = NL_EVAL / 'eval_test_predictions.csv.gz'
NINE = ['resnet18', 'resnet34', 'resnet50', 'swin', 'unet', 'vit', 'satclip', 'mosaiks', 'olmoearth']
BASEL = ['vgg16bn', 'mlp']
ENC = ['wavelet', 'DFT', 'peak:pixel11/wavelet6', 'TT:cyclic']
B, SEED = 1000, 20261117

def koppen(lat, lon):
    try:
        return kgcpy.lookupCZ(float(lat), float(lon))
    except Exception:
        return 'NA'

def climate_group(z):
    if z.startswith('BW'): return 'Arid desert (BW)'
    if z.startswith('BS'): return 'Semi-arid steppe (BS)'
    if z in ('Af', 'Am'): return 'Tropical humid (Af, Am)'
    if z in ('Aw', 'As'): return 'Tropical savanna (Aw, As)'
    if z.startswith('C'): return 'Temperate (C)'
    return 'Other'

def load():
    I = pd.read_csv(IWI)
    N = pd.read_csv(NL, usecols=lambda c: c in ('cluster_id', 'nl') or c.startswith('ten_models|'))
    N = N.rename(columns={c: 'NL::' + c for c in N.columns if c.startswith('ten_models|')})
    D = I.merge(N, on='cluster_id', how='inner')
    assert len(D) == len(I) == len(N)
    D['rural'] = D['rural'].astype(str).str.lower().eq('true')
    D['kz'] = [koppen(a, b) for a, b in zip(D['lat'], D['lon'])]
    D['climate'] = D['kz'].map(climate_group)
    # NL terciles within each sensor period (dark / middle / bright), from the NL target
    D['lights'] = ''
    for s, g in D.groupby('nl_sensor'):
        q = g['nl'].rank(pct=True)
        D.loc[g.index, 'lights'] = np.where(q <= 1 / 3, 'dark third', np.where(q <= 2 / 3, 'middle third', 'bright third'))
    return D

def subsets(D):
    S = [('all', 'all', np.ones(len(D), bool))]
    S += [('area', a, (D['rural'] == (a == 'rural')).to_numpy()) for a in ('rural', 'urban')]
    S += [('period', p, (D['nl_sensor'] == p).to_numpy()) for p in ('DMSP', 'VIIRS')]
    S += [('period x area', f'{p} {a}', ((D['nl_sensor'] == p) & (D['rural'] == (a == 'rural'))).to_numpy())
          for p in ('DMSP', 'VIIRS') for a in ('rural', 'urban')]
    S += [('climate', c, (D['climate'] == c).to_numpy()) for c in sorted(D['climate'].unique())]
    S += [('lights', l, (D['lights'] == l).to_numpy()) for l in ('dark third', 'middle third', 'bright third')]
    S += [('lights x area', f'{l} {a}', ((D['lights'] == l) & (D['rural'] == (a == 'rural'))).to_numpy())
          for l in ('dark third', 'bright third') for a in ('rural', 'urban')]
    S += [('country', c, (D['country'] == c).to_numpy()) for c in sorted(D['country'].unique())]
    return S

def run(D, target, out_rows):
    y = D[target].to_numpy(float)
    pre = '' if target == 'iwi' else 'NL::'
    base = 'L+NL' if target == 'iwi' else 'L'
    g = pd.factorize(D['group'])[0]; G = g.max() + 1
    rng = np.random.default_rng(SEED)
    Wg = np.vstack([np.ones((1, G)), rng.poisson(1.0, size=(B, G))]).astype(np.float64)   # row 0 = point estimate
    for start in ('scratch', 'pretrained'):
        cols, names = [], []
        for m in NINE + BASEL:
            for c in [base] + [f'{base}+{e}' for e in ENC]:
                cols.append(D[f'{pre}ten_models|{m}|{c}|{start}'].to_numpy(float)); names.append((m, c))
        Pm = np.column_stack(cols)                      # n x runs
        for stype, sname, mask in subsets(D):
            if mask.sum() < 30: continue
            sub = mask[:, None] & np.isfinite(Pm)
            # paired masks per (model, encoding): clusters both runs predict
            res = {}
            # group sums for SST
            gi = g[mask]
            def gsum(v):
                return np.bincount(gi, weights=v, minlength=G)
            sw, swy, swyy = Wg @ gsum(np.ones(mask.sum())), Wg @ gsum(y[mask]), Wg @ gsum(y[mask] ** 2)
            sst = swyy - swy ** 2 / sw
            for mi, m in enumerate(NINE + BASEL):
                bi = names.index((m, base))
                for e in ENC:
                    ai = names.index((m, f'{base}+{e}'))
                    k = mask & np.isfinite(Pm[:, ai]) & np.isfinite(Pm[:, bi])
                    gk = g[k]
                    ea, eb = (y[k] - Pm[k, ai]) ** 2, (y[k] - Pm[k, bi]) ** 2
                    A, Bb = Wg @ np.bincount(gk, weights=ea, minlength=G), Wg @ np.bincount(gk, weights=eb, minlength=G)
                    if k.sum() == mask.sum():
                        sst_k = sst
                    else:
                        swk, swyk, swyyk = (Wg @ np.bincount(gk, weights=np.ones(k.sum()), minlength=G),
                                            Wg @ np.bincount(gk, weights=y[k], minlength=G),
                                            Wg @ np.bincount(gk, weights=y[k] ** 2, minlength=G))
                        sst_k = swyyk - swyk ** 2 / swk
                    res[(m, e)] = dict(dR2=(Bb - A) / sst_k, rel=(A - Bb) / Bb, r2b=1 - Bb / sst_k)
            for e in ENC + ['mean4']:
                es = ENC if e == 'mean4' else [e]
                nine_d = np.mean([np.mean([res[(m, x)]['dR2'] for x in es], 0) for m in NINE], 0)
                nine_r = np.mean([np.mean([res[(m, x)]['rel'] for x in es], 0) for m in NINE], 0)
                nine_b = np.mean([res[(m, ENC[0])]['r2b'] for m in NINE], 0)
                row = dict(target=target, start=start, enc=e, stype=stype, subset=sname, n=int(mask.sum()),
                           base_r2=nine_b[0], dR2=nine_d[0], dR2_lo=np.percentile(nine_d[1:], 2.5),
                           dR2_hi=np.percentile(nine_d[1:], 97.5), dR2_se=np.std(nine_d[1:]),
                           relMSE=nine_r[0], rel_lo=np.percentile(nine_r[1:], 2.5), rel_hi=np.percentile(nine_r[1:], 97.5),
                           rel_se=np.std(nine_r[1:]))
                for bl in BASEL:
                    d = np.mean([res[(bl, x)]['dR2'] for x in es], 0)
                    row[f'{bl}_dR2'], row[f'{bl}_lo'], row[f'{bl}_hi'] = d[0], np.percentile(d[1:], 2.5), np.percentile(d[1:], 97.5)
                    row[f'{bl}_base_r2'] = res[(bl, ENC[0])]['r2b'][0]
                out_rows.append(row)
        print(target, start, 'done', flush=True)

if __name__ == '__main__':
    D = load()
    D.to_pickle(OUT / 'D.pkl')
    D[['cluster_id', 'country', 'lat', 'lon', 'year', 'rural', 'nl_sensor', 'kz', 'climate', 'lights', 'iwi', 'nl']].to_csv(
        OUT / 'subset_gains_clusters.csv', index=False)
    rows = []
    for t in ('iwi', 'nl'):
        run(D, t, rows)
    pd.DataFrame(rows).to_csv(OUT / 'subset_gains.csv', index=False)
