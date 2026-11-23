import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
from subset_gains import OUT
sys.path.insert(0, str(HERE))
from subset_gains import NINE
from scipy.stats import spearmanr, chi2
D = pd.read_pickle(OUT / 'D.pkl')
R = pd.read_csv(OUT / 'subset_gains.csv')
C = R[(R.stype == 'country')]
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)

def het(x, se):
    w = 1 / se ** 2; xb = (w * x).sum() / w.sum(); Q = (w * (x - xb) ** 2).sum(); k = len(x)
    I2 = max(0.0, (Q - (k - 1)) / Q); tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    return dict(k=k, mean_w=xb, Q=Q, p=chi2.sf(Q, k - 1), I2=I2, tau=np.sqrt(tau2), median=np.median(x), sd=np.std(x, ddof=1),
                typical_se=np.sqrt(np.mean(se ** 2)))

print('=== heterogeneity across 36 countries (nine-model mean gain, mean of four encodings)')
for t in ('iwi', 'nl'):
    for st in ('scratch', 'pretrained'):
        x = C[(C.target == t) & (C.start == st) & (C.enc == 'mean4')].dropna(subset=['rel', 'rel_se'] if 'rel' in C else ['relMSE'])
        for col, se in (('relMSE', 'rel_se'), ('dR2', 'dR2_se')):
            h = het(x[col].to_numpy(), x[se].to_numpy())
            print(t, st, col, ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in h.items()))

print('=== consistency of country gains: Spearman across encodings (IWI, nine-mean dR2)')
for t in ('iwi', 'nl'):
    for st in ('scratch', 'pretrained'):
        piv = C[(C.target == t) & (C.start == st) & (C.enc != 'mean4')].pivot(index='subset', columns='enc', values='relMSE')
        cs = piv.corr(method='spearman').to_numpy(); off = cs[np.triu_indices(4, 1)]
        print(t, st, 'pairwise Spearman across encodings: min %.2f median %.2f max %.2f' % (off.min(), np.median(off), off.max()))
    a = C[(C.target == t) & (C.enc == 'mean4') & (C.start == 'scratch')].set_index('subset').relMSE
    b = C[(C.target == t) & (C.enc == 'mean4') & (C.start == 'pretrained')].set_index('subset').relMSE
    print(t, 'scratch vs pretrained Spearman %.2f' % spearmanr(a, b.loc[a.index])[0])
ai = C[(C.target == 'iwi') & (C.enc == 'mean4') & (C.start == 'scratch')].set_index('subset').relMSE
an = C[(C.target == 'nl') & (C.enc == 'mean4') & (C.start == 'scratch')].set_index('subset').relMSE
print('IWI vs NL country gains (scratch) Spearman %.2f' % spearmanr(ai, an.loc[ai.index])[0])

# country covariates
rows = []
yb = {}
for c, g in D.groupby('country'):
    base_pred = np.mean([g[f'ten_models|{m}|L+NL|scratch'] for m in NINE], 0)
    rows.append(dict(country=c, n=len(g), rural=g.rural.mean(), iwi_mean=g.iwi.mean(), iwi_sd=g.iwi.std(),
                     dmsp=(g.nl_sensor == 'DMSP').mean(), savanna=(g.climate == 'Tropical savanna (Aw, As)').mean(),
                     steppe=(g.climate == 'Semi-arid steppe (BS)').mean(), arid=(g.climate == 'Arid desert (BW)').mean(),
                     temperate=(g.climate == 'Temperate (C)').mean(), humid=(g.climate == 'Tropical humid (Af, Am)').mean(),
                     abslat=g.lat.abs().mean(), bright=(g.lights == 'bright third').mean(),
                     bias=float(np.mean(base_pred - g.iwi)), nl_mean=g.nl.mean()))
X = pd.DataFrame(rows).set_index('country')
for st in ('scratch', 'pretrained'):
    x = C[(C.target == 'iwi') & (C.start == st) & (C.enc == 'mean4')].set_index('subset')
    X[f'rel_{st}'] = x.relMSE; X[f'dR2_{st}'] = x.dR2; X[f'base_r2_{st}'] = x.base_r2; X[f'se_{st}'] = x.rel_se
    xn = C[(C.target == 'nl') & (C.start == st) & (C.enc == 'mean4')].set_index('subset')
    X[f'nl_rel_{st}'] = xn.relMSE
X['abs_bias'] = X.bias.abs()
X['seasonal'] = X.savanna + X.steppe
X.to_csv(OUT / 'country_covariates.csv')
print('=== Spearman of IWI country gain (relative MSE change, negative = better) with covariates')
for st in ('scratch', 'pretrained'):
    out = []
    for v in ['n', 'rural', 'iwi_mean', 'iwi_sd', 'dmsp', 'savanna', 'steppe', 'seasonal', 'arid', 'temperate', 'humid', 'abslat', 'bright', 'abs_bias', f'base_r2_{st}']:
        r, p = spearmanr(X[v], X[f'rel_{st}'])
        out.append((v, r, p))
    print(st, '; '.join(f'{v} {r:+.2f} (p {p:.3f})' for v, r, p in out))
import numpy.linalg as la
for st in ('scratch', 'pretrained'):
    y = X[f'rel_{st}'].to_numpy(); w = 1 / X[f'se_{st}'].to_numpy() ** 2
    for cols in (['dmsp', 'seasonal'], ['dmsp', 'seasonal', 'iwi_sd'], ['dmsp', 'seasonal', 'rural', f'base_r2_{st}']):
        A = np.column_stack([np.ones(len(X))] + [X[c].to_numpy() for c in cols])
        sw = np.sqrt(w); coef, *_ = la.lstsq(A * sw[:, None], y * sw, rcond=None)
        resid = y - A @ coef; ss_res = (w * resid ** 2).sum(); ss_tot = (w * (y - (w * y).sum() / w.sum()) ** 2).sum()
        # coefficient SEs (weighted LS, scaled by residual variance)
        s2 = ss_res / (len(y) - A.shape[1]); cov = s2 * la.inv((A * w[:, None]).T @ A); se = np.sqrt(np.diag(cov))
        print(st, cols, 'R2_w %.2f' % (1 - ss_res / ss_tot), ' '.join(f'{c}={b:+.3f}({s:.3f})' for c, b, s in zip(['const'] + cols, coef, se)))
print(X[['n', 'rural', 'iwi_sd', 'dmsp', 'seasonal', 'arid', 'temperate', 'humid', 'base_r2_scratch', 'rel_scratch', 'dR2_scratch', 'se_scratch', 'rel_pretrained', 'nl_rel_scratch']].sort_values('rel_scratch').to_string(float_format=lambda v: '%.3f' % v))
