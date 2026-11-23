# Murphy decomposition of the change in MSE per country: bias^2, ranking (var_y*(1-r^2)), scale terms.
import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
from subset_gains import OUT
sys.path.insert(0, str(HERE))
from subset_gains import NINE, ENC
D = pd.read_pickle(OUT / 'D.pkl')
y = D.iwi.to_numpy(float)
def decomp(p, yy):
    b = p.mean() - yy.mean(); r = np.corrcoef(p, yy)[0, 1]; sy, sp = yy.std(), p.std()
    return dict(mse=np.mean((p - yy) ** 2), bias2=b ** 2, rank=sy ** 2 * (1 - r ** 2), scale=(sp - r * sy) ** 2)
rows = []
for start in ('scratch', 'pretrained'):
    for c, g in D.groupby('country'):
        idx = g.index.to_numpy(); yy = y[idx]
        acc = {k: [] for k in ('mse', 'bias2', 'rank', 'scale')}
        for m in NINE:
            d0 = decomp(D.loc[idx, f'ten_models|{m}|L+NL|{start}'].to_numpy(float), yy)
            for e in ENC:
                d1 = decomp(D.loc[idx, f'ten_models|{m}|L+NL+{e}|{start}'].to_numpy(float), yy)
                for k in acc: acc[k].append((d1[k] - d0[k], d0[k]))
        row = dict(country=c, start=start)
        for k in acc:
            row[f'd_{k}'] = np.mean([a for a, _ in acc[k]]); row[f'b_{k}'] = np.mean([b for _, b in acc[k]])
        # satclip vs other eight
        def rel(models):
            return np.mean([np.mean((D.loc[idx, f'ten_models|{m}|L+NL+{e}|{start}'] - yy) ** 2) / np.mean((D.loc[idx, f'ten_models|{m}|L+NL|{start}'] - yy) ** 2) - 1 for m in models for e in ENC])
        row['rel_satclip'] = rel(['satclip']); row['rel_other8'] = rel([m for m in NINE if m != 'satclip'])
        rows.append(row)
T = pd.DataFrame(rows)
T['rel'] = T.d_mse / T.b_mse
T['share_bias'] = T.d_bias2 / T.d_mse
T.to_csv(OUT / 'country_decomp.csv', index=False)
pd.set_option('display.width', 250)
for start in ('scratch', 'pretrained'):
    t = T[T.start == start].sort_values('rel')
    print('=====', start)
    print(t[['country', 'b_mse', 'rel', 'd_mse', 'd_bias2', 'd_rank', 'd_scale', 'rel_satclip', 'rel_other8']].to_string(index=False, float_format=lambda v: '%.2f' % v))
    worse = t[t.rel > 0.03]; better = t[t.rel < -0.03]
    print('countries with MSE up >3%%: %d; bias term > half of the increase in %d' % (len(worse), (worse.d_bias2 > 0.5 * worse.d_mse).sum()))
    print('countries with MSE down >3%%: %d; bias term > half of the decrease in %d; ranking term > half in %d' % (len(better), (better.d_bias2 < 0.5 * better.d_mse).sum(), (better.d_rank < 0.5 * better.d_mse).sum()))
