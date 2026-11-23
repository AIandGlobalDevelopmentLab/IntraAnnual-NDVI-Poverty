import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
from subset_gains import OUT
sys.path.insert(0, str(HERE))
from gainlib import Gains
from subset_gains import NINE, ENC
D = pd.read_pickle(OUT / 'D.pkl')
G = Gains(D, 'iwi')
cases = ['Nigeria', 'Ghana', 'Mozambique', 'Namibia', 'Kenya', 'Egypt', 'Chad', 'Liberia']
rows = []
for c in cases:
    m = (D.country == c).to_numpy()
    g = D[m]
    for start in ('scratch', 'pretrained'):
        r = G.nine(start, m)
        row = dict(country=c, start=start, n=int(m.sum()), rural=g.rural.mean(), dmsp=(g.nl_sensor == 'DMSP').mean(),
                   years=f'{g.year.min()}-{g.year.max()}', iwi=f'{g.iwi.mean():.1f} ({g.iwi.std():.1f})',
                   climate=', '.join(f'{k.split(" (")[0]} {v:.0%}' for k, v in g.climate.value_counts(normalize=True).items() if v >= 0.1),
                   base_r2=r['base_r2'], dR2=r['dR2'], rel=r['rel'], rel_lo=r['rel_ci'][0], rel_hi=r['rel_ci'][1])
        # rural / urban and period splits
        for lab, mm in (('rural', m & D.rural.to_numpy()), ('urban', m & ~D.rural.to_numpy()),
                        ('DMSP', m & (D.nl_sensor == 'DMSP').to_numpy()), ('VIIRS', m & (D.nl_sensor == 'VIIRS').to_numpy())):
            if mm.sum() >= 60:
                q = G.nine(start, mm); row[f'rel_{lab}'] = q['rel']; row[f'ci_{lab}'] = q['rel_ci']
        # bias of the nine with and without NDVI (mean over models and the four encodings)
        b0 = np.mean([np.mean(G.P[(start, mo, 'L+NL')][m] - G.y[m]) for mo in NINE])
        b1 = np.mean([np.mean(G.P[(start, mo, f'L+NL+{e}')][m] - G.y[m]) for mo in NINE for e in ENC])
        row['bias_base'], row['bias_ndvi'] = b0, b1
        # correlation-based: squared correlation of nine-mean prediction with IWI, base and NDVI
        p0 = np.mean([G.P[(start, mo, 'L+NL')][m] for mo in NINE], 0)
        p1 = np.mean([G.P[(start, mo, f'L+NL+{e}')][m] for mo in NINE for e in ENC], 0)
        row['r2corr_base'] = np.corrcoef(p0, G.y[m])[0, 1] ** 2; row['r2corr_ndvi'] = np.corrcoef(p1, G.y[m])[0, 1] ** 2
        rows.append(row)
    print(c, flush=True)
K = pd.DataFrame(rows)
K.to_pickle(OUT / 'cases.pkl'); K.to_csv(OUT / 'case_studies.csv', index=False)
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
print(K.drop(columns=[c for c in K.columns if c.startswith('ci_')]).to_string(index=False, float_format=lambda v: '%.3f' % v))
# Nigeria: north / middle / south
for lab, lo, hi in (('North (>= 10 N)', 10, 99), ('Middle belt (8-10 N)', 8, 10), ('South (< 8 N)', -99, 8)):
    m = ((D.country == 'Nigeria') & (D.lat >= lo) & (D.lat < hi)).to_numpy()
    for start in ('scratch', 'pretrained'):
        r = G.nine(start, m)
        g = D[m]
        print('Nigeria', lab, start, 'n', m.sum(), 'rural %.2f' % g.rural.mean(), 'IWI %.1f (%.1f)' % (g.iwi.mean(), g.iwi.std()),
              'climate', dict(g.climate.value_counts(normalize=True).round(2)), 'base %.3f dR2 %.3f rel %.3f [%.3f, %.3f]' % (r['base_r2'], r['dR2'], r['rel'], *r['rel_ci']))
