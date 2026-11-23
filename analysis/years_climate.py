import sys, numpy as np, pandas as pd
from pathlib import Path
HERE = Path(__file__).resolve().parent
from subset_gains import OUT
sys.path.insert(0, str(HERE))
from gainlib import Gains
D = pd.read_pickle(OUT / 'D.pkl')
out = []
for t in ('iwi', 'nl'):
    G = Gains(D, t)
    yb = pd.cut(D['year'], [2000, 2002, 2008, 2011, 2012, 2013, 2016, 2019, 2023],
                labels=['2001', '2003-08', '2009-11', '2012', '2013', '2014-16', '2017-19', '2020-23'])
    subs = [('year', str(l), (yb == l).to_numpy()) for l in yb.cat.categories]
    for c in ['Tropical savanna (Aw, As)', 'Semi-arid steppe (BS)', 'Temperate (C)', 'Arid desert (BW)', 'Tropical humid (Af, Am)']:
        for p in ('DMSP', 'VIIRS'):
            subs.append(('climate x period', f'{c} | {p}', ((D['climate'] == c) & (D['nl_sensor'] == p)).to_numpy()))
        for a in ('rural', 'urban'):
            subs.append(('climate x area', f'{c} | {a}', ((D['climate'] == c) & (D['rural'] == (a == 'rural'))).to_numpy()))
    for start in ('scratch', 'pretrained'):
        for st, name, mask in subs:
            if mask.sum() < 50: continue
            r = G.nine(start, mask)
            out.append(dict(target=t, start=start, stype=st, subset=name, n=r['n'], base_r2=r['base_r2'], dR2=r['dR2'],
                            lo=r['dR2_ci'][0], hi=r['dR2_ci'][1], rel=r['rel'], rlo=r['rel_ci'][0], rhi=r['rel_ci'][1]))
        print(t, start, flush=True)
O = pd.DataFrame(out); O.to_csv(OUT / 'years_climate.csv', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 300)
print(O.to_string(index=False, float_format=lambda v: '%.3f' % v))
