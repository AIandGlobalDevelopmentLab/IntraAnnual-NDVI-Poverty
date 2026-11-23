"""LaTeX table bodies of the paper and its supplement (see paper/README.md for the table numbers)."""
from common import *
G = pd.read_csv(REV / 'subset_gains.csv')
OUT = str(HERE / 'tables')
import os; os.makedirs(OUT, exist_ok=True)

def m(x, d=3, sign=True, star=False):
    """A number in math mode: $+0.008$, $-0.006$, 0.004; values that round to zero lose their sign."""
    if x is None or not np.isfinite(x): return '--'
    s = f'{x:+.{d}f}' if sign else f'{x:.{d}f}'
    if float(s) == 0: s = f'{0:.{d}f}'
    st = '^{*}' if star else ''
    if s.startswith('-') or s.startswith('+') or star: return f'${s}{st}$'
    return s
def iv(lo, hi, d=3):
    return f'[{m(lo, d, False)}, {m(hi, d, False)}]'
def e3(x, lo, hi, star=False, d=3):
    return f'{m(x, d)} {iv(lo, hi, d)}' + ('$^{*}$' if star else '')
def pct(x, d=1):
    return m(100 * x, d)
def sig(lo, hi): return (lo > 0) or (hi < 0)
SHORT = {'wavelet': 'Wavelet', 'DFT': 'Fourier', 'peak:pixel11/wavelet6': 'Peak-aligned', 'TT:cyclic': 'Transformer', 'mean4': 'Mean of four'}
PANEL = {'scratch': 'From scratch', 'pretrained': 'Pretrained'}
def two(x, lo, hi, star=False, extra=''):
    """Estimate on the first line, interval on the second."""
    st = '^{*}' if star else ''
    first = m(x, star=star) + (f' {extra}' if extra else '')
    return '\\makecell[l]{' + first + '\\\\{' + iv(lo, hi) + '}}'
def panel(s, n):
    return f'\\multicolumn{{{n}}}{{@{{}}l}}{{\\textit{{{PANEL[s]}}}}} \\\\'
START = {'scratch': 'Scratch', 'pretrained': 'Pretrained'}

# ------------------------------------------------------------------ accuracy (X.1)
def t_accuracy(t):
    M = pd.read_csv(ev(t, 'overall_metrics.csv')); M = M[(M.family == 'ten_models') & (M.combo == REF[t])]
    P = pd.read_csv(ev(t, 'overall_pretrained_minus_scratch.csv')); P = P[(P.family == 'ten_models') & (P.combo == REF[t])]
    rows = []
    for k in LAB:
        r = {}
        for s in ['pretrained', 'scratch']:
            x = M[(M.model == k) & (M.start == s)].iloc[0]
            r[s] = [f'{x.R2:.3f}', f'{x.RMSE:.2f}', f'{x.MAE:.2f}']
            r[s + 'R2'] = x.R2
        p = P[P.model == k]
        if len(p) and k not in ('mlp', 'mosaiks', 'satclip_only'):
            p = p.iloc[0]; r['pms'] = m(p.dR2, star=sig(p['dR2 low'], p['dR2 high']))
        else:
            r['pms'] = '--'
        rows.append((k, r))
    rows.sort(key=lambda z: -z[1]['pretrainedR2'])
    body = '\n'.join(f'{LAB[k]} & ' + ' & '.join(r['pretrained']) + ' & ' + ' & '.join(r['scratch']) + f' & {r["pms"]} \\\\'
                     for k, r in rows)
    return body

# ------------------------------------------------------------------ newer vs baselines (X.2)
def t_newer(t):
    S = pd.read_csv(ev(t, 'summary_bootstrap.csv')); S = S[S.subset == 'all']
    V = pd.read_csv(ev(t, 'overall_vs_baseline_models_summary.csv'))
    lines = []
    for e in [None] + ENC:
        c = combo(t, e); cells = []
        for b in ['vgg16bn', 'mlp']:
            for s in ['pretrained', 'scratch']:
                q = S[(S.start == s) & (S.combo == c) & (S.quantity == f'nine minus {b}, same input')].iloc[0]
                v = V[(V.start == s) & (V['baseline model'] == b) & (V.against == 'same input') & (V.combo == c) & (V.family == 'ten_models')].iloc[0]
                cells.append(two(q.estimate, q.low, q.high, extra=f'({int(v["sig. better"])}/{int(v["sig. worse"])})'))
        for k in ['eight: mean', 'vgg16bn']:
            q = S[(S.start == 'pretrained minus scratch') & (S.combo == c) & (S.quantity == k)].iloc[0]
            cells.append(two(q.estimate, q.low, q.high))
        name = 'Landsat + NL' if e is None else '+ ' + SHORT[e]
        lines.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

# ------------------------------------------------------------------ encodings (X.3)
IM = pd.read_csv(REV / 'iwi_mae.csv')
def t_enc(t):
    S = pd.read_csv(ev(t, 'summary_bootstrap.csv')); S = S[S.subset == 'all']
    M = pd.read_csv(ev(t, 'overall_metrics.csv')); M = M[M.family == 'ten_models']
    lines = []
    for s in ['scratch', 'pretrained']:
        if s == 'pretrained': lines.append('\\addlinespace')
        lines.append(panel(s, 6))
        for e in ENC:
            c = combo(t, e)
            q = S[(S.start == s) & (S.combo == c) & (S.quantity == f'nine: mean gain over {REF[t]}')].iloc[0]
            mm = M[(M.start == s) & (M.combo == c) & (M.model.isin(NINE))]
            b = (mm['dR2 low'] > 0).sum(); w = (mm['dR2 high'] < 0).sum()
            bq = ((mm['dR2 low'] > 0) & (mm.q < 0.05)).sum(); wq = ((mm['dR2 high'] < 0) & (mm.q < 0.05)).sum()
            a = IM[(IM.start == s) & (IM.enc == e) & (IM.subset == 'all')].iloc[0]; mae = two(a.dMAE, a.lo, a.hi)
            cells = [SHORT[e], two(q.estimate, q.low, q.high), f'{b} / {w} ({bq} / {wq})', mae]
            for k in ['vgg16bn', 'mlp']:
                x = M[(M.start == s) & (M.combo == c) & (M.model == k)].iloc[0]
                cells.append(two(x.dR2, x['dR2 low'], x['dR2 high'], star=x.q < 0.05))
            lines.append(' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

# ------------------------------------------------------------------ subgroups (X.4 area, X.6 period)
def t_sub(t, stype):
    subs = ['rural', 'urban'] if stype == 'area' else ['DMSP', 'VIIRS']
    B = pd.read_csv(ev(t, 'rural_urban_continental_vs_baseline.csv')); B = B[B.family == 'ten_models']
    R = pd.read_csv(ev(t, 'rural_urban_country_consensus.csv')); R = R[R.family == 'ten_models']
    ncol = 7 if stype == 'area' else 6
    lines = []
    for s in ['scratch', 'pretrained']:
        if s == 'pretrained': lines.append('\\addlinespace')
        lines.append(panel(s, ncol))
        for e in ENC + ['mean4']:
            c = combo(t, e) if e != 'mean4' else None
            g = {a: G[(G.target == t) & (G.start == s) & (G.enc == e) & (G.stype == stype) & (G.subset == a)].iloc[0] for a in subs}
            cells = [SHORT[e]]
            cells += [two(g[a].dR2, g[a].dR2_lo, g[a].dR2_hi) for a in subs]
            cells.append(' / '.join(m(100 * g[a].relMSE, 1, star=sig(g[a].rel_lo, g[a].rel_hi)) for a in subs))
            for k in ['vgg16bn', 'mlp']:
                vals = []
                for a in subs:
                    if stype == 'area' and e != 'mean4':
                        x = B[(B.start == s) & (B.combo == c) & (B.model == k) & (B.area == a)].iloc[0]
                        v, lo, hi = x.dR2, x['dR2 low'], x['dR2 high']
                    else:
                        v, lo, hi = g[a][f'{k}_dR2'], g[a][f'{k}_lo'], g[a][f'{k}_hi']
                    vals.append(m(v, star=sig(lo, hi)))
                cells.append(' / '.join(vals))
            if stype == 'area':
                if e == 'mean4':
                    cells.append('--')
                else:
                    cc = []
                    for a in subs:
                        p, gg, l = clear_counts(R[(R.start == s) & (R.combo == c) & (R.area == a)])
                        cc.append(f'{gg}/{l}')
                    cells.append('; '.join(cc))
            lines.append(' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

# ------------------------------------------------------------------ countries (X.5)
def t_countries(t):
    C = pd.read_csv(ev(t, 'country_consensus.csv')); C = C[(C.family == 'ten_models') & (C.country != 'all 36 countries')]
    V = pd.read_csv(ev(t, 'country_vs_baseline.csv')); V = V[(V.family == 'ten_models') & (V.country != 'all 36 countries')]
    lines = []
    for s in ['scratch', 'pretrained']:
        if s == 'pretrained': lines.append('\\addlinespace')
        lines.append(panel(s, 6))
        for e in ENC:
            c = combo(t, e)
            p, g, l = clear_counts(C[(C.start == s) & (C.combo == c)])
            cells = [SHORT[e], str(p), str(g), str(l)]
            for k in ['vgg16bn', 'mlp']:
                x = V[(V.start == s) & (V.combo == c) & (V.model == k)]
                cells.append(f'{(x.dR2 > 0).sum()} ({(x.verdict == "better").sum()} / {(x.verdict == "worse").sum()})')
            lines.append(' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

def t_subgroups(t):
    """Rural / urban and DMSP / VIIRS rows of the compact subgroup tables (mean of the four encodings)."""
    lines = []
    N = {'rural': '27,461', 'urban': '12,899', 'DMSP': '19,758', 'VIIRS': '20,602'}
    LABS = {'rural': 'Rural', 'urban': 'Urban', 'DMSP': 'DMSP period (to 2013)', 'VIIRS': 'VIIRS period (from 2014)'}
    for s in ['scratch', 'pretrained']:
        if s == 'pretrained': lines.append('\\addlinespace')
        lines.append(panel(s, 6))
        for stype, subs in [('area', ['rural', 'urban']), ('period', ['DMSP', 'VIIRS'])]:
            for a in subs:
                g = G[(G.target == t) & (G.start == s) & (G.enc == 'mean4') & (G.stype == stype) & (G.subset == a)].iloc[0]
                mse = '\\makecell[l]{' + m(100 * g.relMSE, 1) + '\\\\{[' + m(100 * g.rel_lo, 1, False) + ', ' + m(100 * g.rel_hi, 1, False) + ']}}'
                cells = [LABS[a], N[a], two(g.dR2, g.dR2_lo, g.dR2_hi), mse,
                         m(g.vgg16bn_dR2, star=sig(g.vgg16bn_lo, g.vgg16bn_hi)), m(g.mlp_dR2, star=sig(g.mlp_lo, g.mlp_hi))]
                lines.append(' & '.join(cells) + ' \\\\')
    return '\n'.join(lines)

if __name__ == '__main__':
    for t in ['iwi']:
        for name, f in [('acc', t_accuracy), ('newer', t_newer), ('enc', t_enc), ('countries', t_countries)]:
            open(f'{OUT}/tab_{t}_{name}.tex', 'w').write(f(t) + '\n')
        open(f'{OUT}/tab_{t}_area.tex', 'w').write(t_sub(t, 'area') + '\n')
        open(f'{OUT}/tab_{t}_period.tex', 'w').write(t_sub(t, 'period') + '\n')
        open(f'{OUT}/tab_{t}_subgroups.tex', 'w').write(t_subgroups(t) + '\n')
    import glob
    for p in sorted(glob.glob(f'{OUT}/tab_*.tex')):
        print('=====', p); print(open(p).read())
