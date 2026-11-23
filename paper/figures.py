"""Figures of the paper and its supplement (see paper/README.md for the figure numbers)."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from common import *
OUT = str(HERE / 'figures')
import os; os.makedirs(OUT, exist_ok=True)

INK, SECONDARY, MUTED, GRID, AXIS, SURFACE = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7', '#ffffff'
BLUE, ORANGE = '#2a78d6', '#eb6834'
DIVERGING = LinearSegmentedColormap.from_list('delta', ['#e34948', '#f0efec', '#2a78d6'])
plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                     'axes.edgecolor': AXIS, 'axes.labelcolor': SECONDARY, 'axes.titlecolor': INK,
                     'axes.titlesize': 10, 'axes.titlelocation': 'left', 'axes.labelsize': 9, 'font.size': 9,
                     'xtick.color': SECONDARY, 'ytick.color': SECONDARY, 'xtick.labelsize': 8,
                     'ytick.labelsize': 8, 'text.color': INK, 'legend.frameon': False, 'legend.fontsize': 8,
                     'axes.spines.top': False, 'axes.spines.right': False, 'grid.color': GRID,
                     'grid.linewidth': 0.8, 'font.family': 'DejaVu Sans'})
MODELS = ['resnet18', 'resnet34', 'resnet50', 'swin', 'unet', 'vit', 'satclip', 'satclip_only', 'mosaiks', 'olmoearth',
          'vgg16bn', 'mlp']
MLAB = dict(LAB); MLAB['unet'] = 'U-Net'; MLAB['olmoearth'] = 'OlmoEarth'; MLAB['mlp'] = 'MLP (band statistics)'
REFLAB = {'iwi': 'Landsat + NL'}

def forest(ax, y, d, low, high, color, marker='o', s=34, clip=None):
    d, low, high, y = (np.asarray(v, dtype=float) for v in (d, low, high, y))
    has = np.isfinite(low) & np.isfinite(high)
    lo_c, hi_c = low.copy(), high.copy()
    if clip is not None:
        lo_c = np.clip(lo_c, *clip); hi_c = np.clip(hi_c, *clip)
        for yy, l, h in zip(y[has], low[has], high[has]):
            if h > clip[1]: ax.annotate('', xy=(clip[1], yy), xytext=(clip[1] - 0.04 * (clip[1] - clip[0]), yy),
                                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0, mutation_scale=7))
            if l < clip[0]: ax.annotate('', xy=(clip[0], yy), xytext=(clip[0] + 0.04 * (clip[1] - clip[0]), yy),
                                        arrowprops=dict(arrowstyle='-|>', color=color, lw=1.0, mutation_scale=7))
    ax.hlines(y[has], lo_c[has], hi_c[has], color=color, linewidth=1.3)
    sig = has & ((low > 0) | (high < 0))
    ok = np.isfinite(d)
    dd = np.clip(d, *clip) if clip is not None else d
    ax.scatter(dd[ok & sig], y[ok & sig], s=s, marker=marker, color=color, edgecolors=SURFACE, linewidths=0.9, zorder=3)
    ax.scatter(dd[ok & ~sig], y[ok & ~sig], s=s, marker=marker, facecolors=SURFACE, edgecolors=color, linewidths=1.2, zorder=3)

def forest_v(ax, x, d, low, high, color, marker='o', s=30):
    """Vertical version: estimates d at the positions x with vertical 95% intervals."""
    d, low, high, x = (np.asarray(v, dtype=float) for v in (d, low, high, x))
    has = np.isfinite(low) & np.isfinite(high)
    ax.vlines(x[has], low[has], high[has], color=color, linewidth=1.3)
    sig = has & ((low > 0) | (high < 0)); ok = np.isfinite(d)
    ax.scatter(x[ok & sig], d[ok & sig], s=s, marker=marker, color=color, edgecolors=SURFACE, linewidths=0.9, zorder=3)
    ax.scatter(x[ok & ~sig], d[ok & ~sig], s=s, marker=marker, facecolors=SURFACE, edgecolors=color, linewidths=1.2, zorder=3)

FILLED_OPEN = [Line2D([], [], marker='o', linestyle='none', color=SECONDARY, markersize=6, label='filled: the 95% interval excludes 0'),
               Line2D([], [], marker='o', linestyle='none', markerfacecolor=SURFACE, markeredgecolor=SECONDARY,
                      markersize=6, label='open: it holds 0')]

# ---------------------------------------------------------------- 1. encodings, every model (re-export)
def encodings_figure(t):
    M = pd.read_csv(ev(t, 'overall_metrics.csv')); M = M[M.family == 'ten_models']
    lims = []
    for s in ['pretrained', 'scratch']:
        x = M[(M.start == s) & (M.combo != REF[t])]
        lims += [x['dR2 low'].min(), x['dR2 high'].max()]
    lo, hi = np.nanmin(lims), np.nanmax(lims)
    pad = 0.05 * (hi - lo); lo -= pad; hi += pad
    for s in ['pretrained', 'scratch']:
        fig, axes = plt.subplots(1, 4, figsize=(13.5, 4.6), sharey=True, constrained_layout=True)
        y = np.arange(len(MODELS))[::-1]
        for ax, e in zip(axes, ENC):
            x = M[(M.start == s) & (M.combo == combo(t, e))].set_index('model').reindex(MODELS)
            ax.axvline(0, color=SECONDARY, linewidth=0.9)
            ax.grid(axis='x')
            forest(ax, y, x.dR2, x['dR2 low'], x['dR2 high'], BLUE)
            ax.set_xlim(lo, hi)
            ax.set_title(ENCLAB[e], fontsize=9.5)
            ax.set_xlabel(f'$\\Delta R^2$ against {REFLAB[t]}')
            ax.set_yticks(y); ax.set_yticklabels([MLAB[m] for m in MODELS])
            ax.tick_params(axis='y', length=0)
        start = 'Pretrained' if s == 'pretrained' else 'From scratch'
        target = 'IWI'
        fig.suptitle(f'{target}, {start.lower()}: gain of each NDVI encoding over {REFLAB[t]}, all 40,360 clusters',
                     x=0.01, ha='left', fontsize=10.5)
        fig.legend(handles=FILLED_OPEN, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.06))
        fig.savefig(f'{OUT}/{t}_encodings_{s}.png', dpi=200, bbox_inches='tight')
        plt.close(fig)

# ---------------------------------------------------------------- 2. country map (re-export)
CLUSTERS_CSV = REV / 'subset_gains_clusters.csv'   # cluster coordinates (DHS): made by analysis/subset_gains.py, not distributed
CL = pd.read_csv(CLUSTERS_CSV) if CLUSTERS_CSV.exists() else None
def country_map(t, s='scratch'):
    C = pd.read_csv(ev(t, 'country_consensus.csv')); C = C[(C.family == 'ten_models') & (C.start == s) & (C.country != 'all 36 countries')]
    v = np.abs(C['median dR2'].to_numpy()); top = np.ceil(np.quantile(v, 0.95) / 0.005) * 0.005
    norm = Normalize(-top, top)
    fig, axes = plt.subplots(2, 2, figsize=(9.8, 10.2), constrained_layout=True)
    lat, lon = CL.lat.to_numpy(float), CL.lon.to_numpy(float)
    for ax, e in zip(axes.flat, ENC):
        val = CL.country.map(C[C.combo == combo(t, e)].set_index('country')['median dR2']).to_numpy(float)
        ax.scatter(lon, lat, s=5, color=AXIS, linewidths=0, rasterized=True)
        sc = ax.scatter(lon, lat, s=2, c=val, cmap=DIVERGING, norm=norm, linewidths=0, rasterized=True)
        ax.set_aspect(1 / np.cos(np.radians(np.nanmedian(lat))))
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_title(ENCLAB[e], fontsize=9.5)
    fig.colorbar(sc, ax=list(axes.flat), shrink=0.55, extend='both',
                 label='median within-country $\\Delta R^2$ of the nine newer architectures')
    target = 'IWI'
    fig.suptitle(f'{target}, from scratch: gain of each NDVI encoding over {REFLAB[t]} within each country', x=0.01, ha='left', fontsize=10.5)
    fig.savefig(f'{OUT}/{t}_country_map_{s}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------- 3. country forest plot (new)
def country_forest():
    G = pd.read_csv(REV / 'subset_gains.csv')
    c = G[(G.stype == 'country') & (G.enc == 'mean4') & (G.start == 'scratch')]
    I = c[c.target == 'iwi'].set_index('subset')
    order = I.sort_values('relMSE', ascending=False).index.tolist()     # best at the top
    name = {'Congo - Kinshasa': 'DR Congo', 'Côte d’Ivoire': "Côte d'Ivoire"}
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(6.0, 7.0), constrained_layout=True)
    T = I.reindex(order)
    ax.axvline(0, color=SECONDARY, linewidth=0.9); ax.grid(axis='x')
    forest(ax, y, 100 * T.relMSE, 100 * T.rel_lo, 100 * T.rel_hi, BLUE, s=26)
    ax.set_xlabel('relative change in MSE (%)')
    ax.tick_params(axis='y', length=0)
    ax.set_yticks(y); ax.set_yticklabels([f'{name.get(k, k)} ({int(I.loc[k, "n"]):,})' for k in order], fontsize=7.5)
    fig.suptitle('IWI from scratch: change in the within-country MSE when an NDVI encoding is added\n'
                 '(nine newer architectures, mean of the four encodings; negative = more accurate)', x=0.01, ha='left', fontsize=9.5)
    fig.legend(handles=FILLED_OPEN, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.035))
    fig.savefig(f'{OUT}/country_forest_scratch.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------- 4. survey period and climate (new)
CLIM = ['Tropical savanna (Aw, As)', 'Semi-arid steppe (BS)', 'Tropical humid (Af, Am)', 'Arid desert (BW)', 'Temperate (C)']
CLIMLAB = {'Tropical savanna (Aw, As)': 'Savanna\n(Aw, As)', 'Semi-arid steppe (BS)': 'Steppe\n(BS)',
           'Tropical humid (Af, Am)': 'Humid\n(Af, Am)', 'Arid desert (BW)': 'Desert\n(BW)', 'Temperate (C)': 'Temperate\n(C)'}
def period_climate():
    Y = pd.read_csv(REV / 'years_climate.csv')
    bins = ['2003-08', '2009-11', '2012', '2013', '2014-16', '2017-19', '2020-23']
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.1), constrained_layout=True, gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    for s, col, mk, dx in [('scratch', BLUE, 'o', -0.12), ('pretrained', ORANGE, 's', 0.12)]:
        x = Y[(Y.target == 'iwi') & (Y.start == s) & (Y.stype == 'year')].set_index('subset').reindex(bins)
        forest_v(ax, np.arange(len(bins)) + dx, 100 * x.rel, 100 * x.rlo, 100 * x.rhi, col, marker=mk, s=30)
    ax.axhline(0, color=SECONDARY, linewidth=0.9); ax.axvline(3.5, color=MUTED, linestyle='--', linewidth=0.9)
    ax.text(3.4, 6.6, 'DMSP lights', ha='right', va='top', fontsize=8, color=SECONDARY)
    ax.text(3.6, 6.6, 'VIIRS lights', ha='left', va='top', fontsize=8, color=SECONDARY)
    ax.set_xticks(range(len(bins))); ax.set_xticklabels([b.replace('-', '–') for b in bins])
    ax.set_ylabel('relative change in MSE (%)'); ax.set_xlabel('survey year')
    ax.set_title('(a) IWI by survey year', fontsize=10)
    ax.grid(axis='y'); ax.set_ylim(-11, 7)
    # (b) IWI climate x period, scratch
    ax = axes[1]
    for p, col, mk, dx in [('DMSP', BLUE, 'o', -0.12), ('VIIRS', ORANGE, 's', 0.12)]:
        x = Y[(Y.target == 'iwi') & (Y.start == 'scratch') & (Y.stype == 'climate x period')].set_index('subset')
        x = x.reindex([f'{c} | {p}' for c in CLIM])
        forest_v(ax, np.arange(len(CLIM)) + dx, 100 * x.rel, 100 * x.rlo, 100 * x.rhi, col, marker=mk, s=30)
    ax.axhline(0, color=SECONDARY, linewidth=0.9); ax.grid(axis='y')
    ax.set_xticks(range(len(CLIM))); ax.set_xticklabels([CLIMLAB[c] for c in CLIM])
    ax.set_title('(b) IWI by climate zone, from scratch', fontsize=10)
    for a in axes:
        for lab in a.get_xticklabels(): lab.set_fontsize(7.5)
    h = [Line2D([], [], marker='o', linestyle='none', color=BLUE, label='(a): from scratch;  (b): DMSP period'),
         Line2D([], [], marker='s', linestyle='none', color=ORANGE, label='(a): pretrained;  (b): VIIRS period')] + FILLED_OPEN
    fig.legend(handles=h, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.17))
    fig.savefig(f'{OUT}/period_climate.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

# ---------------------------------------------------------------- 6. study area by climate zone (new)
def climate_map():
    pal = {'Tropical savanna (Aw, As)': '#e0a526', 'Semi-arid steppe (BS)': '#b5651d', 'Arid desert (BW)': '#d9cf9c',
           'Tropical humid (Af, Am)': '#1b7a3e', 'Temperate (C)': '#5b8fd6'}
    lab = {'Tropical savanna (Aw, As)': 'Tropical savanna (Aw, As)', 'Semi-arid steppe (BS)': 'Semi-arid steppe (BS)',
           'Arid desert (BW)': 'Arid desert (BW)', 'Tropical humid (Af, Am)': 'Tropical humid (Af, Am)', 'Temperate (C)': 'Temperate (C)'}
    fig, ax = plt.subplots(figsize=(6.6, 6.9), constrained_layout=True)
    lat, lon = CL.lat.to_numpy(float), CL.lon.to_numpy(float)
    for c in ['Arid desert (BW)', 'Semi-arid steppe (BS)', 'Temperate (C)', 'Tropical humid (Af, Am)', 'Tropical savanna (Aw, As)']:
        m = (CL.climate == c).to_numpy()
        ax.scatter(lon[m], lat[m], s=2.2, color=pal[c], linewidths=0, rasterized=True,
                   label=f'{lab[c]}: {m.sum():,} clusters ({100 * m.mean():.0f}%)')
    ax.set_aspect(1 / np.cos(np.radians(np.nanmedian(lat))))
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)
    leg = ax.legend(loc='lower left', markerscale=4, fontsize=7.5, handletextpad=0.3)
    ax.set_title('The 40,360 DHS clusters by Köppen–Geiger climate zone', fontsize=10)
    fig.savefig(f'{OUT}/clusters_climate_map.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    encodings_figure('iwi')
    country_forest(); period_climate()
    if CL is not None:
        country_map('iwi'); climate_map()
    else:
        print(f'{CLUSTERS_CSV} not found: the maps (Figures S1 and S3) need the cluster coordinates; '
              'run analysis/subset_gains.py on the cluster-level predictions first')
    print(sorted(os.listdir(OUT)))
