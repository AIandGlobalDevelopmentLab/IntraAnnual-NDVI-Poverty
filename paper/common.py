"""Shared names and number formats for the tables and figures of the paper.

Inputs (override with environment variables):
  IWI_EVAL_DIR   evaluation outputs (notebooks 16-18, evaluation/iwi)   default: results/iwi
  ANALYSIS_OUT   outputs of the scripts in analysis/                    default: results/analysis
"""
import os
import pandas as pd, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
EVAL = {'iwi': Path(os.environ.get('IWI_EVAL_DIR', REPO / 'results' / 'iwi'))}
REV = Path(os.environ.get('ANALYSIS_OUT', REPO / 'results' / 'analysis'))
def ev(t, name): return EVAL[t] / name
LAB = {'resnet18': 'ResNet-18', 'resnet34': 'ResNet-34', 'resnet50': 'ResNet-50', 'swin': 'Swin v2-B',
       'unet': 'U-Net (EfficientNet-B3)', 'vit': 'ViT-S/16', 'satclip': 'ResNet-18 + SatCLIP', 'satclip_only': 'SatCLIP only',
       'mosaiks': 'MOSAIKS', 'olmoearth': 'OlmoEarth v1 Base', 'vgg16bn': 'VGG-16-BN', 'mlp': 'MLP on band statistics'}
NINE = ['resnet18', 'resnet34', 'resnet50', 'swin', 'unet', 'vit', 'satclip', 'mosaiks', 'olmoearth']
ENC = ['wavelet', 'DFT', 'peak:pixel11/wavelet6', 'TT:cyclic']
ENCLAB = {'wavelet': 'Wavelet, 6 bands', 'DFT': 'Fourier, 5 bands', 'peak:pixel11/wavelet6': 'Peak-aligned wavelet, 6 bands',
          'TT:cyclic': 'Temporal transformer'}
ENCSHORT = {'wavelet': 'Wavelet', 'DFT': 'Fourier', 'peak:pixel11/wavelet6': 'Peak-aligned wavelet', 'TT:cyclic': 'Temporal transformer'}
REF = {'iwi': 'L+NL'}                   # reference input: Landsat + nighttime lights
def combo(t, e): return REF[t] if e is None else f'{REF[t]}+{e}'
def f3(x, sign=True):
    if x is None or (isinstance(x, float) and not np.isfinite(x)): return '--'
    s = f'{x:+.3f}' if sign else f'{x:.3f}'
    return s.replace('-', '$-$')
def ci(lo, hi):
    return f'[{f3(lo, False)}, {f3(hi, False)}]'
def est(x, lo, hi, star=False):
    return f'{f3(x)} {ci(lo, hi)}' + ('*' if star else '')

def clear_counts(C):
    """C: consensus rows (one per country). Positive / clear gain / clear loss as in Section 5.3 of the paper."""
    pos = (C['median dR2'] > 0).sum()
    gain = ((C['median dR2'] > 0) & (C['sig. better'] >= 3) & (C['sig. worse'] <= 1)).sum()
    loss = ((C['median dR2'] < 0) & (C['sig. worse'] >= 3) & (C['sig. better'] <= 1)).sum()
    return pos, gain, loss
