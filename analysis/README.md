# Subgroup analyses

These scripts produce the numbers on when intra-annual NDVI helps:

- rural and urban clusters;
- the two nighttime-light sensor periods and the survey years;
- climate zones;
- countries, with the meta-analysis, the MSE decomposition and the case studies.

They read the cluster-level test predictions written by notebook 16, so they need the DHS data but no GPU.

## Requirements

Python 3.10 or later, with numpy, pandas, scipy and kgcpy (`pip install kgcpy`). kgcpy gives the Köppen–Geiger class of each cluster (about 3 km grid).

## Inputs and outputs

| variable | default | content |
|---|---|---|
| `IWI_EVAL_DIR` | `results/iwi` | evaluation folder, with `eval_test_predictions.csv.gz` |
| `ANALYSIS_OUT` | `results/analysis` | where the scripts write |

The prediction file is not part of the repository. Point `IWI_EVAL_DIR` to your evaluation folder, or copy the file into `results/iwi`; `.gitignore` keeps it out of git.

## Run order

Run the scripts from the repository root. Run `subset_gains.py` first, because the others read its `D.pkl`.

| step | command | writes | used in the paper |
|---|---|---|---|
| 1 | `python analysis/subset_gains.py` (about 1 min) | `subset_gains.csv`, `subset_gains_clusters.csv`*, `D.pkl`* | Sections 6.3–6.5, Table 5, Figures 4 and 5 |
| 2 | `python analysis/years_climate.py` | `years_climate.csv` | Section 6.5, Figure 5 |
| 3 | `python analysis/within_country_period.py` | `within_country_period.csv` | Section 6.5 |
| 4 | `python analysis/country_heterogeneity.py` | `country_covariates.csv`; prints Q, I², τ, Spearman correlations and the weighted regression | Section 6.4 |
| 5 | `python analysis/case_studies.py` | `case_studies.csv`, `cases.pkl`* | Supplementary Section S5 (Table S5) |
| 6 | `python analysis/country_decomposition.py` | `country_decomp.csv` | Section 6.4, Supplementary Section S5 |
| 7 | `python analysis/mae_iwi.py $IWI_EVAL_DIR/eval_test_predictions.csv.gz results/analysis/iwi_mae.csv` | `iwi_mae.csv` | Section 6.2 (MAE) |

\* Cluster-level files with DHS coordinates or wealth values: do not publish them (`.gitignore` excludes them).

`gainlib.py` holds the shared bootstrap helper and is imported by steps 2–5.

## Method

- **Gain.** The NDVI encoding minus the reference input, Landsat + NL.
- **Models.** The mean over the nine newer architectures (resnet18/34/50, swin, unet, vit, satclip, mosaiks, olmoearth); VGG-16-BN and the MLP are reported separately.
- **Encodings.** `wavelet`, `DFT`, `peak:pixel11/wavelet6` and `TT:cyclic`; `mean4` is their mean.
- **Intervals.** A paired Poisson(1) bootstrap over the 10,944 spatial groups, with 1,000 replicates and seed 20261117. The same weights are used for every model and subset, so differences between subsets are paired.
- **`relMSE`.** The relative change in MSE; negative means that NDVI helps. ΔR² = −relMSE × (1 − base R²).
- **Climate groups.** BW arid desert, BS semi-arid steppe, Aw/As tropical savanna, Af/Am tropical humid, C temperate.
