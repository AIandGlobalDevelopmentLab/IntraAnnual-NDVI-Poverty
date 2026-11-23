# Aggregated results

These are the outputs behind the tables and figures of the paper. They contain no cluster-level data: every row summarizes many clusters, and the counts per subset are given in the `n` columns.

| folder | written by |
|---|---|
| `iwi/` | the evaluation notebooks 16–18 and `evaluation/iwi/summary_bootstrap.py` |
| `analysis/` | the scripts in `analysis/` |

## Main files

| file | content |
|---|---|
| `eval_runs.csv` | every training run: model, input (`combo`), start, test folds and run folder |
| `overall_metrics.csv` | R², RMSE, MAE and bias of every run on all test clusters |
| `overall_pretrained_minus_scratch.csv` | pretrained against scratch for every model and input |
| `overall_vs_baseline_models*.csv` | every model against VGG-16-BN and the MLP |
| `summary_bootstrap.csv` | means over the nine newer architectures, with 95% intervals |
| `rural_urban_*.csv` | the same comparisons within rural and urban clusters |
| `country_*.csv` | per-country results and the clear gains and losses per encoding |
| `analysis/iwi_mae.csv` | changes in MAE |
| `analysis/subset_gains.csv` | gains by area, sensor period, climate zone and country |
| `analysis/years_climate.csv`, `within_country_period.csv` | survey-year bins, climate zones and the within-country period split |
| `analysis/country_covariates.csv`, `country_decomp.csv`, `case_studies.csv` | country traits, the MSE decomposition and the case studies |

## Column conventions

- `dR2`: change in R², with `dR2 low` and `dR2 high` as the 95% interval of the paired cluster bootstrap.
- `P(dR2 > 0)`: share of bootstrap replicates with a positive change.
- `p`: two-sided p-value.
- `q`: Benjamini–Hochberg value.
- `relMSE` (analysis): relative change in MSE; negative means that NDVI helps.
