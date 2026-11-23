# Tables and figures of the paper

These scripts make the result tables and the new figures of the paper and its supplement. They read the evaluation outputs and the outputs of the subgroup analyses, by default from `results/` (override with `IWI_EVAL_DIR` and `ANALYSIS_OUT`).

```bash
python paper/make_tables.py    # writes paper/tables/
python paper/figures.py        # writes paper/figures/
```

Requirements: numpy, pandas and matplotlib.

## Tables (`paper/tables/`, the table bodies)

| file | paper |
|---|---|
| `tab_iwi_acc.tex` | Table 3: accuracy on the reference input, Landsat + NL |
| `tab_iwi_enc.tex` | Table 4: gains of the NDVI encodings |
| `tab_iwi_subgroups.tex` | Table 5: rural/urban and sensor periods, mean of the four encodings |
| `tab_iwi_newer.tex` | Table S2: newer architectures against the baseline models |
| `tab_iwi_countries.tex` | Table S4: country counts per encoding |
| `tab_iwi_area.tex` | Table S6: rural and urban clusters per encoding |
| `tab_iwi_period.tex` | Table S7: sensor periods per encoding |

## Figures (`paper/figures/`)

| file | paper |
|---|---|
| `iwi_encodings_pretrained.png`, `iwi_encodings_scratch.png` | Figure 3 |
| `country_forest_scratch.png` | Figure 4 |
| `period_climate.png` | Figure 5 |
| `clusters_climate_map.png` | Figure S1* |
| `iwi_country_map_scratch.png` | Figure S3* |

\* The maps need the cluster coordinates in `subset_gains_clusters.csv`. This file is made by `analysis/subset_gains.py` from the DHS data and is not distributed. Without it, `figures.py` skips the maps.
