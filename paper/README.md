# Tables and figures of the paper

These scripts make the result tables and the new figures of the paper and its supplement. They read the evaluation outputs of both experiments and the outputs of the subgroup analyses, by default from `results/` (override with `IWI_EVAL_DIR`, `NL_EVAL_DIR` and `ANALYSIS_OUT`).

```bash
python paper/make_tables.py    # writes paper/tables/
python paper/figures.py        # writes paper/figures/
```

Requirements: numpy, pandas and matplotlib.

## Tables (`paper/tables/`, the table bodies)

| file | paper |
|---|---|
| `tab_iwi_acc.tex`, `tab_nl_acc.tex` | Tables 3 and 6: accuracy on the reference input |
| `tab_iwi_enc.tex`, `tab_nl_enc.tex` | Tables 4 and 7: gains of the NDVI encodings |
| `tab_iwi_subgroups.tex`, `tab_nl_subgroups.tex` | Tables 5 and 8: rural/urban and sensor periods, mean of the four encodings |
| `tab_iwi_newer.tex`, `tab_nl_newer.tex` | Tables S2 and S3: newer architectures against the baseline models |
| `tab_iwi_countries.tex`, `tab_nl_countries.tex` | Tables S6 and S8: country counts per encoding |
| `tab_iwi_area.tex`, `tab_nl_area.tex` | Tables S9 and S10: rural and urban clusters per encoding |
| `tab_iwi_period.tex`, `tab_nl_period.tex` | Tables S11 and S12: sensor periods per encoding |

## Figures (`paper/figures/`)

| file | paper |
|---|---|
| `iwi_encodings_pretrained.png`, `iwi_encodings_scratch.png` | Figure 3 |
| `country_forest_scratch.png` | Figure 4 |
| `period_climate.png` | Figure 5 |
| `nl_encodings_pretrained.png`, `nl_encodings_scratch.png` | Figure 6 |
| `iwi_nl_side_by_side.png` | Figure 7 |
| `clusters_climate_map.png` | Figure S1* |
| `iwi_country_map_scratch.png`, `nl_country_map_scratch.png` | Figures S3 and S4* |

\* The maps need the cluster coordinates in `subset_gains_clusters.csv`. This file is made by `analysis/subset_gains.py` from the DHS data and is not distributed. Without it, `figures.py` skips the maps.
