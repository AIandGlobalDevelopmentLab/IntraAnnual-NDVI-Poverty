# How much does the growing season tell us about wealth?

Code for the paper **"How much does the growing season tell us about wealth? Intra-annual vegetation and satellite-based poverty mapping in Africa"** by Mohammad Kakooei, Klaudia Solska and Adel Daoud (The AI and Global Development Lab).

The paper tests whether the intra-annual dynamics of vegetation improve satellite estimates of household wealth, measured by the International Wealth Index (IWI), beyond annual Landsat imagery and nighttime lights (NL). It uses 40,360 Demographic and Health Surveys (DHS) clusters in 36 African countries (2001–2023).

The MODIS NDVI series of the survey year is encoded in four ways and added to the reference input, Landsat + NL:

- six wavelet statistics;
- five Fourier bands;
- six peak-aligned wavelet statistics;
- a temporal transformer learned end to end.

Each input is used in two baseline models (VGG-16 with batch normalization and an MLP on band statistics) and in nine newer architectures. The newer architectures are ResNet-18/34/50, Swin v2-B, ViT-S/16, U-Net, ResNet-18 + SatCLIP, MOSAIKS and OlmoEarth. Every model is trained from pretrained weights and from scratch on five spatially separated folds.

## Repository layout

```
notebooks/
  iwi/         01–18  download, quality control, spatial folds, training and evaluation
evaluation/
  iwi/         bootstrap summaries over architectures
analysis/      subgroup analyses: rural/urban, sensor period, climate zone, countries
paper/         the tables and figures of the paper and its supplement
results/       aggregated outputs behind the paper (no cluster-level data)
tools/         export of the fold assignment for publication
environment/   container definition used for the runs
```

## Data

**DHS.** The cluster coordinates and survey data are available from the [DHS Program](https://dhsprogram.com/data/available-datasets.cfm) after registration and **must not be redistributed**. This repository therefore contains no DHS data. Notebook `iwi/01` starts from a table of clusters, `dhs_data_cluster.csv`, with one row per cluster and these columns:

| column | content |
|---|---|
| `cluster_id` | unique identifier; the part before the first dot is read as the country (for example `NG.2018.123`), unless a `country` column is present |
| `lat`, `lon` | DHS cluster coordinates (displaced by DHS) |
| `year` | survey year; clusters from 2000 onwards are used |
| `rural` | rural (true/1/R) or urban (false/0/U), as classified by DHS |
| `iwi` | cluster mean of the household International Wealth Index (0–100) |

**Earth observation.** Notebook `iwi/01` downloads three 224 × 224 GeoTIFFs per cluster with Google Earth Engine:

- Landsat Collection 2 annual median;
- DMSP-OLS (to 2013) or VIIRS (from 2014) nighttime lights;
- the 23 MOD13Q1 NDVI composites of the survey year.

This needs an Earth Engine account and a Google Cloud project registered for Earth Engine. Set its name in `EE_PROJECT`.

**Climate zones.** The analysis scripts look up the Köppen–Geiger class of each cluster with [kgcpy](https://pypi.org/project/kgcpy/).

**Pretrained weights.** torchvision and torchgeo download them into `TORCH_HOME`. Section 4 of notebooks 11–15 lists the files, for machines without internet access.

## Setup

Python 3.10 or later. Training needs a CUDA GPU; evaluation and analyses run on a CPU.

```bash
pip install -r requirements.txt
# OlmoEarth (notebooks 11–15) needs one more package, installed without its dependencies:
pip install --no-deps --target /path/to/python_packages olmoearth-pretrain-minimal==0.1.5 universal-pathlib pathlib-abc
```

The original runs used the NVIDIA PyTorch 26.05 container with the packages in `environment/apptainer_gpu.def`.

## Configure the paths

Every notebook has a **Settings** cell near the top. All locations in it are sample paths under `/path/to/`; replace them with your own:

| setting | sample value | content |
|---|---|---|
| `CSV_PATH` | `/path/to/data/dhs_data_cluster.csv` | the DHS cluster table (notebooks 01, 02) |
| `OUT_DIR`, `QC_DIR`, `WORK_DIR` | `/path/to/data/` | images (`Dataset/`), quality control, folds, caches, runs and evaluation |
| `TORCH_HOME` | `/path/to/torch_home` | pretrained weights (`hub/checkpoints/`) |
| `PY_PACKAGES` | `/path/to/python_packages` | where `olmoearth-pretrain-minimal` was installed |
| `EE_PROJECT` | `your-earth-engine-project` | Google Cloud project for Earth Engine (notebook 01) |

To change them all at once, run for example `sed -i 's#/path/to/#/data/myproject/#g' notebooks/*/*.ipynb`.

## Run the notebooks

Run the notebooks in their numbered order, interactively or headless:

```bash
cd notebooks/iwi
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
    --output-dir /path/to/executed 12_Landsat_NL_Models.ipynb
```

The training notebooks read optional environment variables, so that one call can train a subset. Set `TRAIN_FROM_SCRATCH=1` for the runs from random weights; the default starts from pretrained weights. The `*_MODELS` variables take `all` (default), `none` (only build the caches and the result tables) or comma-separated model keys. The keys are `resnet18`, `resnet34`, `resnet50`, `swin`, `unet`, `vit`, `satclip`, `satclip_only`, `mosaiks`, `olmoearth`, `vgg16bn` and `mlp`. The `*_FOLDS` variables take `all` or test folds out of `A`–`E`. For example:

```bash
TRAIN_FROM_SCRATCH=1 LN_MODELS=resnet18,vgg16bn LN_FOLDS=A,B jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=-1 --output-dir /path/to/executed 12_Landsat_NL_Models.ipynb
```

Each run saves its state after every epoch. A stopped run continues where it stopped, and finished runs are skipped.

### The notebooks (`notebooks/iwi`)

| notebook | what it does | environment variables |
|---|---|---|
| `01_Download_Landsat_NL_MODIS` | Landsat, nighttime lights and MODIS NDVI for every cluster (Earth Engine) | |
| `02_Check_Dataset` | quality control; writes `train_clusters.csv` and band statistics | |
| `03_Spatial_Folds` | five folds whose clusters lie more than 9.5 km apart, balanced by country and IWI | |
| `04_Table2_DHS` | rerun of the preliminary experiment with its original VGG-16 (reference run) | `TABLE2_EXPERIMENTS`, `TABLE2_FOLDS` |
| `05_Table2_ResNet18` | the same with ResNet-18 (reference run) | `TABLE2_EXPERIMENTS`, `TABLE2_FOLDS`, `RESNET_MODE` |
| `06_NL_ResNet18_Features` | ResNet-18 on nighttime lights alone, with its features | `NLFEAT_FOLDS` |
| `07_Landsat_ResNet18_Features` | ResNet-18 on Landsat with and without NDVI bands, with its features | `LSFEAT_ROWS`, `LSFEAT_FOLDS` |
| `08_Landsat_ViTS16_Features` | the same with ViT-S/16 (SSL4EO-L weights) | `VITFEAT_ROWS`, `VITFEAT_FOLDS`, `VIT_MODE` |
| `09_Ridge_NL_Landsat`, `10_Ridge_NL_ViT` | ridge regression on the frozen features of 06–08 (CPU) | |
| `12_Landsat_NL_Models` | all models on Landsat + NL, the reference input | `LN_MODELS`, `LN_FOLDS` |
| `11_Landsat_NL_Wavelet_Models` | + 6 wavelet bands | `LNW_MODELS`, `LNW_FOLDS` |
| `13_Landsat_NL_DFT_Models` | + 5 Fourier bands | `LND_MODELS`, `LND_FOLDS` |
| `14_Landsat_NL_PeakAligned_Wavelet_Models` | + 6 peak-aligned wavelet bands | `PK_MODELS`, `PK_FOLDS`, `PK_ALIGN`, `PK_INPUT` |
| `15_Landsat_NL_Temporal_Transformer_Models` | + temporal transformer on the 23 composites | `LNT_MODELS`, `LNT_FOLDS`, `TEMPORAL_POSITION` |
| `16_Evaluate_All_Models` | metrics and paired bootstrap comparisons over all runs (CPU) | |
| `17_Evaluate_Per_Country`, `18_Evaluate_Rural_Urban` | the same per country and for rural and urban clusters (CPU) | |

All training notebooks also read `TRAIN_FROM_SCRATCH`. Notebooks 11–15 and their evaluation give the main results; 04–10 give the reference runs of Supplementary Section S4.

## Reproduce the numbers of the paper

1. **Evaluation.** Notebook 16 writes `evaluation/DMSP_VIIRS/` in its work folder, with `eval_runs.csv` and the cluster-level `eval_test_predictions.csv.gz`.
2. **Bootstrap summaries.** These give 1,000 paired Poisson(1) resamples of the 10,944 spatial groups, with seed 20261117:

   ```bash
   python evaluation/iwi/summary_bootstrap.py /path/to/data/evaluation/DMSP_VIIRS
   ```

3. **Subgroup analyses** (see `analysis/README.md`):

   ```bash
   export IWI_EVAL_DIR=/path/to/data/evaluation/DMSP_VIIRS
   python analysis/subset_gains.py
   ```

4. **Tables and figures** (see `paper/README.md`). These read `results/` by default, so they run without the DHS data:

   ```bash
   python paper/make_tables.py
   python paper/figures.py
   ```

## Results

`results/` holds the aggregated outputs behind the tables and figures of the paper:

- `results/iwi/`: from the evaluation notebooks and `evaluation/iwi/summary_bootstrap.py`;
- `results/analysis/`: from the subgroup analyses.

They contain metrics and comparisons for the whole continent, for rural and urban clusters, for sensor periods, for climate zones and for countries. They contain no cluster-level data. The cluster-level test predictions and the cluster coordinates are not distributed, because they are derived from DHS data. The two maps of the supplement therefore need the DHS data.

The fold assignment can be published without DHS values. `tools/export_folds.py` writes `results/folds.csv` from the folds file of notebook `iwi/03`, with only `cluster_id`, the spatial group and the fold:

```bash
python tools/export_folds.py /path/to/data/train_clusters_folds.csv results/folds.csv
```

## Notes

- Some notebook text refers to numbered notebooks of the preliminary study's TensorFlow pipeline, such as `26_tfrecords_generation_DMSP_VIIRS` or `27_Train_NL`. These are not part of this repository.
- Notebooks `iwi/04`–`10` use earlier notebook numbers in their text; a note at the top of each gives the mapping.
- GPU training is not bit-for-bit reproducible, and the variation between repeated runs was not measured.

## Citation

See `CITATION.cff`. The paper is under review at *Remote Sensing Applications: Society and Environment*.

## License

MIT, see `LICENSE`. The DHS data and the pretrained weights keep their own terms of use.
