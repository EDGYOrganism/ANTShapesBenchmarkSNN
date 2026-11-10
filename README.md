# ANTShapes Benchmarking Datasets for Event-Based Object Classification

Code used to produce the results and figures for the publication above (pending review).

## Usage

```bash
pip install -r requirements.txt
python main.py --dataset antshapes_vanilla
python main.py --dataset all --epochs 30 --lr 5e-4
```

| Flag | Default | |
|---|---|---|
| `--dataset` | `all` | dataset key or `all` |
| `--epochs` | 50 | epochs per trial |
| `--batch_size` | 64 | |
| `--lr` | 1e-3 | Adam learning rate (cosine-annealed) |
| `--beta` | 0.5 | LIF membrane decay |
| `--time_window_us` | per dataset | frame width override |
| `--clear_cache` | off | rebuild the frame cache |

Datasets are read from `data/` and frame caches written to `cache/`, both next to `main.py`. 
Set environment variables `SNNBENCH_DATA_ROOT` and `SNNBENCH_CACHE_ROOT` to use other locations. 
If a dataset is missing, the error says how to get it.

## Datasets

| Key | Resolution | Classes | Frame width | Denoise | Split |
|---|---|---|---|---|---|
| `nmnist` | 34×34 | 10 | 10 ms | yes | tonic (auto-download) |
| `pokerdvs` | 35×35 | 4 | 2 ms | no | 80/20 per class, from `convert_pokerdvs.py` |
| `dvsgesture` | 128×128 | 11 | 50 ms | yes | tonic (manual download) |
| `cifar10dvs` | 128×128 | 10 | 10 ms | yes | `CIFAR10DVS/split.json` (8000 / 2000) |
| `antshapes_vanilla` | 128×128 | 12 | 10 ms | no | `VANILLA/split.json` (2400 / 600) |
| `antshapes_distortion` | 128×128 | 12 | 10 ms | no | `DISTORTION/split.json` (2400 / 600) |
| `antshapes_rotation` | 128×128 | 12 | 10 ms | no | `ROTATION/split.json` (2400 / 600) |
| `antshapes_translation` | 128×128 | 12 | 10 ms | no | `TRANSLATION/split.json` (2400 / 600) |

Expected layout under the dataset root folder:
- `NMNIST/`: downloaded by tonic on first use.
- `DVSGesture/`: tonic's layout, downloaded manually (links in the error message).
- `POKERDVS/`: written by `python convert_pokerdvs.py --tar poker_dvs.tar.gz`, requires the original dataset to be downloaded from http://www2.imse-cnm.csic.es/caviar/POKER_DVS/poker_dvs.tar.gz.
- `CIFAR10DVS/{class}/{class}/*.aedat` + `CIFAR10DVS/split.json`: original AEDAT 2.0 release.
- `{VANILLA,DISTORTION,ROTATION,TRANSLATION}/{id}/events.bin, objects.bin` +
  `split.json`: ANTShapes exports.

### Split files

`split.json` lists the `train` and `test` samples and gives per-class counts.
ANTShapes entries are sample folder names; CIFAR10-DVS entries are file paths relative to `CIFAR10DVS/`.

## Pipeline

1. **Denoise** (where enabled): drop events with no spatial neighbour within 10 ms.  This is disabled for ANTShapes datasets.
2. **ToFrame**: bin events into fixed-width time windows, keeping the trailing partial window. Frame width is chosen per dataset so frames stay sparse.
3. **Pad and binarise**: batches are zero-padded to the longest sequence, then frames are thresholded to `{0, 1}`. The input has shape `(T, B, 2, H, W)`.

Framed samples are cached to disk with `tonic.DiskCachedDataset`, so the first epoch is slow.

## Model

```
(T, B, 2, H, W)
 → Conv(2->16, 5×5)  -> AdaptiveMaxPool(H/4)  -> LIF
 → Conv(16->32, 5×5) -> AdaptiveMaxPool(H/16) -> LIF
 → Conv(32->64, 5×5) -> AdaptiveMaxPool(4)    -> LIF
 → Linear(64*4*4 -> classes) -> leaky integrator (non-spiking)
```

Each stage pools to `max(size // 4, 4)`. 
Stages stop once the map reaches 4×4, so 34×34 and 35×35 inputs use two stages. 
Hidden neurons are snnTorch `Leaky` LIF (beta = 0.5, threshold 0.5, subtractive reset, `fast_sigmoid` surrogate with slope 25). T
The output membrane follows `mem[t] = β·mem[t−1] + fc(x[t])`.
The loss is cross-entropy on each class's peak membrane over time, and the prediction is the argmax of that peak.

Training uses the Adam optimiser with cosine annealing and a gradient-norm clip of 1.0. 
Each trial seeds torch with `i*42 + 7` and stops early if test accuracy reaches 95%. 
After training, the checkpoint with the best test accuracy is restored for the
final evaluation.

## Output

Sparsity is the fraction of a layer's neurons that are silent at a timestep, averaged over timesteps and samples.

`overall_accuracy.csv` has one row per trial with `accuracy`.
Cohen's `kappa` and `chance_adjusted` = (acc − 1/K) / (1 − 1/K). 
Class counts differ (4, 10, 11, 12), so compare datasets on the chance-adjusted scores.

## Analysis

```bash
python analysis.py data                       # model-free separability, all datasets
python analysis.py results --results results  # statistics over the latest SNN run per dataset
```

**`data`** doesn't use the SNN. Each sample is reduced to two fixed hand-crafted
representations, computed from the raw events (no denoising or framing):

- `descriptor`: the 639-d event descriptor that the PCA figure is drawn from.
  It contains an absolute occupancy map (16x16), a motion-compensated shape map
  (16x16), an ON/OFF imbalance map (8x8), a radial density profile (8),
  event-rate and ON-fraction profiles over normalised time (24 + 8), and 23
  scalar statistics. Some blocks were designed with ANTShapes classes in mind.
- `histogram`: a 2x16x16 per-polarity event-count histogram, with no design
  choices tied to any dataset.

Features are z-scored per feature, and descriptor blocks are scaled to equal total variance. 
That is the same input the PCA figure uses. 
On the full feature space it reports the following (written to `analysis/separability.csv`):

| Column | Meaning |
|---|---|
| `pc1_var`, `pc2_var` | explained variance ratio of PC1/PC2 |
| `davies_bouldin` | lower = more separable |
| `calinski_harabasz` | higher = more separable |
| `fisher_ratio` | trace(S_within) / trace(S_between), lower = more separable |
| `knn_*`, `probe_*` | 5-NN and logistic-regression test `accuracy`, `kappa`, `chance_adjusted`, trained on the official train split |

Cluster indices use all samples. The probes fit standardisation on the train split only. 
Features are cached under `$SNNBENCH_CACHE_ROOT/features/`.

**`results`** reads the latest run per dataset and computes per-trial scores from the confusion matrices. 
It writes to `analysis/`:

- `snn_summary.csv`: mean, std and 95% t-interval for accuracy, κ and
  chance-adjusted accuracy.
- `snn_pairwise.csv`: Welch's t-test for every pair of datasets, on raw
  accuracy and on chance-adjusted accuracy. Each row has the difference, its
  95% CI and p, with Holm correction over all pairs. Raw accuracy is only
  comparable when `same_n_classes` is true.
- `ranking_agreement.csv`: Spearman ρ between the SNN ranking and each
  model-free metric (signed so that positive means the same order). This needs
  `data` to have been run, and at least 3 datasets.
