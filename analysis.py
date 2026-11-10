"""Quantitative dataset separability and statistics over SNN results.

    python analysis.py data    [--dataset all] [--force]
    python analysis.py results [--results results]

`data` scores how separable each dataset's classes are using two hand-crafted
event representations and no SNN; it writes analysis/separability.csv.
`results` summarises the latest SNN run per dataset (mean and 95% CI, pairwise
Welch's t-tests, rank agreement with `data`); it writes analysis/snn_*.csv and
analysis/ranking_agreement.csv. See the README for every column.
"""

import argparse
import functools
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (calinski_harabasz_score, confusion_matrix,
                             davies_bouldin_score, silhouette_score)
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader

import main as sb

FEATURE_DIR = sb.CACHE_ROOT / "features"
OUT_DIR = Path("analysis")

K_NEIGHBOURS = 5
PROBE_MAX_ITER = 2000
SILHOUETTE_SAMPLES = 10_000   # silhouette is O(N²); larger datasets (N-MNIST) are subsampled
SEED = 0

# ---------------------------------------------------------------------------
# Hand-crafted representations
# ---------------------------------------------------------------------------

MAP_BINS, POLARITY_BINS, RADIAL_BINS, TEMPORAL_BINS, ONFRAC_BINS = 16, 8, 8, 24, 8
HIST_BINS = 16
N_CHUNKS = 8          # time chunks for motion compensation and centroid tracking
SHAPE_EXTENT = 2.5    # half-width of the recentred map, in RMS radii
MIN_EVENTS = 32       # sparser samples get no descriptor and are dropped
N_STATS = 23

# Feature columns: the 639-d descriptor (its blocks are weighted equally after
# z-scoring, so the wide maps don't swamp the 23 statistics), then the histogram.
LAYOUT = {
    "descriptor": {"spatial": MAP_BINS ** 2, "shape": MAP_BINS ** 2, "polarity": POLARITY_BINS ** 2,
                   "radial": RADIAL_BINS, "temporal": TEMPORAL_BINS + ONFRAC_BINS, "stats": N_STATS},
    "histogram": {"counts": 2 * HIST_BINS ** 2},
}
N_FEATURES = sum(sum(b.values()) for b in LAYOUT.values())


def representation(X: np.ndarray, rep: str) -> tuple[np.ndarray, list[slice]]:
    """The columns of one representation, and its block slices within them."""
    start = 0
    for name, blocks in LAYOUT.items():
        width = sum(blocks.values())
        if name == rep:
            edges = np.cumsum([0, *blocks.values()])
            return X[:, start:start + width], [slice(a, b) for a, b in zip(edges[:-1], edges[1:])]
        start += width
    raise KeyError(rep)


def _hist2d(u, v, bins, weights=None):
    """Histogram over the unit square; u indexes rows."""
    idx = (np.clip(u, 0, 1 - 1e-9) * bins).astype(np.int32) * bins + (np.clip(v, 0, 1 - 1e-9) * bins).astype(np.int32)
    return np.bincount(idx, weights=weights, minlength=bins * bins).astype(np.float64)


def _sqrt_l1(h):
    """L1-normalise, then sqrt: the variance-stabilising transform for counts."""
    total = h.sum()
    return np.sqrt(h / total) if total > 0 else h


def describe(events: np.ndarray, sensor_size: tuple) -> np.ndarray:
    """Descriptor and histogram for one event stream (NaN if under MIN_EVENTS events).

    Lengths are normalised by sensor size and times by recording duration, so
    datasets with different resolutions and durations share one feature space.
    """
    W, H = float(sensor_size[0]), float(sensor_size[1])
    n = len(events)
    if n < MIN_EVENTS:
        return np.full(N_FEATURES, np.nan)
    order = np.argsort(events["t"], kind="stable")
    x, y, t, p = (events[f][order].astype(np.float64) for f in ("x", "y", "t", "p"))
    dur_us = max(t[-1] - t[0], 1.0)
    tn = (t - t[0]) / dur_us

    spatial = _sqrt_l1(_hist2d(x / W, y / H, MAP_BINS))

    # Motion compensation: recentre each time chunk on its own centroid.
    chunk = np.clip((tn * N_CHUNKS).astype(np.int32), 0, N_CHUNKS - 1)
    counts = np.bincount(chunk, minlength=N_CHUNKS).astype(np.float64)
    nz = counts > 0
    safe = np.where(nz, counts, 1.0)
    cx_c = np.bincount(chunk, weights=x, minlength=N_CHUNKS) / safe
    cy_c = np.bincount(chunk, weights=y, minlength=N_CHUNKS) / safe
    dx, dy = x - cx_c[chunk], y - cy_c[chunk]
    r_rms = max(float(np.sqrt(np.mean(dx * dx + dy * dy))), 1e-6)
    half = SHAPE_EXTENT * r_rms
    u, v = (dx / half + 1) / 2, (dy / half + 1) / 2
    shape = _sqrt_l1(_hist2d(u, v, MAP_BINS))

    on, tot = _hist2d(u, v, POLARITY_BINS, weights=p), _hist2d(u, v, POLARITY_BINS)
    polarity = np.divide(2 * on - tot, tot, out=np.zeros_like(tot), where=tot > 0)

    # Radial density (per unit annulus area) of the recentred cloud.
    rad = np.clip(np.sqrt(dx * dx + dy * dy) / half, 0, 1 - 1e-9)
    radial = np.histogram(rad, bins=RADIAL_BINS, range=(0, 1))[0].astype(np.float64)
    edges = np.linspace(0, 1, RADIAL_BINS + 1)
    radial /= np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
    radial = radial / radial.sum() if radial.sum() > 0 else radial

    rate = np.histogram(tn, bins=TEMPORAL_BINS, range=(0, 1))[0].astype(np.float64)
    rate = rate / rate.sum() if rate.sum() > 0 else rate
    ob = np.clip((tn * ONFRAC_BINS).astype(np.int32), 0, ONFRAC_BINS - 1)
    ob_tot = np.bincount(ob, minlength=ONFRAC_BINS).astype(np.float64)
    onfrac = np.divide(np.bincount(ob, weights=p, minlength=ONFRAC_BINS), ob_tot,
                       out=np.full(ONFRAC_BINS, 0.5), where=ob_tot > 0)

    # Scalar statistics.
    cx, cy = x.mean(), y.mean()
    vxx, vyy, vxy = np.mean((x - cx) ** 2), np.mean((y - cy) ** 2), np.mean((x - cx) * (y - cy))
    tr, det = vxx + vyy, vxx * vyy - vxy * vxy
    disc = max(tr * tr / 4 - det, 0.0) ** 0.5
    lam_max, lam_min = tr / 2 + disc, max(tr / 2 - disc, 0.0)
    anisotropy = (lam_min / lam_max) ** 0.5 if lam_max > 0 else 1.0
    two_theta = np.arctan2(2 * vxy, vxx - vyy)   # principal axis, free of the 180° ambiguity
    bw, bh = x.max() - x.min() + 1, y.max() - y.min() + 1

    cxs, cys = cx_c[nz], cy_c[nz]
    if cxs.size >= 2:
        step = np.hypot(np.diff(cxs), np.diff(cys))
        path, speed, net = step.sum(), step.mean(), np.hypot(cxs[-1] - cxs[0], cys[-1] - cys[0])
    else:
        path = speed = net = 0.0
    rc = np.sqrt((np.bincount(chunk, weights=dx * dx + dy * dy, minlength=N_CHUNKS) / safe)[nz])
    jitter = np.abs(np.diff(rc)).mean() if rc.size >= 2 else 0.0
    bin_c = (np.arange(TEMPORAL_BINS) + 0.5) / TEMPORAL_BINS
    t_cen = (rate * bin_c).sum()
    t_spr = np.sqrt(max((rate * (bin_c - t_cen) ** 2).sum(), 0.0))

    scale = (W + H) / 2
    statistics = np.array([
        np.log10(n), np.log10(n / (dur_us * 1e-6)), dur_us * 1e-6, p.mean(),
        cx / W, cy / H, np.sqrt(vxx) / W, np.sqrt(vyy) / H, r_rms / scale,
        bw / W, bh / H, np.log(bw / bh), np.log10(n / (bw * bh)),
        anisotropy, np.cos(two_theta), np.sin(two_theta),
        speed / scale, path / scale, net / scale, net / path if path > 1e-9 else 1.0,
        jitter / scale, t_cen, t_spr,
    ])

    pol = (p > 0).astype(np.int32)
    hist = np.concatenate([_hist2d(x[pol == k] / W, y[pol == k] / H, HIST_BINS) for k in (0, 1)])

    return np.concatenate([spatial, shape, polarity, radial, rate, onfrac, statistics, _sqrt_l1(hist)])

# ---------------------------------------------------------------------------
# Feature cache
# ---------------------------------------------------------------------------

def feature_path(key: str) -> Path:
    return FEATURE_DIR / f"{key}{sb.split_tag(sb.DATASETS[key])}.npz"


def load_features(key: str, force: bool = False):
    """(X, y, is_train) for every sample of both splits, computed once and cached."""
    path = feature_path(key)
    if not path.exists() or force:
        cfg = sb.DATASETS[key]
        sb.check_data(cfg)
        transform = functools.partial(describe, sensor_size=cfg.sensor_size)
        X, y, train = [], [], []
        for is_train in (True, False):
            ds = cfg.build(is_train, transform)
            loader = DataLoader(ds, batch_size=64, num_workers=sb.NUM_WORKERS)
            for i, (f, label) in enumerate(loader):
                X.append(f.numpy())
                y.append(label.numpy())
                train.append(np.full(len(label), is_train))
                print(f"  {key} {'train' if is_train else 'test'}: {min((i + 1) * 64, len(ds))}/{len(ds)}",
                      end="\r", flush=True)
            print()
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=np.concatenate(X).astype(np.float32),
                            y=np.concatenate(y), train=np.concatenate(train))
    z = np.load(path)
    X, y, train = z["X"].astype(np.float64), z["y"], z["train"]
    valid = np.isfinite(X).all(axis=1)
    if not valid.all():
        print(f"  {key}: dropping {(~valid).sum()} sample(s) with < {MIN_EVENTS} events")
    return X[valid], y[valid], train[valid]


# ---------------------------------------------------------------------------
# Separability
# ---------------------------------------------------------------------------

def standardise(X: np.ndarray, blocks: list[slice], ref: np.ndarray) -> np.ndarray:
    """Z-score with statistics from rows `ref`, drop constant columns, and give
    every block one unit of total variance."""
    mu, sd = X[ref].mean(0), X[ref].std(0)
    keep = sd > 1e-9
    Z = (X - mu) / np.where(keep, sd, 1.0)
    for sl in blocks:
        n = keep[sl].sum()
        if n:
            Z[:, sl] /= np.sqrt(n)
    return Z[:, keep]


def fisher_ratio(Z: np.ndarray, y: np.ndarray) -> float:
    """trace(S_within) / trace(S_between): within-class over between-class scatter."""
    mu = Z.mean(0)
    within = between = 0.0
    for c in np.unique(y):
        Zc = Z[y == c]
        mc = Zc.mean(0)
        within += ((Zc - mc) ** 2).sum()
        between += len(Zc) * ((mc - mu) ** 2).sum()
    return within / between


def probe_scores(clf, Z, y, train, n_classes) -> dict:
    pred = clf.fit(Z[train], y[train]).predict(Z[~train])
    return sb.chance_scores(confusion_matrix(y[~train], pred, labels=range(n_classes)))


def separability(X: np.ndarray, y: np.ndarray, train: np.ndarray, blocks: list[slice], n_classes: int) -> dict:
    Z = standardise(X, blocks, np.ones(len(X), bool))
    evr = PCA(n_components=2, random_state=SEED).fit(Z).explained_variance_ratio_
    sample = SILHOUETTE_SAMPLES if len(Z) > SILHOUETTE_SAMPLES else None
    row = {
        "n_samples": len(Z), "n_features": Z.shape[1], "pc1_var": evr[0], "pc2_var": evr[1],
        "silhouette": silhouette_score(Z, y, sample_size=sample, random_state=SEED),
        "davies_bouldin": davies_bouldin_score(Z, y),
        "calinski_harabasz": calinski_harabasz_score(Z, y),
        "fisher_ratio": fisher_ratio(Z, y),
    }
    # Probes: standardisation fitted on the train split only.
    Z = standardise(X, blocks, train)
    for name, clf in (("knn", KNeighborsClassifier(K_NEIGHBOURS)),
                      ("probe", LogisticRegression(max_iter=PROBE_MAX_ITER))):
        row |= {f"{name}_{k}": v for k, v in probe_scores(clf, Z, y, train, n_classes).items()}
    return row


def run_data(keys: list[str], force: bool):
    rows = []
    for key in keys:
        cfg = sb.DATASETS[key]
        print(f"{cfg.display_name}")
        X, y, train = load_features(key, force)
        for rep in LAYOUT:
            Xr, blocks = representation(X, rep)
            row = {"dataset": key, "representation": rep, "n_classes": len(cfg.class_names),
                   **separability(Xr, y, train, blocks, len(cfg.class_names))}
            rows.append(row)
            print(f"  {rep:10s}  PC1 {row['pc1_var']:.1%}  PC2 {row['pc2_var']:.1%}  "
                  f"sil {row['silhouette']:+.3f}  DB {row['davies_bouldin']:.2f}  "
                  f"CH {row['calinski_harabasz']:.1f}  W/B {row['fisher_ratio']:.2f}  "
                  f"kNN {row['knn_accuracy']:.3f} (κ {row['knn_kappa']:.3f})  "
                  f"linear {row['probe_accuracy']:.3f} (κ {row['probe_kappa']:.3f})")

    OUT_DIR.mkdir(exist_ok=True)
    out = OUT_DIR / "separability.csv"
    df = pd.DataFrame(rows)
    if out.exists():   # keep rows for datasets not recomputed this time
        old = pd.read_csv(out)
        df = pd.concat([old[~old["dataset"].isin(keys)], df])
    order = {k: i for i, k in enumerate(sb.DATASETS)}
    df.sort_values(["dataset", "representation"], key=lambda s: s.map(order) if s.name == "dataset" else s) \
      .to_csv(out, index=False)
    print(f"\nWrote {out}")


# ---------------------------------------------------------------------------
# SNN result statistics
# ---------------------------------------------------------------------------

METRICS = ("accuracy", "kappa", "chance_adjusted")

# Model-free metrics compared with the SNN ranking and whether higher means easier.
HIGHER_IS_EASIER = {"silhouette": True, "davies_bouldin": False, "calinski_harabasz": True,
                    "fisher_ratio": False, "knn_chance_adjusted": True, "probe_chance_adjusted": True}


def latest_run(results_root: Path, key: str) -> Path | None:
    """Newest run directory under results/{key}/ with per-trial confusion matrices.
    Run ids are timestamps, so the lexicographic max is the newest."""
    runs = {p.parent.parent for p in (results_root / key).rglob("trial_*/confusion_matrix.csv")}
    return max(runs, key=lambda p: p.name) if runs else None


def load_trials(results_root: Path) -> pd.DataFrame:
    """Per-trial scores from each dataset's latest run, computed from its confusion matrices."""
    rows = []
    for key in sb.DATASETS:
        run = latest_run(results_root, key)
        if run is None:
            continue
        for f in sorted(run.glob("trial_*/confusion_matrix.csv")):
            cm = pd.read_csv(f, index_col=0).to_numpy()
            rows.append({"dataset": key, "run": str(run), "trial": f.parent.name, **sb.chance_scores(cm)})
    return pd.DataFrame(rows)


def mean_ci(v: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    m, se = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
    h = se * stats.t.ppf((1 + level) / 2, len(v) - 1)
    return m, m - h, m + h


def holm(p: np.ndarray) -> np.ndarray:
    adj, running = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def run_results(results_root: Path):
    trials = load_trials(results_root)
    if trials.empty:
        raise SystemExit(f"No runs found under {results_root}/")
    OUT_DIR.mkdir(exist_ok=True)
    keys = list(dict.fromkeys(trials["dataset"]))

    summary = []
    for key in keys:
        t = trials[trials["dataset"] == key]
        row = {"dataset": key, "n_classes": len(sb.DATASETS[key].class_names), "n_trials": len(t),
               "run": t["run"].iloc[0]}
        for m in METRICS:
            row[f"{m}_mean"], row[f"{m}_ci_low"], row[f"{m}_ci_high"] = mean_ci(t[m].to_numpy())
            row[f"{m}_std"] = t[m].std(ddof=1)
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT_DIR / "snn_summary.csv", index=False)

    print("SNN test accuracy over trials (mean, 95% CI)")
    print(f"  {'dataset':24s} {'K':>3s} {'n':>3s}  {'accuracy':>23s}  {'κ':>23s}  {'chance-adjusted':>23s}")
    for r in summary.itertuples():
        cells = "  ".join(f"{getattr(r, m + '_mean'):.4f} [{getattr(r, m + '_ci_low'):.4f}, "
                          f"{getattr(r, m + '_ci_high'):.4f}]" for m in METRICS)
        print(f"  {r.dataset:24s} {r.n_classes:3d} {r.n_trials:3d}  {cells}")

    # Pairwise Welch's t-tests. Raw accuracy is only comparable between datasets with the same class count; chance-adjusted accuracy is comparable across all.
    pairs = []
    for m in ("accuracy", "chance_adjusted"):
        family = []
        for a, b in itertools.combinations(keys, 2):
            va = trials.loc[trials["dataset"] == a, m].to_numpy()
            vb = trials.loc[trials["dataset"] == b, m].to_numpy()
            res = stats.ttest_ind(va, vb, equal_var=False)
            ci = res.confidence_interval(0.95)
            family.append({"metric": m, "a": a, "b": b, "mean_a": va.mean(), "mean_b": vb.mean(),
                           "diff": va.mean() - vb.mean(), "ci_low": ci.low, "ci_high": ci.high,
                           "t": res.statistic, "df": res.df, "p": res.pvalue,
                           "same_n_classes": len(sb.DATASETS[a].class_names) == len(sb.DATASETS[b].class_names)})
        family = pd.DataFrame(family)
        family["p_holm"] = holm(family["p"].fillna(1.0).to_numpy())
        family["significant"] = family["p_holm"] < 0.05
        pairs.append(family)
    pairs = pd.concat(pairs)
    pairs.to_csv(OUT_DIR / "snn_pairwise.csv", index=False)

    print("\nWelch's t-tests on chance-adjusted accuracy (Holm-adjusted over all pairs)")
    print(f"  {'a':24s} {'b':24s} {'Δ':>8s} {'95% CI of Δ':>20s} {'p':>8s} {'p_holm':>8s}")
    for r in pairs[pairs["metric"] == "chance_adjusted"].itertuples():
        print(f"  {r.a:24s} {r.b:24s} {r.diff:+8.4f} [{r.ci_low:+.4f}, {r.ci_high:+.4f}] "
              f"{r.p:8.4f} {r.p_holm:8.4f}{'  *' if r.significant else ''}")

    sep_file = OUT_DIR / "separability.csv"
    if not sep_file.exists():
        print(f"\n{sep_file} not found: run `python analysis.py data` for the model-free ranking.")
        return
    sep = pd.read_csv(sep_file)
    snn = summary.set_index("dataset")["chance_adjusted_mean"]
    rank_rows = []
    for rep, block in sep.groupby("representation"):
        block = block.set_index("dataset").reindex(snn.index).dropna(how="all")
        common = block.index
        if len(common) < 3:
            print(f"\nOnly {len(common)} dataset(s) have both SNN results and {rep} metrics; "
                  "need at least 3 to compare rankings.")
            continue
        for metric, higher in HIGHER_IS_EASIER.items():
            score = block[metric] if higher else -block[metric]
            rho, p = stats.spearmanr(snn[common], score)
            rank_rows.append({"representation": rep, "metric": metric, "n_datasets": len(common),
                              "spearman_rho": rho, "p": p})
    if not rank_rows:
        return
    ranking = pd.DataFrame(rank_rows)
    ranking.to_csv(OUT_DIR / "ranking_agreement.csv", index=False)
    print("\nAgreement between SNN ranking (chance-adjusted) and model-free ranking "
          "(Spearman ρ, positive = same order)")
    for r in ranking.itertuples():
        print(f"  {r.representation:10s} {r.metric:22s} ρ = {r.spearman_rho:+.3f}  "
              f"p = {r.p:.3f}  (n = {r.n_datasets})")
    print(f"\nWrote {OUT_DIR}/snn_summary.csv, snn_pairwise.csv, ranking_agreement.csv")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    d = sub.add_parser("data", help="model-free separability metrics")
    d.add_argument("--dataset", choices=[*sb.DATASETS, "all"], default="all")
    d.add_argument("--force", action="store_true", help="recompute cached features")
    r = sub.add_parser("results", help="statistics over SNN runs")
    r.add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    if args.command == "data":
        run_data(list(sb.DATASETS) if args.dataset == "all" else [args.dataset], args.force)
    else:
        run_results(args.results)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    main()
