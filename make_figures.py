"""Figures: per-trial/summary plots written by main.py and one-page grid
figures across datasets.

    python make_figures.py pca       [--rows 2 --cols 4] [--fit-width INCHES]
    python make_figures.py confusion [--rows 3 --cols 3] [--fit-width INCHES]

`pca` plots the leading PCA plane of the event descriptor for every dataset
whose feature cache exists (run `python analysis.py data` first). The input is
the same as analysis.py's: descriptor columns, z-scored, blocks scaled to equal
total variance. PCA is fitted per dataset, since each panel asks how separable
that dataset's classes are. Ellipses (1σ) and centroids use every sample;
dense datasets are subsampled for drawing only.

`confusion` plots the mean confusion matrix of the latest run of every dataset
under --results: each trial's counts are row-normalised, then averaged over
trials. Cells show integer percentages, with exact zeros left blank.

Every panel gets the same physical size and gaps are sized from the measured
text, so labels never collide at any --fit-width. Output goes to
figures/{pca,confusion}_grid.{png,pdf}.

The per-trial and summary functions (loss_curves, confusion, sparsity, ...) are
called by main.py after each trial and run.
"""

import argparse
import functools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Ellipse
from matplotlib.textpath import TextToPath
from sklearn.decomposition import PCA


OUT_DIR = Path("figures")
SEED = 0

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e8e7e3"
CLASS_COLORS = ["#852e48", "#d1778b", "#e40149", "#8a5d00", "#be8805", "#0f7f59",
                "#259cc4", "#0367eb", "#344e8d", "#8b82fd", "#7802c2", "#8e5c9f"]
CMAP = "Blues"
LEGEND_RULE = "#b4b2ac"   # separators between per-panel legend keys


@functools.lru_cache(maxsize=None)
def text_w(s: str, fs: float) -> float:
    """Rendered width of one line of text, in inches, from the active font's
    real metrics, so every gap can be budgeted tightly without guessing."""
    w, _, _ = TextToPath().get_text_width_height_descent(s, FontProperties(size=fs), ismath=False)
    return w / 72 * 1.02


def wrap_to(text: str, fs: float, max_w: float) -> str:
    """Greedy word wrap so no line is wider than max_w inches."""
    lines = [""]
    for word in text.split():
        trial = f"{lines[-1]} {word}".strip()
        if lines[-1] and text_w(trial, fs) > max_w:
            lines.append(word)
        else:
            lines[-1] = trial
    return "\n".join(lines)


def row_groups(n: int, rows: int, cols: int) -> list[list[int]]:
    groups = [list(range(r * cols, min((r + 1) * cols, n))) for r in range(rows)]
    return [g for g in groups if g]


def pick(keys: list[str], rows: int, cols: int) -> list[str]:
    if len(keys) > rows * cols:
        print(f"WARNING: {len(keys)} datasets but only {rows * cols} grid slots; dropping {keys[rows * cols:]}")
    return keys[:rows * cols]


def save(fig, stem: str, dpi: int, tight: bool):
    OUT_DIR.mkdir(exist_ok=True)
    png = OUT_DIR / f"{stem}.png"
    kw = {"bbox_inches": "tight"} if tight else {}
    fig.savefig(png, dpi=dpi, **kw)
    fig.savefig(png.with_suffix(".pdf"), **kw)
    plt.close(fig)
    w, h = fig.get_size_inches()
    print(f"\nWrote {png} and {png.with_suffix('.pdf')}  ({w:.2f}×{h:.2f} in)")


# ---------------------------------------------------------------------------
# Per-trial and summary figures written by main.py
# ---------------------------------------------------------------------------

PCT = plt.FuncFormatter(lambda y, _: f"{y:.0%}")
PCT1 = plt.FuncFormatter(lambda y, _: f"{y:.1%}")


def _run_style(fn):
    """Seaborn darkgrid theme for the per-run plots, scoped to the call so it
    does not leak into the grid figures below."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with plt.rc_context():
            sns.set_theme(style="darkgrid", palette="muted")
            return fn(*args, **kwargs)
    return wrapper


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _sparsity_long(log: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in log.columns if c.startswith("sparsity_")]
    df = log.melt(id_vars="epoch", value_vars=cols, var_name="layer", value_name="sparsity")
    df["layer"] = df["layer"].str.removeprefix("sparsity_")
    return df


# --- per trial (main.py) ----------------------------------------------------

@_run_style
def loss_curves(log: pd.DataFrame, path: Path):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))
    ax0.plot(log["epoch"], log["train_loss"], marker="o", label="Train loss")
    ax0.plot(log["epoch"], log["val_loss"], marker="s", label="Val loss")
    ax0.set(title="Loss curves", xlabel="Epoch", ylabel="Loss")
    ax0.legend()
    ax1.plot(log["epoch"], log["val_acc"], marker="o", color="green")
    ax1.set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    ax1.yaxis.set_major_formatter(PCT)
    _save(fig, path)


@_run_style
def confusion(cm: np.ndarray, names: list[str], path: Path):
    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    n = len(names)
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    sns.heatmap(norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=names, yticklabels=names,
                ax=ax, linewidths=0.5)
    ax.set(title="Confusion matrix (row-normalised)", xlabel="Predicted", ylabel="True")
    _save(fig, path)


@_run_style
def sparsity(log: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.lineplot(data=_sparsity_long(log), x="epoch", y="sparsity", hue="layer", marker="o", ax=ax)
    ax.set(title="Sparsity per layer (fraction of silent neurons)", xlabel="Epoch", ylabel="Sparsity")
    ax.yaxis.set_major_formatter(PCT1)
    _save(fig, path)


@_run_style
def spike_activity_by_class(activity: dict, names: list[str], path: Path):
    """activity[class][layer] = mean spike rate per timestep (correct predictions)."""
    layers = list(activity[0])
    fig, axes = plt.subplots(len(names), len(layers), figsize=(5 * len(layers), 2.2 * len(names)),
                             sharex=True, squeeze=False)
    colours = sns.color_palette("muted")
    for row, name in enumerate(names):
        for col, layer in enumerate(layers):
            ax = axes[row][col]
            trace = activity[row][layer]
            ax.bar(range(len(trace)), trace, color=colours[col])
            ax.set_ylim(0, 1)
            if col == 0:
                ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=60, va="center")
            if row == 0:
                ax.set_title(layer)
        axes[-1][0].set_xlabel("Timestep")
    fig.suptitle("Mean spike rate per timestep (correct predictions only)", y=1.01)
    _save(fig, path)


@_run_style
def stimulus_response(responses: dict, names: list[str], path: Path):
    """One test sample per class: layer spike rates and output membrane over time."""
    palette = sns.color_palette("muted", max(len(names), 2))
    fig, axes = plt.subplots(len(names), 2, figsize=(10, 2.5 * len(names)), squeeze=False)
    for row, name in enumerate(names):
        ax_rate, ax_mem = axes[row]
        r = responses.get(row)
        if r is None:
            for ax in axes[row]:
                ax.text(0.5, 0.5, "no sample", ha="center", va="center", transform=ax.transAxes, fontsize=8)
        else:
            for key, trace in r.items():
                if key.endswith("_spike_rate"):
                    ax_rate.plot(trace, label=key.removesuffix("_spike_rate"), linewidth=1.2)
            ax_rate.set_ylim(0, 1)
            for c in range(r["output_mem"].shape[1]):
                is_true = c == row
                ax_mem.plot(r["output_mem"][:, c], color=palette[c], alpha=1.0 if is_true else 0.15,
                            linewidth=1.8 if is_true else 0.6,
                            linestyle="--" if c == r["pred_class"] and not is_true else "-")
        ax_rate.set_ylabel(name, fontsize=8, rotation=0, labelpad=70, va="center")
    for ax in axes[-1]:
        ax.set_xlabel("Timestep")
    axes[0][0].set_title("Layer spike rates")
    axes[0][0].legend(fontsize=7, loc="upper right")
    axes[0][1].set_title("Output membrane (true class bold, wrong prediction dashed)")
    fig.suptitle("Stimulus response per class", y=1.005)
    _save(fig, path)


# --- run summary (main.py) --------------------------------------------------

@_run_style
def accuracy_distribution(accs: list[float], title: str, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.stripplot(x=accs, ax=ax, size=10, jitter=False, color="steelblue")
    ax.axvline(np.mean(accs), color="red", linestyle="--", label=f"Mean={np.mean(accs):.4f}")
    ax.set(title=f"Trial accuracy distribution: {title}", xlabel="Accuracy")
    ax.xaxis.set_major_formatter(PCT1)
    ax.legend()
    _save(fig, path)


@_run_style
def per_class_heatmap(pcm: np.ndarray, names: list[str], title: str, path: Path):
    df = pd.DataFrame(pcm, columns=names, index=[f"Trial {i + 1}" for i in range(len(pcm))])
    fig, ax = plt.subplots(figsize=(max(8, int(len(names) * 1.2)), max(5, int(len(pcm) * 0.6))))
    sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, linewidths=0.4, vmin=0, vmax=1)
    ax.set(title=f"Per-class accuracy per trial: {title}", xlabel="Class", ylabel="Trial")
    _save(fig, path)


@_run_style
def per_class_bar(pcm: np.ndarray, names: list[str], title: str, path: Path):
    means, stds = pcm.mean(axis=0), pcm.std(axis=0)
    fig, ax = plt.subplots(figsize=(max(7, int(len(names) * 0.9)), 4))
    sns.barplot(x=names, y=means, hue=names, ax=ax, palette="muted", legend=False)
    ax.errorbar(range(len(names)), means, yerr=stds, fmt="none", color="black", capsize=4, linewidth=1.5)
    ax.set(title=f"Per-class accuracy (mean ± std, {len(pcm)} trials): {title}",
           xlabel="Class", ylabel="Accuracy", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(PCT)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, path)


@_run_style
def sparsity_summary(trial_logs: list[pd.DataFrame], path: Path):
    """Mean ± std sparsity per layer across trials (trials may stop at different epochs)."""
    df = pd.concat([_sparsity_long(log) for log in trial_logs])
    stats = df.groupby(["layer", "epoch"])["sparsity"].agg(mean="mean", std=lambda v: v.std(ddof=0)).reset_index()
    fig, ax = plt.subplots(figsize=(9, 4))
    for colour, (layer, sub) in zip(sns.color_palette("muted"), stats.groupby("layer")):
        ax.plot(sub["epoch"], sub["mean"], marker="o", label=layer, color=colour)
        ax.fill_between(sub["epoch"], sub["mean"] - sub["std"], sub["mean"] + sub["std"], alpha=0.2, color=colour)
    ax.set(title="Sparsity across trials (mean ± std)", xlabel="Epoch", ylabel="Sparsity")
    ax.yaxis.set_major_formatter(PCT1)
    ax.legend()
    _save(fig, path)


# ---------------------------------------------------------------------------
# PCA grid
# ---------------------------------------------------------------------------

# Every PCA type size is a fixed multiple of one base size (--fontsize).
PCA_FS = {"panel_title": 1.39, "subtitle": 1.11, "axis_label": 1.28, "tick": 1.11,
          "legend": 1.17, "fig_title": 1.89, "legend_marker": 0.61}


def load_pca(key: str):
    import analysis
    X, y, _ = analysis.load_features(key)
    Xr, blocks = analysis.representation(X, "descriptor")
    Z = analysis.standardise(Xr, blocks, np.ones(len(Xr), bool))
    pca = PCA(n_components=2, random_state=SEED).fit(Z)
    return pca.transform(Z), y, pca.explained_variance_ratio_


def subsample(y: np.ndarray, max_points: int) -> np.ndarray:
    """Class-stratified subset for drawing, so dense panels stay readable."""
    if max_points <= 0 or len(y) <= max_points:
        return np.arange(len(y))
    rng = np.random.default_rng(SEED)
    keep = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        n = max(1, round(len(idx) * max_points / len(y)))
        keep.append(rng.choice(idx, size=min(n, len(idx)), replace=False))
    return np.sort(np.concatenate(keep))


def class_ellipse(ax, pts, colour):
    """1σ covariance ellipse; skipped for classes too small to estimate one."""
    if len(pts) < 3:
        return
    vals, vecs = np.linalg.eigh(np.cov(pts.T))
    vals, vecs = vals[::-1], vecs[:, ::-1]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    w, h = 2 * np.sqrt(np.maximum(vals, 0))
    ax.add_patch(Ellipse(pts.mean(0), w, h, angle=angle, facecolor="none",
                         edgecolor=colour, lw=1.0, alpha=0.85, zorder=2))


def pca_subtitle(y, names) -> str:
    return f"{len(y)} samples · {len(names)} classes"


def subtitle_pad(fs) -> float:
    """Points between the plot box and the subtitle. The subtitle can be wider
    than the box, so it must clear half a y tick label above the top edge."""
    return 0.6 * fs["tick"] + 3


def draw_pca(ax, P, y, evr, cfg, fs, point_size, draw_idx):
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    shown = np.zeros(len(y), bool)
    shown[draw_idx] = True
    handles = []
    for c, name in enumerate(cfg.class_names):
        m = y == c
        colour = CLASS_COLORS[c % len(CLASS_COLORS)]
        if m.any():
            ax.scatter(P[m & shown, 0], P[m & shown, 1], s=point_size, c=colour, lw=0, alpha=0.5, zorder=2)
            class_ellipse(ax, P[m], colour)
            ax.scatter(*P[m].mean(0), s=point_size * 6.5, c=colour, lw=1.1, edgecolor="white", zorder=4)
        handles.append(plt.Line2D([], [], ls="none", marker="o", color=colour, markersize=fs["legend_marker"],
                                  markeredgecolor="white", markeredgewidth=0.5, label=name))
    # Two lines at two sizes: dataset name, then a smaller count line.
    sub_pad = subtitle_pad(fs)
    ax.set_title(pca_subtitle(y, cfg.class_names), fontsize=fs["subtitle"], pad=sub_pad, color=INK_2)
    ax.annotate(cfg.display_name, xy=(0.5, 1), xycoords="axes fraction",
                xytext=(0, sub_pad + 1.35 * fs["subtitle"]), textcoords="offset points",
                ha="center", va="bottom", fontsize=fs["panel_title"], color=INK)
    ax.set_xlabel(f"PC1 ({evr[0]:.1%})", fontsize=fs["axis_label"], labelpad=3)
    ax.set_ylabel(f"PC2 ({evr[1]:.1%})", fontsize=fs["axis_label"], labelpad=3)
    ax.tick_params(labelsize=fs["tick"], length=3, pad=2)
    ax.grid(alpha=0.55)
    ax.set_axisbelow(True)
    return handles


# Legend geometry, in units of the legend font size (matplotlib's defaults for
# handle length, plus the pads passed to ax.legend below).
LEG_HANDLE, LEG_TEXTPAD, LEG_COLSPACE, LEG_LABELSPACE = 2.0, 0.35, 1.0, 0.3


def legend_rows(n: int, ncol: int) -> int:
    return int(np.ceil(n / ncol))


def legend_w(names, fs: float, ncol: int) -> float:
    """Width in inches of a legend laid out column-major in ncol columns, as
    matplotlib fills it. Every column is as wide as its own longest label."""
    rows = legend_rows(len(names), ncol)
    cols = [names[j * rows:(j + 1) * rows] for j in range(ncol)]
    cols = [c for c in cols if c]
    return (sum((LEG_HANDLE + LEG_TEXTPAD) * fs / 72 + max(text_w(str(s), fs) for s in c) for c in cols)
            + (len(cols) - 1) * LEG_COLSPACE * fs / 72)


def place_legends(widths, centres, lo, hi, gap):
    """Left edges for a row of legends: as close to centred under their panels
    as possible, nudged sideways just enough to keep `gap` between neighbours
    and stay inside [lo, hi]. None if they cannot all fit."""
    if sum(widths) + gap * (len(widths) - 1) > hi - lo:
        return None
    left = [c - w / 2 for c, w in zip(centres, widths)]
    for i in range(len(left)):                       # push right past the previous key
        left[i] = max(left[i], lo if i == 0 else left[i - 1] + widths[i - 1] + gap)
    for i in reversed(range(len(left))):              # then back inside the right edge
        left[i] = min(left[i], hi - widths[i] if i == len(left) - 1 else left[i + 1] - gap - widths[i])
    return left


def fit_row_legends(names_list, fs, centres, lo, hi, max_shift, gap=0.6, max_cols=6):
    """(column counts, left edges) for a row of per-panel legends.

    Only the tallest keys set the row's height, so only they are widened, one
    row-saving step at a time, while every key still fits between its
    neighbours and its centre stays within max_shift of its own panel's centre
    (so each key still reads as belonging to the panel above it).
    """
    ncol = [1] * len(names_list)

    def layout(cols):
        w = [legend_w(nm, fs, n) for nm, n in zip(names_list, cols)]
        left = place_legends(w, centres, lo, hi, gap)
        if left is None or any(abs(l + wi / 2 - c) > max_shift for l, wi, c in zip(left, w, centres)):
            return None
        return left

    best = layout(ncol)
    while True:
        rows = [legend_rows(len(nm), n) for nm, n in zip(names_list, ncol)]
        trial = list(ncol)
        for i, nm in enumerate(names_list):
            if rows[i] == max(rows):
                n = trial[i] + 1
                while n <= min(max_cols, len(nm)) and legend_rows(len(nm), n) >= rows[i]:
                    n += 1
                if n > min(max_cols, len(nm)):
                    return ncol, best
                trial[i] = n
        left = layout(trial)
        if left is None:
            return ncol, best
        ncol, best = trial, left


def run_pca(args):
    import analysis
    import main as sb
    keys = pick([k for k in sb.DATASETS if analysis.feature_path(k).exists()], args.rows, args.cols)
    missing = [k for k in sb.DATASETS if not analysis.feature_path(k).exists()]
    if missing:
        print(f"No feature cache for {missing}. Run `python analysis.py data --dataset <key>`")
    if not keys:
        return

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": GRID,
        "axes.linewidth": 0.8, "axes.labelcolor": INK_2, "text.color": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    pieces = []
    for key in keys:
        P, y, evr = load_pca(key)
        cfg = sb.DATASETS[key]
        pieces.append((P, y, evr, cfg))
        print(f"  {key:24s} {len(y):6d} samples  {len(cfg.class_names):2d} classes  "
              f"PC1 {evr[0]:.1%}  PC2 {evr[1]:.1%}")

    fs = {k: v * args.fontsize for k, v in PCA_FS.items()}
    pt = 1 / 72
    title_h = (subtitle_pad(fs) + 1.3 * fs["subtitle"] + 1.3 * fs["panel_title"] + 4) * pt
    # Left of the box: widest likely tick label, tick pad, rotated axis label.
    y_lbl = text_w("−2.5", fs["tick"]) + (2 + 1.2 * fs["axis_label"] + 3 + 4) * pt
    # Below the box: tick label height, tick pad, axis label.
    x_lbl = (1.2 * fs["tick"] + 2 + 1.2 * fs["axis_label"] + 3 + 4) * pt
    header = 1.5 * fs["fig_title"] * pt + 0.11
    gutter_x, gutter_y, pad = 0.25, 0.30, 0.10

    # Titles are centred on their plot box. Rather than widen the gutters until
    # titles clear each other, grow the box itself into that width.
    title_w = max(max(text_w(cfg.display_name, fs["panel_title"]),
                      text_w(pca_subtitle(y, cfg.class_names), fs["subtitle"])) for _, y, _, cfg in pieces)
    groups = row_groups(len(pieces), args.rows, args.cols)
    widest = max(map(len, groups))
    if args.fit_width is None:
        cell = max(args.cell, title_w + 0.25 - y_lbl - gutter_x)
    else:
        cell = (args.fit_width - 2 * pad - gutter_x * (widest - 1) - y_lbl * widest) / widest
        gutter_x = max(gutter_x, title_w + 0.25 - y_lbl - cell)
    right_extra = max(0.0, (title_w - cell) / 2)   # last title's overhang past its box
    panel_w = y_lbl + cell
    pitch = panel_w + gutter_x
    content_w = widest * panel_w + gutter_x * (widest - 1)

    # A row whose panels share a class list (the ANTShapes variants) gets one
    # legend centred beneath it; otherwise each panel gets its own, sized jointly.
    shared = [len(g) > 1 and len({tuple(pieces[i][3].class_names) for i in g}) == 1 for g in groups]
    lf = fs["legend"]
    line_h = (1 + LEG_LABELSPACE) * lf * pt * 1.08
    fig_w = 2 * pad + content_w + right_extra
    n_cols, legend_left, legend_h = [0] * len(pieces), [0.0] * len(pieces), [0.0] * len(groups)
    for r, (g, sh) in enumerate(zip(groups, shared)):
        names = [list(pieces[i][3].class_names) for i in g]
        row_w = len(g) * panel_w + gutter_x * (len(g) - 1)
        x0 = pad + (content_w - row_w) / 2
        centres = [x0 + k * pitch + y_lbl + cell / 2 for k in range(len(g))]
        if sh:
            n = 1
            while n < min(6, len(names[0])) and legend_w(names[0], lf, n + 1) <= content_w:
                n += 1
            mid = (centres[0] + centres[-1]) / 2
            for i in g:
                n_cols[i], legend_left[i] = n, mid - legend_w(names[0], lf, n) / 2
        else:
            cols, lefts = fit_row_legends(names, lf, centres, pad, fig_w - pad, max_shift=pitch / 2 - 0.2)
            for i, n, l in zip(g, cols, lefts):
                n_cols[i], legend_left[i] = n, l
        legend_h[r] = 0.10 + line_h * max(legend_rows(len(pieces[i][3].class_names), n_cols[i]) for i in g)

    row_h = [title_h + cell + x_lbl + legend_h[r] for r in range(len(groups))]
    fig_h = 2 * pad + header + sum(row_h) + gutter_y * (len(groups) - 1)

    fig = plt.figure(figsize=(fig_w, fig_h))
    kw = dict(loc="upper left", frameon=False, fontsize=lf, handlelength=LEG_HANDLE,
              handletextpad=LEG_TEXTPAD, columnspacing=LEG_COLSPACE, labelspacing=LEG_LABELSPACE,
              borderpad=0.0, borderaxespad=0.0, bbox_transform=fig.transFigure)
    for r, (g, sh) in enumerate(zip(groups, shared)):
        x = pad + (content_w - (len(g) * panel_w + gutter_x * (len(g) - 1))) / 2
        top = fig_h - pad - header - sum(row_h[:r]) - gutter_y * r
        axes = []
        for i in g:
            P, y, evr, cfg = pieces[i]
            ax = fig.add_axes(((x + y_lbl) / fig_w, (top - title_h - cell) / fig_h, cell / fig_w, cell / fig_h))
            handles = draw_pca(ax, P, y, evr, cfg, fs, args.point_size, subsample(y, args.max_points))
            axes.append((ax, handles))
            x += pitch
        legend_top = (top - title_h - cell - x_lbl - 0.10) / fig_h
        legends = [axes[k][0].legend(handles=axes[k][1], bbox_to_anchor=(legend_left[i] / fig_w, legend_top),
                                     ncol=n_cols[i], **kw)
                   for k, i in enumerate(g[:1] if sh else g)]   # a shared row draws one legend
        if len(legends) > 1:
            # Keys can be nudged off-centre to save height, so rule a grey line
            # midway between neighbours to show which key belongs to which panel.
            # Positions come from the rendered extents, not the width estimates.
            fig.canvas.draw()
            boxes = [lg.get_window_extent().transformed(fig.transFigure.inverted()) for lg in legends]
            y0 = min(b.y0 for b in boxes) - 0.04 / fig_h
            y1 = legend_top + 0.04 / fig_h
            for a, b in zip(boxes, boxes[1:]):
                xm = (a.x1 + b.x0) / 2
                fig.add_artist(plt.Line2D([xm, xm], [y0, y1], transform=fig.transFigure,
                                          color=LEGEND_RULE, lw=1.6, solid_capstyle="butt"))

    fig.text(0.10 / fig_w, 1 - 0.12 / fig_h, "Leading PCA plane of the event descriptor, by dataset",
             fontsize=fs["fig_title"], fontweight="bold", va="top")
    save(fig, "pca_grid", args.dpi, tight=True)


# ---------------------------------------------------------------------------
# Confusion grid
# ---------------------------------------------------------------------------

def load_confusion(run: Path, cfg) -> tuple[np.ndarray, str]:
    """(mean row-normalised matrix over trials, accuracy subtitle)."""
    import main as sb
    mats, acc = [], []
    for f in sorted(run.glob("trial_*/confusion_matrix.csv")):
        df = pd.read_csv(f, index_col=0)
        df.index, df.columns = df.index.astype(str), df.columns.astype(str)
        unknown = (set(df.index) | set(df.columns)) - set(cfg.class_names)
        if unknown:
            raise ValueError(f"{f}: class labels {sorted(unknown)} not in {cfg.display_name}'s class list")
        cm = df.reindex(index=cfg.class_names, columns=cfg.class_names, fill_value=0).to_numpy(float)
        mats.append(cm / cm.sum(1, keepdims=True).clip(min=1))
        acc.append(sb.chance_scores(cm)["accuracy"])
    acc = np.array(acc)
    return np.mean(mats, 0), f"{acc.mean():.1%} ± {acc.std():.1%}  ({len(acc)} trials)"


def run_confusion(args):
    import analysis
    import main as sb
    found = {k: analysis.latest_run(args.results, k) for k in sb.DATASETS}
    keys = pick([k for k, r in found.items() if r is not None], args.rows, args.cols)
    if not keys:
        print(f"No runs with trial_*/confusion_matrix.csv under {args.results}/")
        return

    sns.set_theme(style="white")
    pieces = []
    for key in keys:
        cfg = sb.DATASETS[key]
        pieces.append((cfg, *load_confusion(found[key], cfg)))
        print(f"  {key:24s} {found[key].name}  ({len(cfg.class_names)} classes)")

    tick = args.fontsize
    title_fs, label_fs, annot_fs = tick * 1.2, tick * 1.1, args.annot_fontsize
    pt = 1 / 72
    title_h = 2 * 1.3 * title_fs * pt + 8 * pt              # two lines + title pad
    axis_lbl = 1.2 * label_fs * pt + 5 * pt                 # "Predicted" / "True" + labelpad
    gutter_x, gutter_y, pad = 0.25, 0.25, 0.08

    # (y tick label width, height of 45° x tick labels) per panel, measured: a
    # 45° label spans (text width + text height) · sin 45° vertically.
    ext = []
    for cfg, _, _ in pieces:
        longest = max(text_w(n, tick) for n in cfg.class_names)
        ext.append((longest + 4 * pt, (longest + 1.2 * tick * pt) * 0.7071 + 4 * pt))

    groups = row_groups(len(pieces), args.rows, args.cols)
    n_cls = [len(cfg.class_names) for cfg, _, _ in pieces]
    # The colourbar goes in the last row's empty slot when there is one, rather
    # than in a strip down the whole right-hand side.
    inline_cbar = len(groups) > 1 and len(groups[-1]) < args.cols

    label = "Mean rate (row-normalised per trial, then averaged over trials)"
    cbar_bar, line_h = 0.24, 1.25 * label_fs * pt

    def cbar_block(height):   # (wrapped label, block width) for a bar of this height
        wrapped = wrap_to(label, label_fs, height)
        n_lines = wrapped.count("\n") + 1
        return wrapped, cbar_bar + 6 * pt + text_w("100%", tick) + 10 * pt + n_lines * line_h

    cell = args.cell
    if args.fit_width is not None:   # binding row is the one needing the smallest cell
        _, strip = cbar_block(4.0)
        cell = min((args.fit_width - 2 * pad - (0 if inline_cbar else strip + gutter_x)
                    - sum(ext[i][0] + axis_lbl for i in g) - gutter_x * (len(g) - 1))
                   / sum(n_cls[i] for i in g) for g in groups)
    if cell < text_w("100", annot_fs) + 0.04:
        print(f"WARNING: cell {cell:.3f} in is too small for '100' at {annot_fs:g} pt; "
              f"use a wider --fit-width or a smaller --annot-fontsize.")

    heat = [n * cell for n in n_cls]
    # A small matrix can be narrower than its own title; the title is centred
    # on the matrix, so pad the slot on both sides until it fits.
    title_pad = [max(0.0, (max(text_w(cfg.display_name, title_fs), text_w(sub, title_fs)) - heat[i]) / 2)
                 for i, (cfg, _, sub) in enumerate(pieces)]
    lead = [max(ext[i][0] + axis_lbl, title_pad[i]) for i in range(len(pieces))]
    widths = [lead[i] + heat[i] + title_pad[i] for i in range(len(pieces))]
    row_w = [sum(widths[i] for i in g) + gutter_x * (len(g) - 1) for g in groups]
    row_h = [title_h + max(heat[i] for i in g) + max(ext[i][1] for i in g) + axis_lbl for g in groups]
    fig_h = 2 * pad + sum(row_h) + gutter_y * (len(groups) - 1)

    if inline_cbar:
        cbar_h = max(heat[i] for i in groups[-1])
        cbar_label, cbar_w = cbar_block(cbar_h)
        row_w[-1] += gutter_x + cbar_w
        content_w = max(row_w)
        fig_w = 2 * pad + content_w
    else:
        cbar_h = 0.8 * (fig_h - 2 * pad)
        cbar_label, cbar_w = cbar_block(cbar_h)
        content_w = max(row_w)
        fig_w = 2 * pad + content_w + gutter_x + cbar_w

    fig = plt.figure(figsize=(fig_w, fig_h))
    for r, g in enumerate(groups):
        items = len(g) + (inline_cbar and r == len(groups) - 1)
        # Justify a short row toward the full width (gutter capped at 4×).
        slack, gut = content_w - row_w[r], gutter_x
        if slack > 0 and items > 1:
            gut = min(gutter_x + slack / (items - 1), gutter_x * 4)
            slack = content_w - (row_w[r] + (gut - gutter_x) * (items - 1))
        x = pad + max(slack, 0) / 2
        top = fig_h - pad - sum(row_h[:r]) - gutter_y * r
        for i in g:
            cfg, mat, subtitle = pieces[i]
            left = x + lead[i]
            ax = fig.add_axes((left / fig_w, (top - title_h - heat[i]) / fig_h, heat[i] / fig_w, heat[i] / fig_h))
            pct = mat * 100
            annot = np.where(pct.round() >= 1, np.round(pct).astype(int).astype(str), "")
            sns.heatmap(mat, ax=ax, cmap=CMAP, vmin=0, vmax=1, cbar=False, annot=annot, fmt="",
                        annot_kws={"fontsize": annot_fs}, xticklabels=cfg.class_names,
                        yticklabels=cfg.class_names, linewidths=0.4, linecolor="white", square=True)
            ax.set_title(f"{cfg.display_name}\n{subtitle}", fontsize=title_fs, pad=6, linespacing=1.3)
            ax.tick_params(length=0, pad=2)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=tick)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=tick)
            ax.set_xlabel("Predicted", fontsize=label_fs, labelpad=4)
            ax.set_ylabel("True", fontsize=label_fs, labelpad=4)
            x += widths[i] + gut
        if inline_cbar and r == len(groups) - 1:
            cax_rect = (x, top - title_h - cbar_h)

    if not inline_cbar:
        cax_rect = (pad + content_w + gutter_x, (fig_h - cbar_h) / 2)
    cax = fig.add_axes((cax_rect[0] / fig_w, cax_rect[1] / fig_h, cbar_bar / fig_w, cbar_h / fig_h))
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(0, 1), cmap=CMAP), cax=cax)
    cbar.set_label(cbar_label, fontsize=label_fs, labelpad=10, linespacing=1.25)
    ticks = np.linspace(0, 1, 6)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{v:.0%}" for v in ticks], fontsize=tick)
    cbar.ax.tick_params(pad=6)
    cbar.outline.set_linewidth(0.5)
    save(fig, "confusion_grid", args.dpi, tight=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, rows, cols, cell, fontsize, cell_help in (
            ("pca", 2, 4, 3.0, 22.0, "inches per plot box"),
            ("confusion", 3, 3, 0.55, 21.0, "inches per matrix cell")):
        s = sub.add_parser(name)
        s.add_argument("--rows", type=int, default=rows)
        s.add_argument("--cols", type=int, default=cols)
        s.add_argument("--cell", type=float, default=cell, help=cell_help + ", uniform across panels")
        s.add_argument("--fit-width", type=float, default=None, metavar="INCHES",
                       help="override --cell to make the figure this wide (A3 landscape 16.54)")
        s.add_argument("--fontsize", type=float, default=fontsize,
                       help="base type size (tick labels); every other size follows it")
        s.add_argument("--dpi", type=int, default=200)
    sub.choices["pca"].add_argument("--point-size", type=float, default=5.0)
    sub.choices["pca"].add_argument("--max-points", type=int, default=4000, metavar="N",
                                    help="class-stratified draw cap per panel; 0 draws all")
    sub.choices["confusion"].add_argument("--annot-fontsize", type=float, default=18.0)
    sub.choices["confusion"].add_argument("--results", type=Path, default=Path("results"))
    args = p.parse_args()
    {"pca": run_pca, "confusion": run_confusion}[args.cmd](args)


if __name__ == "__main__":
    main()
