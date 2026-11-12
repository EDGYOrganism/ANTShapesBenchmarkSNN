"""convolutional SNN benchmark on event-based vision datasets.

Trains a spiking ConvSNN with surrogate-gradient BPTT over N_TRIALS seeds and
writes per-trial and summary results to results/{dataset}/{run_id}/.

    python main.py --dataset antshapes_vanilla
    python main.py --dataset all --epochs 30
"""

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import snntorch as snn
import tonic
import tonic.transforms as T
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from snntorch import surrogate
from torch.utils.data import DataLoader, Dataset

import make_figures

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR     = Path(__file__).resolve().parent
DATASET_ROOT = Path(os.environ.get("SNNBENCH_DATA_ROOT", REPO_DIR / "data")).expanduser()
CACHE_ROOT   = Path(os.environ.get("SNNBENCH_CACHE_ROOT", REPO_DIR / "cache")).expanduser()
SPLIT_FILE   = "split.json"   # train/test split, stored inside each dataset folder
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_TRIALS          = 10
TRIAL_SEED_SCALE  = 42      # seed for trial i = i * 42 + 7
TRIAL_SEED_OFFSET = 7
NUM_WORKERS       = 8

DENOISE_FILTER_TIME_US = 10_000   # drop events with no spatial neighbour within this window

SURROGATE_SLOPE = 25      # fast_sigmoid slope
LIF_BETA        = 0.5     # membrane decay (hidden LIF and output integrator)
# snnTorch's default threshold of 1.0 leaves the deeper conv stages silent on
# small/short datasets (POKERDVS never leaves chance); 0.5 lets them fire.
LIF_THRESHOLD   = 0.5

GRAD_CLIP_NORM = 1.0
EARLY_STOP_ACC = 0.95     # stop a trial once test accuracy reaches this

DEFAULT_EPOCHS     = 50
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR         = 1e-3

# ---------------------------------------------------------------------------
# Event readers and split-file datasets
# ---------------------------------------------------------------------------

EVENT_DTYPE = np.dtype([("t", np.int64), ("x", np.int16), ("y", np.int16), ("p", bool)])

SHAPE_NAMES = ["Cube", "Sphere", "Icosphere", "Cylinder", "Pyramid", "Cone",
               "Capsule", "Torus", "L-Block", "T-Block", "Teapot", "Suzanne"]
CIFAR10_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
                 "dog", "frog", "horse", "ship", "truck"]


def read_antshapes(sample_dir: Path) -> tuple[np.ndarray, int]:
    """Decode one ANTShapes export folder (events.bin + objects.bin).

    events.bin: 7 big-endian bytes per event, a 56-bit word with
      bits 0-22 t (µs), bit 23 p, bits 24-39 y, bits 40-55 x.
    objects.bin: the shape class is the first byte of the first entry.
    """
    raw = np.frombuffer((sample_dir / "events.bin").read_bytes(), dtype=np.uint8)
    raw = raw[: len(raw) // 7 * 7].reshape(-1, 7).astype(np.uint64)
    word = np.zeros(len(raw), dtype=np.uint64)
    for i in range(7):
        word = (word << np.uint64(8)) | raw[:, i]
    events = np.empty(len(raw), dtype=EVENT_DTYPE)
    events["t"] = word & np.uint64(0x7FFFFF)
    events["p"] = (word >> np.uint64(23)) & np.uint64(1)
    events["y"] = (word >> np.uint64(24)) & np.uint64(0xFFFF)
    events["x"] = (word >> np.uint64(40)) & np.uint64(0xFFFF)
    label = (sample_dir / "objects.bin").read_bytes()[0]
    return events, int(label)


def read_cifar10dvs(path: Path) -> tuple[np.ndarray, int]:
    """Decode an original-release CIFAR10-DVS AEDAT 2.0 file (DVS128 address layout)."""
    from tonic.io import get_aer_events_from_file, read_aedat_header_from_file

    _, data_start, _ = read_aedat_header_from_file(str(path))
    raw = get_aer_events_from_file(str(path), 2.0, data_start)
    addr = raw["address"].astype(np.uint32)
    events = np.empty(len(addr), dtype=EVENT_DTYPE)
    events["t"] = raw["timeStamp"]
    events["x"] = (addr & 0xFE) >> 1
    events["y"] = (addr & 0x7F00) >> 8
    events["p"] = addr & 0x1
    return events, CIFAR10_NAMES.index(path.parent.name)


class SplitDataset(Dataset):
    """Samples listed under "train" or "test" in root/split.json, relative to root."""

    def __init__(self, root: Path, train: bool, read: Callable, transform: Callable):
        split_file = root / SPLIT_FILE
        if not split_file.exists():
            raise FileNotFoundError(f"{split_file} not found; this dataset needs its train/test split file.")
        spec = json.loads(split_file.read_text())
        self.items = spec["train" if train else "test"]
        self.root, self.read, self.transform = root, read, transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        events, label = self.read(self.root / self.items[idx])
        return self.transform(events), label


def _split_dataset(subdir: str, read: Callable):
    return lambda train, transform: SplitDataset(DATASET_ROOT / subdir, train, read, transform)


def _tonic_dataset(cls, manual: bool = False):
    if manual:
        # tonic may not recognise manually placed data and would try to download it.
        cls.download = lambda self: None
    return lambda train, transform: cls(save_to=str(DATASET_ROOT), train=train, transform=transform)


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DatasetConfig:
    display_name: str
    data_dir: str             # directory under DATASET_ROOT
    sensor_size: tuple        # (W, H, polarities)
    class_names: list
    time_window_us: int       # ToFrame bin width; keeps binarised frames sparse
    denoise: bool             # apply tonic Denoise (off for clean / simulated data)
    build: Callable           # (train, transform) -> Dataset
    setup: str = ""           # how to obtain the data, shown if data_dir is missing


DATASETS: dict[str, DatasetConfig] = {
    "nmnist": DatasetConfig(
        "N-MNIST", "NMNIST", (34, 34, 2), [str(i) for i in range(10)],
        10_000, True, _tonic_dataset(tonic.datasets.NMNIST),
    ),
    "pokerdvs": DatasetConfig(
        "POKERDVS", "POKERDVS", (35, 35, 2), ["clubs", "hearts", "diamonds", "spades"],
        2_000, False, _tonic_dataset(tonic.datasets.POKERDVS, manual=True),
        setup="Download http://www2.imse-cnm.csic.es/caviar/POKER_DVS/poker_dvs.tar.gz and run\n"
              "  python convert_pokerdvs.py --tar poker_dvs.tar.gz",
    ),
    "dvsgesture": DatasetConfig(
        "DVS Gesture", "DVSGesture", (128, 128, 2),
        ["Hand clapping", "Right wave", "Left wave", "Right arm CW", "Right arm CCW",
         "Left arm CW", "Left arm CCW", "Arm roll", "Air drums", "Air guitar", "Other"],
        50_000, True, _tonic_dataset(tonic.datasets.DVSGesture, manual=True),
        setup="Download and extract both archives into this directory:\n"
              "  https://figshare.com/ndownloader/files/38022171 (train)\n"
              "  https://figshare.com/ndownloader/files/38020584 (test)",
    ),
    "cifar10dvs": DatasetConfig(
        "CIFAR10-DVS", "CIFAR10DVS", (128, 128, 2), CIFAR10_NAMES,
        10_000, True, _split_dataset("CIFAR10DVS", read_cifar10dvs),
        setup="Extract the original AEDAT 2.0 release as CIFAR10DVS/{class}/{class}/*.aedat, plus split.json",
    ),
    **{
        f"antshapes_{v.lower()}": DatasetConfig(
            f"ANTShapes ({'Standard' if v == 'VANILLA' else v.title()})", v, (128, 128, 2), SHAPE_NAMES,
            10_000, False, _split_dataset(v, read_antshapes),
            setup=f"Place the ANTShapes {v} export here ({{id}}/events.bin, objects.bin, plus split.json).",
        )
        for v in ("VANILLA", "DISTORTION", "ROTATION", "TRANSLATION")
    },
}


def check_data(cfg: DatasetConfig):
    """Fail early with setup instructions. Datasets without `setup` are downloaded by tonic."""
    data_dir = DATASET_ROOT / cfg.data_dir
    if cfg.setup and (not data_dir.is_dir() or not any(data_dir.iterdir())):
        raise FileNotFoundError(f"{cfg.display_name} not found at {data_dir}.\n{cfg.setup}")


def split_tag(cfg: DatasetConfig) -> str:
    """Short hash of the dataset's split file ("" if it has none), so caches
    built from a different split are never reused."""
    split_file = DATASET_ROOT / cfg.data_dir / SPLIT_FILE
    return "_" + hashlib.sha1(split_file.read_bytes()).hexdigest()[:8] if split_file.exists() else ""


def build_loaders(key: str, cfg: DatasetConfig, batch_size: int, clear_cache: bool):
    check_data(cfg)
    transform = T.Compose([
        *([T.Denoise(filter_time=DENOISE_FILTER_TIME_US)] if cfg.denoise else []),
        # include_incomplete keeps the trailing partial window so no events are dropped.
        T.ToFrame(sensor_size=cfg.sensor_size, time_window=cfg.time_window_us, include_incomplete=True),
    ])

    # Frames are cached to disk by index. The tag covers every preprocessing
    # parameter and the split file, so changing either builds a fresh cache.
    tag = f"{key}_tw{cfg.time_window_us}_{'dn' if cfg.denoise else 'nodn'}{split_tag(cfg)}"

    loaders = []
    for train in (True, False):
        cache = CACHE_ROOT / tag / ("train" if train else "test")
        if clear_cache and cache.exists():
            shutil.rmtree(cache)
        dataset = tonic.DiskCachedDataset(cfg.build(train, transform), cache_path=str(cache))
        loaders.append(DataLoader(
            dataset, batch_size=batch_size, shuffle=train, num_workers=NUM_WORKERS,
            collate_fn=tonic.collation.PadTensors(batch_first=False),
        ))
    return loaders


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ConvSNN(nn.Module):
    """Up to three conv → adaptive-max-pool → LIF stages, then FC → leaky integrator.

    Each stage pools to max(size // 4, 4). A further stage is added only while
    the map is larger than 4×4 (a 5×5 kernel on a 4×4 map is global):
        128×128 → 32 → 8 → 4   (3 stages, FC input 64·4·4)
         34×34  →  8 → 4       (2 stages, FC input 32·4·4)
    The output layer is a non-spiking leaky integrator, one unit per class:
    mem[t] = β·mem[t−1] + fc(x[t]). Classification reads its peak over time.
    """

    def __init__(self, in_channels: int, n_classes: int, spatial: int, beta: float):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=SURROGATE_SLOPE)
        self.convs, self.pools, self.lifs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        size, ch_in = spatial, in_channels
        for ch_out in (16, 32, 64):
            size = max(size // 4, 4)
            self.convs.append(nn.Conv2d(ch_in, ch_out, kernel_size=5, padding=2))
            self.pools.append(nn.AdaptiveMaxPool2d(size))
            self.lifs.append(snn.Leaky(beta=beta, threshold=LIF_THRESHOLD, spike_grad=spike_grad))
            ch_in = ch_out
            if size == 4:
                break
        self.fc = nn.Linear(ch_in * 4 * 4, n_classes)
        self.beta = beta

    @property
    def layer_names(self) -> list[str]:
        return [f"conv{i + 1}" for i in range(len(self.convs))]

    def forward(self, x: torch.Tensor):
        """x: (T, B, C, H, W) binary.

        Returns mem_rec (T, B, n_classes), the output membrane at every step,
        and rates {layer: (T, B)}, the fraction of each sample's neurons that spiked.
        """
        mems = [lif.init_leaky() for lif in self.lifs]
        mem_out, mem_rec = 0.0, []
        rates = [[] for _ in self.lifs]
        for xt in x:
            h = xt
            for i, (conv, pool, lif) in enumerate(zip(self.convs, self.pools, self.lifs)):
                h, mems[i] = lif(pool(conv(h)), mems[i])
                rates[i].append(h.detach().mean(dim=(1, 2, 3)))
            mem_out = self.beta * mem_out + self.fc(h.flatten(1))
            mem_rec.append(mem_out)
        return torch.stack(mem_rec), {n: torch.stack(r) for n, r in zip(self.layer_names, rates)}


def max_membrane_loss(mem_rec: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on each class's peak output membrane over time."""
    return F.cross_entropy(mem_rec.max(dim=0).values, targets)


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimiser) -> float:
    model.train()
    total = 0.0
    for data, targets in loader:
        data, targets = (data.to(DEVICE) > 0).float(), targets.to(DEVICE)
        optimiser.zero_grad()
        loss = max_membrane_loss(model(data)[0], targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimiser.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def evaluate(model, loader, n_classes: int, record: bool = False) -> dict:
    """Accuracy, loss, predictions and per-layer sparsity over a loader.

    With record=True, also returns:
      class_activity[c][layer]: mean spike rate per timestep over correctly
                                classified samples of class c.
      responses[c]: spike rates and output membrane for the first test sample of class c.
    """
    model.eval()
    loss = 0.0
    preds, targets_all = [], []
    sparsity: dict[str, list[float]] = {n: [] for n in model.layer_names}
    activity = {c: {n: [] for n in model.layer_names} for c in range(n_classes)}
    responses: dict[int, dict] = {}

    for data, targets in loader:
        data, targets = (data.to(DEVICE) > 0).float(), targets.to(DEVICE)
        mem_rec, rates = model(data)
        loss += max_membrane_loss(mem_rec, targets).item()
        pred = mem_rec.max(dim=0).values.argmax(dim=1)
        preds.append(pred.cpu())
        targets_all.append(targets.cpu())
        for name, r in rates.items():
            sparsity[name].append(1.0 - r.mean().item())
        if not record:
            continue
        rates_np = {n: r.cpu().numpy() for n, r in rates.items()}
        mem_np = mem_rec.cpu().numpy()
        for b, (c, p) in enumerate(zip(targets.tolist(), pred.tolist())):
            if c == p:
                for name, r in rates_np.items():
                    activity[c][name].append(r[:, b])
            if c not in responses:
                out = mem_np[:, b]
                responses[c] = {**{f"{n}_spike_rate": r[:, b] for n, r in rates_np.items()},
                                "output_mem": out, "pred_class": int(out.max(axis=0).argmax())}

    preds, targets_all = torch.cat(preds).numpy(), torch.cat(targets_all).numpy()
    result = {
        "acc": float((preds == targets_all).mean()),
        "loss": loss / len(loader),
        "preds": preds,
        "targets": targets_all,
        "sparsity": {n: float(np.mean(v)) for n, v in sparsity.items()},
    }
    if record:
        def mean_trace(seqs):
            if not seqs:
                return np.zeros(1)
            n = min(len(s) for s in seqs)
            return np.mean([s[:n] for s in seqs], axis=0)
        result["class_activity"] = {c: {n: mean_trace(s) for n, s in layers.items()}
                                    for c, layers in activity.items()}
        result["responses"] = responses
    return result


def run_trial(trial: int, cfg: DatasetConfig, loaders, args, trial_dir: Path):
    seed = trial * TRIAL_SEED_SCALE + TRIAL_SEED_OFFSET
    torch.manual_seed(seed)
    train_loader, test_loader = loaders
    n_classes = len(cfg.class_names)

    model = ConvSNN(cfg.sensor_size[2], n_classes, cfg.sensor_size[0], args.beta).to(DEVICE)
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)
    print(f"\n  Trial {trial + 1}/{N_TRIALS}  (seed={seed}, beta={args.beta}, lr={args.lr})")

    log, best_acc, best_state = [], -1.0, None
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser)
        scheduler.step()
        ev = evaluate(model, test_loader, n_classes)
        log.append({"epoch": epoch, "train_loss": train_loss, "val_loss": ev["loss"],
                    "val_acc": ev["acc"], **{f"sparsity_{k}": v for k, v in ev["sparsity"].items()}})
        print(f"    epoch {epoch:2d}/{args.epochs}  train={train_loss:.4f}  val_loss={ev['loss']:.4f}  "
              f"val_acc={ev['acc']:.4f}  sparsity "
              + "  ".join(f"{k}={v:.2f}" for k, v in ev["sparsity"].items()))
        if ev["acc"] > best_acc:
            best_acc, best_state = ev["acc"], {k: v.clone() for k, v in model.state_dict().items()}
        if ev["acc"] >= EARLY_STOP_ACC:
            print(f"  → Early stop at epoch {epoch}")
            break

    model.load_state_dict(best_state)
    final = evaluate(model, test_loader, n_classes, record=True)
    print(f"  → trial {trial + 1} accuracy (best epoch): {final['acc']:.4f}")
    log = pd.DataFrame(log)
    scores, per_class = export_trial(trial_dir, cfg, log, final)
    return scores, per_class, log


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def chance_scores(cm: np.ndarray) -> dict:
    """Accuracy, Cohen's κ and chance-adjusted accuracy (acc − 1/K) / (1 − 1/K)
    from a confusion matrix, so datasets with different class counts compare."""
    n, k = cm.sum(), len(cm)
    acc = np.trace(cm) / n
    expected = (cm.sum(0) * cm.sum(1)).sum() / n ** 2
    return {"accuracy": acc, "kappa": (acc - expected) / (1 - expected),
            "chance_adjusted": (acc - 1 / k) / (1 - 1 / k)}


def export_trial(trial_dir: Path, cfg: DatasetConfig, log: pd.DataFrame, final: dict):
    names = cfg.class_names
    log.to_csv(trial_dir / "training_log.csv", index=False)
    cm = confusion_matrix(final["targets"], final["preds"], labels=list(range(len(names))))
    pd.DataFrame(cm, index=names, columns=names).to_csv(trial_dir / "confusion_matrix.csv")
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    pd.DataFrame({"class": names, "accuracy": per_class, "n_samples": cm.sum(axis=1)}).to_csv(
        trial_dir / "per_class_accuracy.csv", index=False)

    stim_dir = trial_dir / "stimulus_response"
    stim_dir.mkdir(exist_ok=True)
    for c, r in final["responses"].items():
        np.savez_compressed(stim_dir / f"{names[c]}.npz", **r)

    plot_dir = trial_dir / "plots"
    make_figures.loss_curves(log, plot_dir / "loss_curves.png")
    make_figures.confusion(cm, names, plot_dir / "confusion_matrix.png")
    make_figures.sparsity(log, plot_dir / "sparsity.png")
    make_figures.spike_activity_by_class(final["class_activity"], names, plot_dir / "spike_activity_by_class.png")
    make_figures.stimulus_response(final["responses"], names, stim_dir / "response_grid.png")

    return chance_scores(cm), per_class


def export_summary(summary_dir: Path, cfg: DatasetConfig, scores: list, per_class: list, logs: list):
    summary_dir.mkdir(parents=True, exist_ok=True)
    pcm = np.stack(per_class)   # (trials, classes)
    overall = pd.DataFrame(scores)
    overall.insert(0, "trial", range(1, len(scores) + 1))
    overall.to_csv(summary_dir / "overall_accuracy.csv", index=False)
    accs = overall["accuracy"].tolist()
    rows = pd.DataFrame({"class": cfg.class_names, "mean_accuracy": pcm.mean(0), "std_accuracy": pcm.std(0),
                         "min_accuracy": pcm.min(0), "max_accuracy": pcm.max(0)})
    rows.to_csv(summary_dir / "per_class_accuracy_summary.csv", index=False)

    plot_dir = summary_dir / "plots"
    make_figures.accuracy_distribution(accs, cfg.display_name, plot_dir / "accuracy_distribution.png")
    make_figures.per_class_heatmap(pcm, cfg.class_names, cfg.display_name, plot_dir / "per_class_heatmap.png")
    make_figures.per_class_bar(pcm, cfg.class_names, cfg.display_name, plot_dir / "per_class_bar.png")
    make_figures.sparsity_summary(logs, plot_dir / "sparsity_summary.png")

    print(f"\n{'=' * 56}\n  {cfg.display_name}: {len(accs)} trials")
    print(f"  Overall  :  {np.mean(accs):.4f}  ±  {np.std(accs):.4f}  "
          f"(κ {overall['kappa'].mean():.4f}, chance-adjusted {overall['chance_adjusted'].mean():.4f})\n"
          "  Per-class (mean ± std):")
    for name, m, s in zip(cfg.class_names, rows["mean_accuracy"], rows["std_accuracy"]):
        print(f"    {name:>20s}:  {m:.4f}  ±  {s:.4f}")
    print(f"\n  Results saved to: {summary_dir.parent}\n{'=' * 56}")


# ---------------------------------------------------------------------------
# Mian
# ---------------------------------------------------------------------------

def run_dataset(key: str, args):
    cfg = DATASETS[key]
    if args.time_window_us is not None:
        cfg = replace(cfg, time_window_us=args.time_window_us)
    run_dir = Path("results") / key / datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'=' * 56}\nDevice  : {DEVICE}\n"
          f"Dataset : {cfg.display_name}  ({cfg.sensor_size[0]}×{cfg.sensor_size[1]}, "
          f"{len(cfg.class_names)} classes, {cfg.time_window_us} µs frames)\n"
          f"Config  : {N_TRIALS} trials × {args.epochs} epochs, batch={args.batch_size}, "
          f"lr={args.lr}, beta={args.beta}\nOutput  : {run_dir}/")
    loaders = build_loaders(key, cfg, args.batch_size, args.clear_cache)
    print(f"Batches - train: {len(loaders[0])}, test: {len(loaders[1])} "
          "(the first epoch builds the frame cache and is slow)")

    scores, per_class, logs = [], [], []
    for i in range(N_TRIALS):
        trial_dir = run_dir / f"trial_{i + 1:02d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        s, pc, log = run_trial(i, cfg, loaders, args, trial_dir)
        scores.append(s)
        per_class.append(pc)
        logs.append(log)
    export_summary(run_dir / "summary", cfg, scores, per_class, logs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    p.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    p.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    p.add_argument("--lr", type=float, default=DEFAULT_LR)
    p.add_argument("--beta", type=float, default=LIF_BETA, help="LIF membrane decay")
    p.add_argument("--time_window_us", type=int, default=None, help="override the dataset's frame width")
    p.add_argument("--clear_cache", action="store_true", help="rebuild the frame cache first")
    args = p.parse_args()
    for key in DATASETS if args.dataset == "all" else [args.dataset]:
        run_dataset(key, args)


if __name__ == "__main__":
    main()
