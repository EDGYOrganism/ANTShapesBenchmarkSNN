"""Convert the IMSE-CNM POKER-DVS release into tonic's POKERDVS folder layout.

    python convert_pokerdvs.py --tar poker_dvs.tar.gz

Download: http://www2.imse-cnm.csic.es/caviar/POKER_DVS/poker_dvs.tar.gz
Writes DATASET_ROOT/POKERDVS/{pips_train,pips_test}/{cl,he,di,sp}/NNNN.npy, each an
int64 (N, 4) array of (t, x, y, p). The split is 80/20 per class (numpy seed 0).
"""

import argparse
import struct
import tarfile
import tempfile
from pathlib import Path

import numpy as np
import scipy.io

from main import DATASET_ROOT

CLASS_CODES = ("cl", "he", "di", "sp")
KEYWORDS = {"club": "cl", "heart": "he", "diamond": "di", "spade": "sp",
            "cl": "cl", "he": "he", "di": "di", "sp": "sp"}


def read_mat(path: Path) -> np.ndarray:
    """Events from a MATLAB .mat file, trying the common jAER field names."""
    mat = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    td = next((v for k, v in mat.items() if not k.startswith("__")
               and (hasattr(v, "_fieldnames") or getattr(getattr(v, "dtype", None), "names", None))), None)

    def field(*names):
        for n in names:
            if td is not None and hasattr(td, n):
                return np.asarray(getattr(td, n)).ravel()
            if isinstance(td, np.ndarray) and n in (td.dtype.names or ()):
                return td[n].ravel()
            if n in mat:
                return np.asarray(mat[n]).ravel()
        raise KeyError(f"none of {names} in {path.name}")

    return np.stack([field("ts", "t", "timestamp", "timestamps"), field("x", "xaddr", "X"),
                     field("y", "yaddr", "Y"), field("p", "pol", "polarity", "Polarity")], axis=1).astype(np.int64)


def read_aedat(path: Path) -> np.ndarray:
    """Events from an AER-DAT 2.0 file: '#' header lines, then big-endian (addr, t) pairs.
    Address: bit 0 polarity, bits 1-7 x, bits 8-14 y."""
    raw = path.read_bytes()
    pos = 0
    while raw.startswith(b"#", pos):
        pos = raw.index(b"\n", pos) + 1
    events = [(t, (a >> 1) & 0x7F, (a >> 8) & 0x7F, a & 1)
              for a, t in struct.iter_unpack(">II", raw[pos: pos + (len(raw) - pos) // 8 * 8])]
    return np.array(events, dtype=np.int64).reshape(-1, 4)


def infer_class(path: Path) -> str | None:
    for part in (path.stem, *reversed(path.parts)):
        for keyword, code in KEYWORDS.items():
            if keyword in part.lower():
                return code
    return None


def convert(tar_path: Path, output: Path, train_ratio: float = 0.8, seed: int = 0):
    readers = {".npy": lambda f: np.load(f).astype(np.int64), ".mat": read_mat,
               ".aedat": read_aedat, ".dat": read_aedat}
    by_class: dict[str, list[np.ndarray]] = {c: [] for c in CLASS_CODES}

    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(tar_path) as tf:
            tf.extractall(tmp, filter="data")
        files = sorted(Path(tmp).rglob("*"))
        # Prefer ready-made .npy, then .mat, then raw AER files.
        for exts in ((".npy",), (".mat",), (".aedat", ".dat")):
            chosen = [f for f in files if f.suffix in exts]
            if chosen:
                break
        else:
            raise RuntimeError(f"No .npy/.mat/.aedat files in {tar_path}")
        print(f"Reading {len(chosen)} {chosen[0].suffix} files")

        for f in chosen:
            cls = infer_class(f)
            try:
                arr = readers[f.suffix](f)
            except Exception as exc:
                print(f"  skipping {f.name}: {exc}")
                continue
            if cls is None or arr.ndim != 2 or arr.shape[1] != 4:
                print(f"  skipping {f.name}: class={cls}, shape={arr.shape}")
                continue
            by_class[cls].append(arr)

    # One permutation per class, cut once, so train and test never overlap.
    rng = np.random.default_rng(seed)
    for cls, arrays in by_class.items():
        order = rng.permutation(len(arrays))
        n_train = max(1, int(len(arrays) * train_ratio))
        for split, idx in (("pips_train", order[:n_train]), ("pips_test", order[n_train:])):
            out = output / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i, k in enumerate(idx):
                np.save(out / f"{i:04d}.npy", arrays[k])
            print(f"  {split}/{cls}: {len(idx)}")
    print(f"Done → {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tar", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=DATASET_ROOT / "POKERDVS")
    args = ap.parse_args()
    convert(args.tar, args.output)
