"""Load the UCI "EMG Data for Gestures" dataset and turn it into LibEMG features.

Dataset: Krilova, Kastalskiy, Kazantsev, Makarov & Lobov (2018), UCI Machine
Learning Repository, https://doi.org/10.24432/C5ZP5C. 36 people, 8-channel Myo
armband, two recording series per person.

Windowing and features both come from LibEMG (Eddy et al., IEEE Access 2023):
  - libemg.utils.get_windows
  - libemg.feature_extractor.FeatureExtractor, Hudgins time-domain group (HTD):
    mean absolute value, zero crossings, slope sign changes, waveform length.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from libemg.utils import get_windows
from libemg.feature_extractor import FeatureExtractor

# UCI labels 1-6. Label 0 = unmarked pause, label 7 = extended palm (not every
# subject performed it), so both are dropped.
GESTURES = {1: "rest", 2: "fist", 3: "wrist flexion", 4: "wrist extension",
            5: "radial deviation", 6: "ulnar deviation"}
WINDOW = 250      # samples, ~250 ms at ~1 kHz
INCREMENT = 50    # samples, ~50 ms step (a new decision every 50 ms)


def _segments(labels):
    """Yield (start, end, label) for each run of identical labels."""
    change = np.flatnonzero(np.diff(labels)) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(labels)]
    for s, e in zip(starts, ends):
        yield s, e, labels[s]


def load_subject(subject_dir):
    """Return windows, labels and series index for one person.

    Windows never cross a label boundary, so every window belongs to exactly
    one gesture.
    """
    files = sorted(Path(subject_dir).glob("*_raw_data_*.txt"))
    win_list, lab_list, ser_list = [], [], []
    for f in files:
        series = int(f.name.split("_")[0])          # 1 or 2
        df = pd.read_csv(f, sep="\t")
        emg = df[[f"channel{i}" for i in range(1, 9)]].to_numpy()
        labels = df["class"].to_numpy()
        for s, e, lab in _segments(labels):
            if lab not in GESTURES or e - s < WINDOW:
                continue
            w = get_windows(emg[s:e], WINDOW, INCREMENT)
            win_list.append(w)
            lab_list.append(np.full(len(w), lab - 1))   # 0..5 for LibEMG
            ser_list.append(np.full(len(w), series))
    return (np.concatenate(win_list), np.concatenate(lab_list),
            np.concatenate(ser_list))


def build_features(data_root, cache="features.npz"):
    """Extract HTD features for all 36 people, cached to disk."""
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"].astype(int), d["subject"], d["series"].astype(int)
    fe = FeatureExtractor()
    feats = fe.get_feature_groups()["HTD"]
    X, y, subj, ser = [], [], [], []
    subject_dirs = sorted(p for p in Path(data_root).iterdir()
                          if p.is_dir() and p.name.isdigit())
    for d in subject_dirs:
        w, lab, s = load_subject(d)
        X.append(fe.extract_features(feats, w, array=True))
        y.append(lab)
        subj.append(np.full(len(lab), int(d.name)))
        ser.append(s)
        print(f"subject {d.name}: {len(lab)} windows")
    X, y = np.concatenate(X), np.concatenate(y).astype(int)
    subj, ser = np.concatenate(subj), np.concatenate(ser)
    np.savez_compressed(cache, X=X, y=y, subject=subj, series=ser)
    return X, y, subj, ser
