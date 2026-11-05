"""Load EMG datasets and turn them into LibEMG features.

Every dataset comes out in the same shape, one row per 250 ms window:
  X        features (LibEMG Hudgins time-domain group: MAV, ZC, SSC, WL per channel)
  y        gesture label, 0..K-1
  subject  person id
  series   1 = calibration session (the new user's short guided recording)
           2 = test session (what every method is scored on)
  segment  id of the continuous recording the window came from; windows in the
           same segment are consecutive in time (majority voting needs this)

Datasets
  uci  UCI "EMG Data for Gestures" (Krilova et al., 2018), doi:10.24432/C5ZP5C.
       36 people, 8-channel Myo armband, 6 gestures, two series per person.
  3dc  3DC dataset (Cote-Allard et al., IEEE TNSRE 2019; Frontiers 2020).
       22 people, 10-channel 3DC armband, 11 gestures, 4 training + 4 test
       repetitions per gesture recorded after a 5-minute break.
Both are sampled at 1 kHz, so 250 samples = 250 ms and 50 samples = 50 ms.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from libemg.utils import get_windows
from libemg.feature_extractor import FeatureExtractor

WINDOW = 250
INCREMENT = 50

GESTURES = {
    "uci": ["rest", "fist", "wrist flexion", "wrist extension",
            "radial deviation", "ulnar deviation"],
    "3dc": ["rest", "radial deviation", "wrist flexion", "ulnar deviation",
            "wrist extension", "supination", "pronation", "power grip",
            "open hand", "chuck grip", "pinch grip"],
}


def _runs(labels):
    """Yield (start, end, label) for each run of identical labels."""
    change = np.flatnonzero(np.diff(labels)) + 1
    for s, e in zip(np.r_[0, change], np.r_[change, len(labels)]):
        yield s, e, labels[s]


def _uci_recordings(subject_dir):
    """UCI: one file per series; each labeled gesture hold is one segment.
    Label 0 (unmarked pause) and 7 (extended palm, not done by everyone) are dropped."""
    for f in sorted(Path(subject_dir).glob("*_raw_data_*.txt")):
        series = int(f.name.split("_")[0])
        df = pd.read_csv(f, sep="\t")
        emg = df[[f"channel{i}" for i in range(1, 9)]].to_numpy()
        lab = df["class"].to_numpy()
        for s, e, g in _runs(lab):
            if 1 <= g <= 6:
                yield emg[s:e], int(g) - 1, series


def _3dc_recordings(subject_dir):
    """3DC: file 3dc_EMG_gesture_<rep>_<class>.txt; train/ = series 1, test/ = series 2."""
    for part, series in (("train", 1), ("test", 2)):
        for f in sorted((Path(subject_dir) / part / "EMG").glob("3dc_EMG_gesture_*.txt")):
            rep, cls = map(int, re.findall(r"gesture_(\d+)_(\d+)", f.name)[0])
            emg = np.loadtxt(f, delimiter=",")
            yield emg, cls, series


def _subjects(dataset, root):
    root = Path(root)
    if dataset == "uci":
        dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        return sorted((int(p.name), p) for p in dirs)
    if dataset == "3dc":
        dirs = [p for p in root.iterdir() if p.name.startswith("Participant")]
        return sorted((int(p.name.replace("Participant", "")), p) for p in dirs)
    raise ValueError(dataset)


def build_features(dataset, root, cache):
    """Extract features for every window of a dataset (cached to `cache`)."""
    cache = Path(cache)
    if cache.exists():
        d = np.load(cache)
        return (d["X"], d["y"].astype(int), d["subject"].astype(int),
                d["series"].astype(int), d["segment"].astype(int))
    fe = FeatureExtractor()
    feats = fe.get_feature_groups()["HTD"]
    recs = _uci_recordings if dataset == "uci" else _3dc_recordings
    X, y, subj, ser, seg = [], [], [], [], []
    seg_id = 0
    for sid, d in _subjects(dataset, root):
        n0 = len(y)
        for emg, g, series in recs(d):
            if len(emg) < WINDOW:
                continue
            w = get_windows(emg, WINDOW, INCREMENT)
            X.append(fe.extract_features(feats, w, array=True))
            n = len(w)
            y += [g] * n; subj += [sid] * n; ser += [series] * n; seg += [seg_id] * n
            seg_id += 1
        print(f"{dataset} subject {sid}: {len(y) - n0} windows")
    out = dict(X=np.concatenate(X), y=np.array(y), subject=np.array(subj),
               series=np.array(ser), segment=np.array(seg))
    np.savez_compressed(cache, **out)
    return out["X"], out["y"], out["subject"], out["series"], out["segment"]
