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
  ciil Electrode-shift dataset (Campbell, Eddy, Bateman, Cote-Allard, Scheme,
       J. NeuroEng. Rehabil. 2024). 21 people, Myo armband at 200 Hz, 5 gestures.
       Series 1 = training session; series 2 = four trials after a simulated
       45-degree armband rotation.
  limbpos  Limb-position dataset (Fougner et al., IEEE TNSRE 2011). 12 people,
       8 EMG channels at 1 kHz, 8 gestures. Series 1 = arm position 1;
       series 2 = positions 2 to 5.
Windows are always 250 ms with a 50 ms step, whatever the sampling rate.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
from libemg.utils import get_windows
from libemg.feature_extractor import FeatureExtractor

WINDOW = 250          # samples at 1 kHz
INCREMENT = 50
RATE = {"uci": 1000, "3dc": 1000, "ciil": 200, "limbpos": 1000}

GESTURES = {
    "uci": ["rest", "fist", "wrist flexion", "wrist extension",
            "radial deviation", "ulnar deviation"],
    "3dc": ["rest", "radial deviation", "wrist flexion", "ulnar deviation",
            "wrist extension", "supination", "pronation", "power grip",
            "open hand", "chuck grip", "pinch grip"],
    "ciil": ["hand close", "hand open", "rest", "wrist extension", "wrist flexion"],
    "limbpos": ["wrist flexion", "wrist extension", "pronation", "supination",
                "hand open", "power grip", "pinch grip", "rest"],
}


# class index of "rest" in each dataset (predicting it holds the device still)
REST = {d: g.index("rest") for d, g in GESTURES.items()}


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


def _ciil_recordings(subject_dir):
    """Electrode shift: training/ = series 1; trial_1..trial_4 (after the
    shift) = series 2, in recording order. Files are R_<rep>_C_<class>.csv."""
    d = Path(subject_dir)
    for part, series in [("training", 1)] + [(f"trial_{i}", 2) for i in range(1, 5)]:
        files = sorted((d / part).glob("R_*_C_*.csv"),
                       key=lambda f: tuple(map(int, re.findall(r"R_(\d+)_C_(\d+)", f.name)[0])))
        for f in files:
            rep, cls = map(int, re.findall(r"R_(\d+)_C_(\d+)", f.name)[0])
            yield np.loadtxt(f, delimiter=","), cls, series


LIMBPOS_CLASSES = [1, 2, 3, 4, 5, 8, 9, 12]      # file class ids, in label order


def _limbpos_recordings(subject_dir):
    """Limb position: position 1 = series 1; positions 2-5 = series 2.
    Files are S<s>_C<class>_P<position>_R<rep>.txt; first 8 columns are EMG."""
    d = Path(subject_dir)
    files = []
    for f in d.glob("*.txt"):
        c, p, r = map(int, re.findall(r"_C(\d+)_P(\d+)_R(\d+)", f.name)[0])
        if c in LIMBPOS_CLASSES:
            files.append((p, r, LIMBPOS_CLASSES.index(c), f))
    for p, r, c, f in sorted(files):
        yield np.loadtxt(f, delimiter=",")[:, :8], c, 1 if p == 1 else 2


def _subjects(dataset, root):
    root = Path(root)
    if dataset == "uci":
        dirs = [p for p in root.iterdir() if p.is_dir() and p.name.isdigit()]
        return sorted((int(p.name), p) for p in dirs)
    if dataset == "3dc":
        dirs = [p for p in root.iterdir() if p.name.startswith("Participant")]
        return sorted((int(p.name.replace("Participant", "")), p) for p in dirs)
    if dataset == "ciil":
        dirs = [p for p in root.iterdir() if p.name.startswith("subject")]
        return sorted((int(p.name.replace("subject", "")), p) for p in dirs)
    if dataset == "limbpos":
        dirs = [p for p in root.iterdir() if re.fullmatch(r"S\d+", p.name)]
        return sorted((int(p.name[1:]), p) for p in dirs)
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
    recs = {"uci": _uci_recordings, "3dc": _3dc_recordings,
            "ciil": _ciil_recordings, "limbpos": _limbpos_recordings}[dataset]
    scale = RATE[dataset] / 1000
    win, inc = int(WINDOW * scale), int(INCREMENT * scale)
    X, y, subj, ser, seg = [], [], [], [], []
    seg_id = 0
    for sid, d in _subjects(dataset, root):
        n0 = len(y)
        for emg, g, series in recs(d):
            if len(emg) < win:
                continue
            w = get_windows(emg, win, inc)
            X.append(fe.extract_features(feats, w, array=True))
            n = len(w)
            y += [g] * n; subj += [sid] * n; ser += [series] * n; seg += [seg_id] * n
            seg_id += 1
        print(f"{dataset} subject {sid}: {len(y) - n0} windows")
    out = dict(X=np.concatenate(X), y=np.array(y), subject=np.array(subj),
               series=np.array(ser), segment=np.array(seg))
    np.savez_compressed(cache, **out)
    return out["X"], out["y"], out["subject"], out["series"], out["segment"]
