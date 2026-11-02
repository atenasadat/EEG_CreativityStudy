"""Shared helpers for v04 analyst-A scripts.

Paths come from config.py. Nothing here writes data.
"""
import os
import re
import glob
import numpy as np
import mne

from config import DATA, OUT
FIGDIR = os.path.join(OUT, 'figures')
os.makedirs(OUT, exist_ok=True)
os.makedirs(FIGDIR, exist_ok=True)

# channel order used by the repo (ERP/utils.py standard_chans)
STANDARD_CHANS = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'A1', 'Fp1',
                  'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'A2', 'P8', 'T8', 'Pz']
SCALP_CHANS = [c for c in STANDARD_CHANS if c not in ('A1', 'A2')]

PARTICIPANTS = ['P3', 'P4', 'P5', 'P6', 'P7', 'P9', 'P10', 'P11', 'P12', 'P13', 'P14',
                'P15', 'P16', 'P17', 'P18', 'P20', 'P21', 'P22', 'P23']
COND_LABEL = {'control': 'Unassisted', 'treatment': 'LLM-assisted'}
COL_UNASSISTED = '#1f6fd1'
COL_LLM = '#d62728'

# README section 11 label corrections (EEG file name -> coding-sheet task)
TASK_CORRECTION = {('P12', 'control', 'B1'): 'B2',
                   ('P23', 'control', 'B3'): 'B2',
                   ('P17', 'control', 'A2'): 'B2'}

FNAME_RE = re.compile(r'^(P\d+)_(control|treatment)_([AB]\d)_postcleaning(\d)\.set$')


def list_recordings():
    """Return list of dicts (pid, cond, task, pass, fname, path) for the 76 cleaned task files."""
    recs = []
    for p in sorted(glob.glob(os.path.join(DATA, '*.set'))):
        fn = os.path.basename(p)
        m = FNAME_RE.match(fn)
        if not m:
            raise ValueError(f'unexpected file name {fn}')
        pid, cond, task, npass = m.groups()
        recs.append(dict(pid=pid, cond=cond, task=task, cpass=int(npass), fname=fn, path=p))
    return recs


def read_raw(path, preload=True):
    raw = mne.io.read_raw_eeglab(path, preload=preload, verbose='ERROR')
    if 'Event' in raw.ch_names:
        raw.drop_channels(['Event'])
    return raw


def annotations_by_code(raw):
    a = raw.annotations
    out = {}
    for o, d, de in zip(a.onset, a.duration, a.description):
        out.setdefault(str(de), []).append((float(o), float(d)))
    return out


def collapse_bursts(times, gap=0.5):
    """Collapse events of the same code less than `gap` s apart (README 8.1)."""
    times = np.sort(np.asarray(times, float))
    if times.size == 0:
        return times
    keep = [times[0]]
    last = times[0]
    for t in times[1:]:
        if t - last >= gap:
            keep.append(t)
        last = t  # chain: a burst is a run of events each < gap from the previous
    return np.asarray(keep)


def boundary_times(raw):
    """Cleaned-clock onsets (s) of boundary events and the removed durations (s)."""
    ab = annotations_by_code(raw).get('boundary', [])
    if not ab:
        return np.zeros(0), np.zeros(0)
    on = np.array([x[0] for x in ab])
    du = np.array([x[1] for x in ab])
    return on, du


def sign_flip_matrix(n):
    """All 2^n sign patterns (rows), first row all +1."""
    idx = np.arange(2 ** n)
    bits = ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
    return 1.0 - 2.0 * bits


def bh_fdr(p):
    from statsmodels.stats.multitest import multipletests
    return multipletests(p, method='fdr_bh')[1]
