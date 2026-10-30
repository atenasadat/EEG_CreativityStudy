"""Import this first to apply a 60 Hz notch, when they are read, to the cleaned files that still had line noise
(22a_line_noise_per_file.csv, 60 Hz peak >= 2x the nearby floor: all 4 P20 files, P11_control_A2 and
P13_control_B3; P6_treatment_B3 is 1.996, below). Every pipeline that reads EEG with mne.io.read_raw_eeglab gets the notched data;
nothing else changes. Boundary durations are set to 0 first so the filter does not skip real data after cuts.
Usage: python -c "import notch_patch, runpy, sys; sys.argv = [...]; runpy.run_path('<script>', run_name='__main__')"
"""
import os
import mne
import pandas as pd
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS  # noqa: E402

_noise = pd.read_csv(os.path.join(RESULTS, '22a_line_noise_per_file.csv'))
NOTCH_FILES = set(_noise[_noise.ratio_median >= 2.0].fname)
_orig = mne.io.read_raw_eeglab
COUNT = {'notched': 0}


def _read(input_fname, *args, **kwargs):
    raw = _orig(input_fname, *args, **kwargs)
    if os.path.basename(str(input_fname)) in NOTCH_FILES:
        raw.load_data()
        a = raw.annotations
        dur = [0.0 if str(d) == 'boundary' else float(x) for d, x in zip(a.description, a.duration)]
        raw.set_annotations(mne.Annotations(a.onset, dur, a.description, orig_time=a.orig_time))
        picks = mne.pick_types(raw.info, eeg=True, misc=True, exclude=[])
        raw.notch_filter(60.0, picks=picks, skip_by_annotation=('edge', 'boundary'), verbose='ERROR')
        raw.set_annotations(a)          # restore the original annotations for epoching and windows
        COUNT['notched'] += 1
    return raw


mne.io.read_raw_eeglab = _read
print('notch_patch active for', sorted(NOTCH_FILES), flush=True)
