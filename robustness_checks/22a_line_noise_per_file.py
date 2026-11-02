"""Line noise per cleaned recording (continuous data, 19 scalp channels, Welch 4-s segments, 0.25 Hz bins).
Peak ratio = PSD at 60 Hz / median PSD in 55-58 and 62-65 Hz; reported as the median and max over channels.
Output: meeting_1013/22a_line_noise_per_file.csv"""
import os
import numpy as np
import pandas as pd
import mne
from common import *

mne.set_log_level('ERROR')
rows = []
for r in list_recordings():
    raw = read_raw(r['path'], preload=True).pick(SCALP_CHANS)
    sp = raw.compute_psd(method='welch', fmin=50, fmax=70, n_fft=1200, reject_by_annotation=True)
    psd, f = sp.get_data(return_freqs=True)
    side = ((f >= 55) & (f <= 58)) | ((f >= 62) & (f <= 65))
    ratio = psd[:, np.argmin(np.abs(f - 60))] / np.median(psd[:, side], axis=1)
    has_aha = '100' in mne.events_from_annotations(raw, verbose='ERROR')[1]
    rows.append(dict(fname=r['fname'], pid=r['pid'], cond=r['cond'], has_aha=has_aha,
                     ratio_median=float(np.median(ratio)), ratio_max=float(ratio.max()),
                     worst_channel=SCALP_CHANS[int(np.argmax(ratio))]))
    print(rows[-1], flush=True)
t = pd.DataFrame(rows).sort_values('ratio_median', ascending=False)
t.to_csv(os.path.join(OUT, 'meeting_1013', '22a_line_noise_per_file.csv'), index=False)
print(t.head(20).round(2).to_string(index=False))
