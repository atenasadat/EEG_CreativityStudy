"""Which electrodes carry the PCA region signals (paper v08, PCA check of Table 2).

Same fit as v8_her_ddtf.py with ROI=pca: per participant, all task recordings of both conditions, 1-45 Hz band-passed
copy, first principal component of each region's electrodes, unit-norm weights with a positive sum. Reports, per
participant, the weights, the variance explained, the correlation of the component with the channel average, and for
the frontal region the share of the (squared) weight on Fp1 + Fp2 and on the single largest electrode.
The paper uses: Fp1 + Fp2 carry more than half of the frontal weight in 8 of 19 participants, and a single F7 or F8
electrode in 5 more.
Input: EEG_DATA (cleaned .set files). Output: OUT/pca_frontal_weights.json
"""
import glob
import json
import os
import re

import mne
import numpy as np

mne.set_log_level('ERROR')
D, OUT = os.environ['EEG_DATA'], os.environ.get('OUT', '.')
ROIS = {'Frontal': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8'], 'Parietal': ['P7', 'P3', 'Pz', 'P4', 'P8'],
        'Occipital': ['O1', 'O2']}
allch = sum(ROIS.values(), [])
files = sorted(glob.glob(os.path.join(D, 'P*_*.set')))
pids = sorted({re.match(r'(P\d+)_', os.path.basename(f)).group(1) for f in files}, key=lambda p: int(p[1:]))
rows = []
for pid in pids:
    data = np.concatenate([mne.io.read_raw_eeglab(f, preload=True).pick(allch).filter(1.0, 45.0).get_data()
                           for f in files if os.path.basename(f).startswith(pid + '_')], axis=1)
    out = {'pid': pid}
    for roi, ch in ROIS.items():
        x = data[[allch.index(c) for c in ch]]
        x = x - x.mean(1, keepdims=True)
        ev, V = np.linalg.eigh(np.cov(x))
        w = V[:, -1]; w = w if w.sum() >= 0 else -w; w = w / np.linalg.norm(w)
        r = dict(weights=dict(zip(ch, [round(float(v), 3) for v in w])), var_explained=float(100 * ev[-1] / ev.sum()),
                 r_with_average=float(np.corrcoef(w @ x, x.mean(0))[0, 1]))
        if roi == 'Frontal':
            top = int(np.argmax(w ** 2))
            r.update(fp_share=float(w[0] ** 2 + w[1] ** 2), top_channel=ch[top], top_share=float(w[top] ** 2))
        out[roi] = r
    rows.append(out)
    print(pid, {k: round(v, 2) for k, v in out['Frontal'].items() if k in ('fp_share', 'top_share')},
          out['Frontal']['top_channel'], flush=True)

fp = [r for r in rows if r['Frontal']['fp_share'] > .5]
lat = [r for r in rows if r['Frontal']['fp_share'] <= .5 and r['Frontal']['top_channel'] in ('F7', 'F8')
       and r['Frontal']['top_share'] > .5]
summ = dict(N=len(rows), fp_more_than_half=[r['pid'] for r in fp],
            single_F7_or_F8_more_than_half=[r['pid'] for r in lat],
            median_fp_share=float(np.median([r['Frontal']['fp_share'] for r in rows])),
            **{f'median_var_{k}': float(np.median([r[k]['var_explained'] for r in rows])) for k in ROIS},
            **{f'median_r_{k}': float(np.median([r[k]['r_with_average'] for r in rows])) for k in ROIS})
print(summ)
json.dump(dict(summary=summ, participants=rows), open(f'{OUT}/pca_frontal_weights.json', 'w'), indent=1)
