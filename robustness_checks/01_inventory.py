"""Task 1+2 inventory: per-recording event-100 counts, durations, boundary info.

Output: outputs/01_recording_inventory.csv
"""
import numpy as np
import pandas as pd
from common import *

rows = []
for r in list_recordings():
    raw = read_raw(r['path'], preload=False)
    sf = raw.info['sfreq']
    dur_clean = raw.n_times / sf
    ann = annotations_by_code(raw)
    t100 = np.array([x[0] for x in ann.get('100', [])])
    b_on, b_du = boundary_times(raw)
    # duplicates / near duplicates of 100
    t100s = np.sort(t100)
    n_dup_exact = int(len(t100s) - len(np.unique(np.round(t100s * sf))))
    n_close_05 = int(np.sum(np.diff(t100s) < 0.5)) if len(t100s) > 1 else 0
    # epoch feasibility for -0.5..+0.5 (ERP) and -5..+5 (TFR) inside the data
    ok_erp = (t100 - 0.5 >= 0) & (t100 + 0.5 <= (raw.n_times - 1) / sf)
    ok_tfr = (t100 - 5 >= 0) & (t100 + 5 <= (raw.n_times - 1) / sf)
    # boundary inside the epoch window (boundary sits on the cleaned clock at its onset)
    def spans(t, half):
        return np.array([np.any((b_on > tt - half) & (b_on < tt + half)) for tt in t], bool)
    sp_erp = spans(t100, 0.5) if len(t100) else np.zeros(0, bool)
    sp_tfr = spans(t100, 5.0) if len(t100) else np.zeros(0, bool)
    t3 = collapse_bursts([x[0] for x in ann.get('3', [])])
    t4 = collapse_bursts([x[0] for x in ann.get('4', [])])
    task_corr = TASK_CORRECTION.get((r['pid'], r['cond'], r['task']), r['task'])
    rows.append(dict(pid=r['pid'], cond=r['cond'], task=r['task'], task_corrected=task_corr,
                     fname=r['fname'], sfreq=sf, n_chan=len(raw.ch_names),
                     dur_clean_s=dur_clean, n_boundary=len(b_on), removed_s=float(b_du.sum()),
                     dur_orig_s=dur_clean + float(b_du.sum()),
                     n100=len(t100), n100_dup_exact=n_dup_exact, n100_gap_lt_0p5=n_close_05,
                     n100_erp_in_data=int(ok_erp.sum()), n100_tfr_in_data=int(ok_tfr.sum()),
                     n100_erp_span_boundary=int(sp_erp.sum()), n100_tfr_span_boundary=int(sp_tfr.sum()),
                     n3_raw=len(ann.get('3', [])), n3_collapsed=len(t3),
                     n4_raw=len(ann.get('4', [])), n4_collapsed=len(t4),
                     t100=';'.join(f'{x:.3f}' for x in np.sort(t100))))
df = pd.DataFrame(rows)
df['pnum'] = df.pid.str[1:].astype(int)
df = df.sort_values(['pnum', 'cond', 'task']).drop(columns='pnum')
df.to_csv(f'{OUT}/01_recording_inventory.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(df.drop(columns=['t100', 'fname']).to_string())
print('total 100 events:', df.n100.sum(), ' control:', df[df.cond == 'control'].n100.sum(),
      ' treatment:', df[df.cond == 'treatment'].n100.sum())
print('exact dup 100:', df.n100_dup_exact.sum(), ' gaps<0.5s:', df.n100_gap_lt_0p5.sum())
print('ERP epochs in data:', df.n100_erp_in_data.sum(), ' spanning boundary:', df.n100_erp_span_boundary.sum())
print('TFR epochs in data:', df.n100_tfr_in_data.sum(), ' spanning boundary:', df.n100_tfr_span_boundary.sum())
print('channels:', df.n_chan.value_counts().to_dict(), ' sfreq:', df.sfreq.unique())
