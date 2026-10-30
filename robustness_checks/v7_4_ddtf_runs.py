"""Option A analyses 4 and 5: dDTF with session quarters (4) and Wallas stages (5), both averaging
and PCA ROI signals. Drives 23_ddtf_stagewise.py four times with the Option A data folder.

Analysis 4 => --segments quarter, --roi mean | pca
Analysis 5 => --segments wallas,  --roi mean | pca, --coding CODING_CSV (config.py)
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import RESULTS, DATA, CODING_CSV, ERP_DIR, WHOLE_SESSION_DIR  # noqa: E402
import os, sys  # noqa: E402
import json
import os
import runpy
import shutil
import subprocess
import sys

OPTA = DATA
CODING = CODING_CSV
OUT = os.path.join(RESULTS, 'optB_ddtf')
os.makedirs(OUT, exist_ok=True)

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# v7_ddtf_wrapper.py points Athena's BASE_PATH at EEG_DATA and applies the notch, then runs 23_ddtf_stagewise.py
WRAPPER = os.path.join(HERE, 'v7_ddtf_wrapper.py')


def run(roi, segments, tag):
    out_dir = os.path.join(OUT, tag)
    os.makedirs(out_dir, exist_ok=True)
    args = [PY, WRAPPER, os.path.join(HERE, '23_ddtf_stagewise.py'),
            '--roi', roi, '--segments', segments, '--out', out_dir]
    if segments == 'wallas':
        args += ['--coding', CODING]
    env = os.environ.copy()
    env['OPTA_DIR'] = OPTA
    env['EEG_DATA'] = OPTA
    env['PYTHONPATH'] = HERE + ':' + env.get('PYTHONPATH', '')
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    print('\n==>', ' '.join(args), flush=True)
    r = subprocess.run(args, env=env, cwd=HERE)
    return dict(tag=tag, roi=roi, segments=segments, out=out_dir, returncode=r.returncode)


results = []
for tag, roi, segments in [
    ('quarter_mean', 'mean', 'quarter'),
    ('quarter_pca', 'pca', 'quarter'),
    ('wallas_mean', 'mean', 'wallas'),
    ('wallas_pca', 'pca', 'wallas'),
]:
    results.append(run(roi, segments, tag))

json.dump(results, open(os.path.join(OUT, '_runs.json'), 'w'), indent=1)

# summarize stats.csv files
import pandas as pd
summary = {}
for r in results:
    sp = os.path.join(r['out'], 'stats.csv')
    if os.path.exists(sp):
        df = pd.read_csv(sp)
        summary[r['tag']] = df.round(5).to_dict(orient='records')

# Save the 4 and 5 analysis JSON separately for the report
ana4 = {k: v for k, v in summary.items() if k.startswith('quarter_')}
ana5 = {k: v for k, v in summary.items() if k.startswith('wallas_')}
with open(os.path.join(RESULTS, 'optB_4_ddtf_quarters.json'), 'w') as fh:
    json.dump(dict(runs=[r for r in results if r['tag'].startswith('quarter_')],
                   stats=ana4), fh, indent=1)
with open(os.path.join(RESULTS, 'optB_5_ddtf_wallas.json'), 'w') as fh:
    json.dump(dict(runs=[r for r in results if r['tag'].startswith('wallas_')],
                   stats=ana5), fh, indent=1)
print('wrote optB_4_ddtf_quarters.json and optB_5_ddtf_wallas.json')
