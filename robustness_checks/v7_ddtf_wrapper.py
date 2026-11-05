"""Thin wrapper: point Athena's BASE_PATH to EEG_DATA, then invoke 23_ddtf_stagewise with argv.
No notch by default. NOTCH=1 adds the v7 60 Hz notch on the 6 line-noise files (notch_patch.py), as a check.
EEG_BASELINES (optional) sets the rest-recording folder, otherwise BASE_PATH_ICA in ERP/erp_shared.py is used."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import ERP_DIR, WHOLE_SESSION_DIR  # noqa: E402
import os, sys  # noqa: E402
if os.environ.get('NOTCH') == '1':
    import notch_patch  # noqa: F401
OPTA = os.environ.get('OPTA_DIR')
assert OPTA, 'OPTA_DIR missing'
sys.path.insert(0, ERP_DIR)
sys.path.insert(0, WHOLE_SESSION_DIR)
import erp_shared
erp_shared.BASE_PATH = OPTA if OPTA.endswith('/') else OPTA + '/'
if os.environ.get('EEG_BASELINES'):
    erp_shared.BASE_PATH_ICA = os.environ['EEG_BASELINES'].rstrip('/') + '/'
    import wholesession_shared
    wholesession_shared.BASE_PATH_ICA = erp_shared.BASE_PATH_ICA
import wholesession_ddtf as W
W.BASE_PATH = erp_shared.BASE_PATH
# the read_eeg_file inside wholesession_ddtf uses mne.io.read_raw_eeglab (patched by notch_patch when NOTCH=1);
# BASE_PATH is only used to find files, which we point to EEG_DATA folder above.

# argv given by driver after '--'
import runpy
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
