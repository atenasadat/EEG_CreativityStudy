"""Thin wrapper: monkey-patch Athena's BASE_PATH to EEG_DATA, then invoke 23_ddtf_stagewise with argv."""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import ERP_DIR, WHOLE_SESSION_DIR  # noqa: E402
import os, sys  # noqa: E402
import notch_patch  # noqa: F401
OPTA = os.environ.get('OPTA_DIR')
assert OPTA, 'OPTA_DIR missing'
sys.path.insert(0, ERP_DIR)
sys.path.insert(0, WHOLE_SESSION_DIR)
import erp_shared
erp_shared.BASE_PATH = OPTA if OPTA.endswith('/') else OPTA + '/'
import wholesession_ddtf as W
W.BASE_PATH = erp_shared.BASE_PATH
# the read_eeg_file inside wholesession_ddtf uses mne.io.read_raw_eeglab (patched by notch_patch);
# BASE_PATH is only used to find files, which we point to EEG_DATA folder above.

# argv given by driver after '--'
import runpy
script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name='__main__')
