"""Paths for the robustness checks. Everything can be set with environment variables.

EEG_DATA      folder with the cleaned .set/.fdt files (default: BASE_PATH in ../ERP/erp_shared.py)
OUT_DIR       where results are written (default: robustness_checks/outputs)
ATHENA_ERP_DIR, ATHENA_WS_DIR   Athena's ERP and "Whole Session" code (default: the folders in this repo)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ERP_DIR = os.environ.get('ATHENA_ERP_DIR', os.path.join(REPO, 'ERP'))
WHOLE_SESSION_DIR = os.environ.get('ATHENA_WS_DIR', os.path.join(REPO, 'Whole Session'))
AHA_JSON = os.path.join(REPO, 'aha_events.json')
OUT = os.environ.get('OUT_DIR', os.path.join(HERE, 'outputs'))
RESULTS = os.path.join(OUT, 'meeting_1013')
REFERENCE = os.path.join(HERE, 'reference')


def _default_data():
    sys.path.insert(0, ERP_DIR)
    try:
        import erp_shared
        return erp_shared.BASE_PATH
    except Exception:  # noqa: BLE001
        return os.path.join(REPO, 'Data')


DATA = os.environ.get('EEG_DATA') or _default_data()
os.makedirs(RESULTS, exist_ok=True)

# Inputs used by the v7 scripts
CODING_CSV = os.environ.get('CODING_CSV', os.path.join(REPO, 'Data', 'creativitycodings.csv'))   # Wallas stage coding


def _default_baselines():
    try:
        import erp_shared
        return erp_shared.BASE_PATH_ICA
    except Exception:  # noqa: BLE001
        return os.path.join(REPO, 'Data', 'baselines')


BASELINES = os.environ.get('EEG_BASELINES') or _default_baselines()
