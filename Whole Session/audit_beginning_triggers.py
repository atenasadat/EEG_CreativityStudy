import os
import sys
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ERP_DIR = os.path.join(PROJECT_ROOT, "ERP")
if ERP_DIR not in sys.path:
    sys.path.insert(0, ERP_DIR)

from erp_shared import BASE_PATH, get_condition_file_groups, read_eeg_file
from wholesession_shared import early_trigger_runs


def run():
    matched_pids, control_groups, treatment_groups = get_condition_file_groups(BASE_PATH)
    print(f"Matched subjects: N={len(matched_pids)}")
    print("Pid,Condition,File,BeginningTrigger,NextTrigger")
    for pid in matched_pids:
        for condition, groups in (("Control", control_groups), ("Treatment", treatment_groups)):
            for fname in groups[pid]:
                raw = read_eeg_file(fname, base_path=BASE_PATH, preload=False, verbose=False)
                runs = early_trigger_runs(raw)
                first = f"{runs[0][0]}@{runs[0][1]:.3f}s" if runs else "None"
                second = f"{runs[1][0]}@{runs[1][1]:.3f}s" if len(runs) > 1 else "None"
                print(f"{pid},{condition},{fname},{first},{second}")


if __name__ == "__main__":
    run()
