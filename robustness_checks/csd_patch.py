"""Surface Laplacian (current source density, MNE spherical spline, default settings) on the 19 scalp channels at read time.
Reference-free, so it removes the common average term. A1/A2/Event are left as they are (not used by the ROIs)."""
import mne, numpy as np
_orig = mne.io.read_raw_eeglab
SCALP = ['P3', 'C3', 'F3', 'Fz', 'F4', 'C4', 'P4', 'Cz', 'Fp1', 'Fp2', 'T7', 'P7', 'O1', 'O2', 'F7', 'F8', 'P8', 'T8', 'Pz']
def _read(input_fname, *a, **k):
    k['preload'] = True
    raw = _orig(input_fname, *a, **k)
    sc = raw.copy().pick(SCALP)
    sc.set_eeg_reference('average', projection=False, verbose='ERROR')
    pos = np.array([ch['loc'][:3] for ch in sc.info['chs']])
    sphere = (0.0, 0.0, 0.0, float(np.linalg.norm(pos, axis=1).mean()))
    csd = mne.preprocessing.compute_current_source_density(sc, sphere=sphere, verbose='ERROR')
    idx = [raw.ch_names.index(c) for c in SCALP]
    raw._data[idx] = csd.get_data()
    return raw
mne.io.read_raw_eeglab = _read
print('csd_patch active', flush=True)
