"""Table 2 of the paper (v08): Athena's own connectivity cells in wallas_analysis/wallas_mapping.ipynb
(cells 2-17 for the sessions, coding and alignment, cell 30 for dDTF and the mixed-model tests, cell 31 for the
stage-wise Wilcoxon tests, cell 32 for the brain figure), run without changing the notebook. The only changes:

  - file paths: EEG_DATA (cleaned .set files), CODING_CSV (creativitycodings.csv), OUT (output folder)
  - each 4-s window is resampled to FS = 100 Hz (the notebook used 128). No notch and no other filter.
  - a beta band (13-30 Hz) next to theta (4-8) and alpha (8-13), so 18 pathways in one FDR family
  - model order ORDER = 10. AIC and BIC are still computed and printed (they pick 5 and 2 at 100 Hz).
    Three bands need more poles than two, and the residuals are closer to white at order 10.
    ORDER=aic uses the AIC order (the check in the paper).
  - ROI=pca: first principal component of each region instead of the channel mean. Weights are fit once per
    participant on a 1-45 Hz band-passed copy of all task data of both conditions, then applied to the
    unfiltered data (the band-passed copy is used only for the fit).
  - NOTCH=1 (check only) puts the v7 60 Hz notch on the six noisy recordings.
  - CSD=1 (check only) replaces the average reference with a surface Laplacian (csd_patch.py), which is
    reference-free (Kaminski et al. 2001 warn against the average reference for DTF).
  - FIG=1 draws cell 32's figures with beta rows added, into OUT/figures (nothing is written to the repo).

Usage:
  EEG_DATA=... CODING_CSV=... OUT=... [FS=100] [ORDER=10|aic] [ROI=mean|pca] [FIG=0|1] python v8_her_ddtf.py
"""
import json, os, re, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = Path(os.environ['OUT']); OUT.mkdir(parents=True, exist_ok=True)
EEG_DATA, CODING_CSV = os.environ['EEG_DATA'], os.environ['CODING_CSV']
FS, ORDER = int(os.environ.get('FS', '100')), os.environ.get('ORDER', '10')
ROI, FIG = os.environ.get('ROI', 'mean'), os.environ.get('FIG', '0') == '1'
if os.environ.get('NOTCH') == '1':  # check only: 60 Hz notch on the six noisy recordings, as in v7
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import notch_patch  # noqa: F401
if os.environ.get('CSD') == '1':  # check only: surface Laplacian (reference-free) at read time
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import csd_patch  # noqa: F401

nb = json.load(open(REPO / 'wallas_analysis' / 'wallas_mapping.ipynb'))
src = lambda i: ''.join(nb['cells'][i]['source'])


def patched(i, pairs):
    s = src(i)
    for a, b in pairs:
        assert a in s, f'cell {i}: {a!r} not found'
        s = s.replace(a, b)
    return s


ORDER_LINE = ("DDTF_ORDER = int(q1_ddtf_order_selection.loc[q1_ddtf_order_selection.mean_AIC.idxmin(), 'order'])")
PATCHES = {
    2: [("Path('/Users/athenasaghi/VSProjects/EEG_CreativityStudy')", f"Path({str(REPO)!r})"),
        ("Path('/Users/athenasaghi/Desktop/CleanDATA/clean')", f"Path({EEG_DATA!r})"),
        ("PROJECT / 'Data' / 'creativitycodings.csv'", f"Path({CODING_CSV!r})"),
        ("PROJECT / 'wallas_analysis'", f"Path({str(OUT)!r})")],
    30: [("DDTF_WINDOW_SEC, DDTF_RESAMPLE_SFREQ = 4.0, 128.0", f"DDTF_WINDOW_SEC, DDTF_RESAMPLE_SFREQ = 4.0, {FS}.0"),
         ("DDTF_BANDS = {'Theta': (4.0, 8.0), 'Alpha': (8.0, 13.0)}",
          "DDTF_BANDS = {'Theta': (4.0, 8.0), 'Alpha': (8.0, 13.0), 'Beta': (13.0, 30.0)}"),
         ("spec.loader.exec_module(whole_session_ddtf)",
          "spec.loader.exec_module(whole_session_ddtf); ROI_HOOK(whole_session_ddtf)"),
         (ORDER_LINE, ORDER_LINE.replace('DDTF_ORDER =', 'aic_order =') + "\nDDTF_ORDER = ORDER_HOOK(aic_order)"),
         ("print(f'MVAR order: AIC -> {DDTF_ORDER} (used), BIC -> {bic_order} ({len(sample)} sampled windows)')",
          "print(f'MVAR order: AIC -> {aic_order}, BIC -> {bic_order}, used {DDTF_ORDER} ({len(sample)} sampled windows)')")],
}


def display(x):
    print(x.to_string() if hasattr(x, 'to_string') else x, flush=True)


def order_hook(aic_order):
    return int(aic_order) if ORDER == 'aic' else int(ORDER)


PCA = {}


def roi_hook(W):
    """ROI=pca: swap the channel-mean region signal for the first principal component."""
    if ROI == 'mean':
        return
    import mne
    eeg, chans = ns['eeg'], ns['CHANNELS']
    for pid in sorted({k[0] for k in eeg}, key=lambda p: int(p[1:])):
        datas = []
        for key in [k for k in eeg if k[0] == pid]:
            raw = mne.io.read_raw_eeglab(str(Path(EEG_DATA) / eeg[key]['file']), preload=True, verbose=False).pick(chans)
            datas.append((raw.ch_names, raw.copy().filter(1.0, 45.0, verbose='ERROR').get_data()))
        names = datas[0][0]
        data = np.concatenate([d for n, d in datas if n == names], axis=1)
        PCA[pid] = {}
        for roi, rchans in W.ROIS.items():
            idx = [names.index(c) for c in rchans if c in names]
            x = data[idx] - data[idx].mean(axis=1, keepdims=True)
            evals, evecs = np.linalg.eigh(np.cov(x))
            w = evecs[:, -1]
            w = w if w.sum() >= 0 else -w
            PCA[pid][roi] = ([names[i] for i in idx], w / np.linalg.norm(w), float(100 * evals[-1] / evals.sum()))
        print('PCA weights', pid, {r: round(v[2], 1) for r, v in PCA[pid].items()}, flush=True)

    def roi_series_pca(raw, ch_names):
        pid = re.match(r'^(P\d+)_', Path(raw.filenames[0]).name).group(1)
        data = raw.get_data()
        rows, out = [], []
        for roi, (rch, w, _) in PCA[pid].items():
            rows.append(w @ data[[raw.ch_names.index(c) for c in rch]])
            out.append(roi)
        return np.asarray(rows), out

    W.roi_series = roi_series_pca


ns = {'display': display, '__name__': '__notebook__', 'ORDER_HOOK': order_hook, 'ROI_HOOK': roi_hook}
for i in [2, 4, 6, 8, 10, 12, 14, 15, 17]:
    print(f'--- cell {i}', flush=True)
    exec(compile(patched(i, PATCHES.get(i, [])), f'cell{i}', 'exec'), ns)
code_cells = [i for i, c in enumerate(nb['cells']) if c['cell_type'] == 'code']
for name in ['STEPS', 'CONDITIONS', 'CHANNELS']:
    if name not in ns:
        for i in code_cells:
            if 17 < i < 30 and re.search(rf'^{name}\s*[,=]|^[A-Z_, ]*\b{name}\b[A-Z_, ]*=', src(i), re.M):
                print(f'--- cell {i} (for {name})', flush=True)
                exec(compile(src(i), f'cell{i}', 'exec'), ns)
                break
for i in (30, 31):
    print(f'--- cell {i}', flush=True)
    exec(compile(patched(i, PATCHES.get(i, [])), f'cell{i}', 'exec'), ns)

t, u, cov = ns['q1_ddtf_tests'], ns['q1_ddtf_unit'], ns['q1_ddtf_coverage']
t.to_csv(OUT / 'tests.csv', index=False)
u.to_csv(OUT / 'unit.csv', index=False)
cov.to_csv(OUT / 'coverage.csv', index=False)
ns['q1_ddtf_order_selection'].to_csv(OUT / 'order_selection.csv', index=False)
ns['q1_ddtf_posthoc'].to_csv(OUT / 'posthoc.csv', index=False)
summary = dict(fs=FS, order_used=int(ns['DDTF_ORDER']), order_aic=int(ns['aic_order']), order_bic=int(ns['bic_order']),
               roi=ROI, n_subjects=int(t.n_subjects.max()), n_rows=int(t.n_rows.max()),
               windows_used=int(cov.n_windows.sum()), windows_unstable=int(cov.n_unstable_excluded.sum()),
               n_tests=len(t), n_fdr_below_05=int((t.p_fdr_all < .05).sum()))
json.dump(summary, open(OUT / 'summary.json', 'w'), indent=1)
print(summary, flush=True)

if FIG:
    # cell 32 with a beta row; the notebook's figure code is read and the band list extended, nothing else
    fig_src = (REPO / 'Whole Session' / 'kosmyna_style_figures.py').read_text()
    two = 'bands = {"Theta": "4–8 Hz", "Alpha": "8–13 Hz"}'
    assert two in fig_src
    fig_src = fig_src.replace(two, 'bands = {"Theta": "4–8 Hz", "Alpha": "8–13 Hz", "Beta": "13–30 Hz"}')
    import types
    kfig = types.ModuleType('kosmyna_style_figures')
    kfig.__file__ = str(REPO / 'Whole Session' / 'kosmyna_style_figures.py')
    exec(compile(fig_src, kfig.__file__, 'exec'), kfig.__dict__)
    kfig.WALLAS_FIG_DIR = str(OUT / 'figures')
    for surface in os.environ.get('FIG_SURFACES', 'brain,head').split(','):
        kfig.plot_ddtf_networks(t, 'q1', surface)
print('DONE', flush=True)
