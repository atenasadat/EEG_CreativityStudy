"""How much of the pre-aha ERP cluster each channel carries (paper v08, ERP Results).

Same test as v8_erp_spatial.py (spatial Delaunay neighbours, -0.5 to 0 s, threshold t(.975, 14)). The observed
clusters do not depend on the permutations, so a short permutation run is enough to get the cluster mask. For the
largest cluster, each channel's share = its summed |t| inside the cluster / the cluster's summed |t|.
The paper uses: F3 and C3 carry 84% of the cluster.
Input: OUT_DIR/03_participant_erps.npz. Output: OUT_DIR/v8_erp_cluster_share.json
"""
import json

import mne
import numpy as np
import scipy.stats as st
from mne.stats import permutation_cluster_1samp_test

from common import *

mne.set_log_level('ERROR')
z = np.load(f'{OUT}/03_participant_erps.npz')
times, chans = z['times'], list(z['chans'])
sc = [chans.index(c) for c in SCALP_CHANS]
scale = 1e6 if np.abs(z['control']).max() < 1e-2 else 1.0
X = np.transpose(z['control'][:, sc] - z['treatment'][:, sc], (0, 2, 1)) * scale
pre = times <= 1e-9
A = mne.channels.find_ch_adjacency(read_raw(list_recordings()[0]['path'], preload=False).pick(SCALP_CHANS).info,
                                   'eeg')[0]
tobs, cl, _, _ = permutation_cluster_1samp_test(X[:, pre], n_permutations=100, threshold=st.t.ppf(.975, len(X) - 1),
                                                tail=0, adjacency=A, out_type='mask', seed=0, verbose='ERROR')
m = cl[int(np.argmax([np.abs(tobs[c]).sum() for c in cl]))]
mass = (np.abs(tobs) * m).sum(0)
dt = float(times[1] - times[0])
share = {SCALP_CHANS[i]: dict(share=float(mass[i] / mass.sum()), ms_in_cluster=float(m[:, i].sum() * dt * 1000))
         for i in np.argsort(-mass) if mass[i] > 0}
out = dict(channels=list(share), share=share, f3_c3_share=share['F3']['share'] + share['C3']['share'],
           n_points=int(m.sum()), grid=list(m.shape))
json.dump(out, open(f'{OUT}/v8_erp_cluster_share.json', 'w'), indent=1)
print({k: round(v['share'], 3) for k, v in share.items()}, 'F3+C3', round(out['f3_c3_share'], 3))
