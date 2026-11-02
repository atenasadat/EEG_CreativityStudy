"""C-d rerun (controller, reviewer R1): insight vs non-insight ERP within condition, 19 scalp channels,
montage-order lattice adjacency (adjacency=None) = same as the primary test. Uses saved ERPs, no redraw."""
import json, numpy as np
from scipy import stats
from mne.stats import permutation_cluster_1samp_test
from common import *
a=np.load(f'{OUT}/03_participant_erps.npz',allow_pickle=True); r=np.load(f'{OUT}/07_noninsight_erps.npz',allow_pickle=True)
assert list(a['pids'])==list(r['pids'])
idx=[STANDARD_CHANS.index(c) for c in SCALP_CHANS]; thr=stats.t.ppf(.975,14); times=a['times']; res={}
for cond,lab in [('control','Unassisted'),('treatment','LLM-assisted')]:
    X=np.transpose(a[cond]-r[cond],(0,2,1))[:,:,idx]
    t,cl,pv,_=permutation_cluster_1samp_test(X,n_permutations=32768,threshold=thr,tail=0,adjacency=None,out_type='mask',seed=0,verbose='ERROR')
    i=int(np.argmin(pv)) if len(pv) else None
    res[lab]=dict(n_clusters=len(cl),min_p=float(pv.min()) if len(pv) else None,
                  span=[float(times[np.where(cl[i])[0].min()]),float(times[np.where(cl[i])[0].max()])] if i is not None else None)
print(json.dumps(res,indent=1)); json.dump(res,open(f'{OUT}/11_noninsight_cd_lattice19.json','w'),indent=1)
