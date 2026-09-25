import sys
from pathlib import Path
sys.path.insert(0, '/home/cryus/code-project/C/Q2')
from q2model import data as q2data
from q2model.data import load_raw, NAMES, Dataset
from q2model.validation import nested_cv

def mock_independent_data(root, key, guard=24., out=None):
    import numpy as np
    p = Path(root).parent / 'eeg_unsupervised_results' / NAMES[key] / '可复核波形.npz'
    with np.load(p) as w:
        raw, on, y, durations = load_raw(Path(root).parent, key)
        ids = w['trial_ids'].copy()
        idx = ids - 1
        ep = w['v7'].copy()
        ds = Dataset(key, ep, y[idx], ids, idx//20, durations[idx],
                     on[idx], np.zeros((len(idx),3)), ep[:,:,:64].copy())
    return ds

ds = mock_independent_data('/home/cryus/code-project/C/Q2', 'A1')
res = nested_cv(ds, quick=True, extras=False)
print("Type of res:", type(res))
if isinstance(res, tuple):
    print("Len of res:", len(res))
    print(res[0][0].keys() if isinstance(res[0], list) and len(res[0]) > 0 else type(res[0]))
else:
    print(res[0].keys() if len(res) > 0 else "Empty")
