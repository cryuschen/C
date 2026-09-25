import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/home/cryus/code-project/C')
sys.path.insert(0, str(ROOT / 'Q2'))

from q2model import data as q2data
from q2model.data import load_raw, NAMES, Dataset
from q2model.validation import nested_cv, block_intervals
from q2model.experiments import save_csv

def mock_independent_data(root, key, guard=24., out=None):
    p = ROOT / 'eeg_unsupervised_results' / NAMES[key] / '可复核波形.npz'
    with np.load(p) as w:
        raw, on, y, durations = load_raw(ROOT, key)
        ids = w['trial_ids'].copy()
        idx = ids - 1
        ep = w['v7'].copy()
        ds = Dataset(key, ep, y[idx], ids, idx//20, durations[idx],
                     on[idx], np.zeros((len(idx),3)), ep[:,:,:64].copy())
    return ds

out_dir = ROOT / 'Q2_unsupervised_results'
(out_dir/'validation').mkdir(parents=True, exist_ok=True)

summary = []
for key in NAMES:
    print(f"Running validation for {key}...", flush=True)
    ds = mock_independent_data(ROOT, key)
    # The return value is (rows, out_roles, refits)
    rows, *_ = nested_cv(ds, quick=False, extras=False, out=None)
    summary.extend(block_intervals(rows, 2000, 20260924))

save_csv(summary, out_dir / 'validation/classification_summary.csv')

orig_summary = pd.read_csv(ROOT / 'Q2_result/validation/classification_summary.csv')
unsup_summary = pd.read_csv(out_dir / 'validation/classification_summary.csv')

print("\n--- RESULTS COMPARISON ---")
for key in NAMES:
    for model in ['mechanism_selected']:
        o = orig_summary[(orig_summary['dataset'] == key) & (orig_summary['model'] == model)]
        u = unsup_summary[(unsup_summary['dataset'] == key) & (unsup_summary['model'] == model)]
        if len(o) == 0 or len(u) == 0: continue
        orig_ba, unsup_ba = o['BA'].values[0], u['BA'].values[0]
        orig_auc, unsup_auc = o['AUC'].values[0], u['AUC'].values[0]
        
        print(f"Dataset {key}:")
        print(f"  Orig BA: {orig_ba:.4f} -> Unsup BA: {unsup_ba:.4f} (Diff: {unsup_ba - orig_ba:+.4f})")
        print(f"  Orig AUC: {orig_auc:.4f} -> Unsup AUC: {unsup_auc:.4f} (Diff: {unsup_auc - orig_auc:+.4f})")
