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

out_dir = ROOT / 'Q2_unsupervised_results_svm'
(out_dir/'validation').mkdir(parents=True, exist_ok=True)

summary = []
for key in NAMES:
    print(f"Running validation (SVM+Fusion) for {key}...", flush=True)
    ds = mock_independent_data(ROOT, key)
    # Using extras=True so tf['window_mean_9'] doesn't crash since we compute it now
    # wait, tf is computed in our new code regardless of `extras`!
    rows, *_ = nested_cv(ds, quick=False, extras=False, out=None)
    summary.extend(block_intervals(rows, 2000, 20260924))

save_csv(summary, out_dir / 'validation/classification_summary.csv')

old_summary = pd.read_csv(ROOT / 'Q2_unsupervised_results/validation/classification_summary.csv')
new_summary = pd.read_csv(out_dir / 'validation/classification_summary.csv')

print("\n--- RESULTS COMPARISON: LDA vs SVM+FUSION ---")
for key in NAMES:
    old = old_summary[(old_summary['dataset'] == key) & (old_summary['model'] == 'mechanism_selected')]
    new = new_summary[(new_summary['dataset'] == key) & (new_summary['model'] == 'mechanism_svm_fused')]
    if len(old) == 0 or len(new) == 0: continue
    old_ba, new_ba = old['BA'].values[0], new['BA'].values[0]
    old_auc, new_auc = old['AUC'].values[0], new['AUC'].values[0]
    
    print(f"Dataset {key}:")
    print(f"  LDA BA: {old_ba:.4f} -> SVM BA: {new_ba:.4f} (Diff: {new_ba - old_ba:+.4f})")
    print(f"  LDA AUC: {old_auc:.4f} -> SVM AUC: {new_auc:.4f} (Diff: {new_auc - old_auc:+.4f})")
