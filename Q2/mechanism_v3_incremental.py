"""Exploratory held-out test of neural basis beyond smooth nuisance templates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mechanism_v3_transfer import make_bases, evaluate_dataset, summary_metrics
from q2model.data import NAMES, independent_data, save_csv, save_json
from optimize_direction_features import ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path,
                        default=ROOT / 'Q2_mechanism_v3_transfer_verified_v2')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'Q2_mechanism_v3_incremental_v2')
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Use a fresh output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    b, _ = make_bases()
    additions = {
        'ocular_plus_neural': np.column_stack((b['ocular_step'], b['visual_neural_mass'])),
        'ramp_plus_neural': np.column_stack((b['slow_ramp'], b['visual_neural_mass'])),
    }
    folds, choices = [], []
    for key in NAMES:
        r, c, _ = evaluate_dataset(independent_data(ROOT, key), additions)
        folds.extend(r)
        choices.extend(c)
    combined = summary_metrics(folds)
    base = pd.read_csv(args.base / 'heldout_summary.csv')
    comparisons = []
    for r in combined.itertuples():
        reference = 'ocular_step' if r.model == 'ocular_plus_neural' else 'slow_ramp'
        row = base[(base.dataset == r.dataset) & (base.model == reference)].iloc[0]
        if not np.isclose(row.zero_MSE, r.zero_MSE):
            raise ValueError('Different evaluation folds or contrast target')
        comparisons.append(dict(dataset=r.dataset, combined_model=r.model,
                                nuisance_reference=reference,
                                S_delta_combined=r.S_delta,
                                S_delta_nuisance=row.S_delta,
                                incremental_S_delta=(row.MSE-r.MSE)/r.zero_MSE))
    save_csv(folds, args.output / 'combined_folds.csv')
    combined.to_csv(args.output / 'combined_summary.csv', index=False, encoding='utf-8-sig')
    save_csv(comparisons, args.output / 'incremental_summary.csv')
    save_json(choices, args.output / 'inner_choices.json')
    save_json(dict(status='exploratory_after_primary_V3_results',
                   source_base=str(args.base),
                   note='same outer folds, added basis has six temporal components and an inner-selected shared ridge'),
              args.output / 'manifest.json')
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == '__main__':
    main()
