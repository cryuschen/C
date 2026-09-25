"""Checks for the new mechanistic contrast transfer experiment."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Q2'))

from mechanism_v3_transfer import evaluate_dataset, make_bases, project  # noqa: E402
from q2model.data import independent_data  # noqa: E402


def test_shape_driven_neural_basis_and_ridge_limit():
    bases, meta = make_bases()
    activation = np.asarray(meta['shape_activation'])
    assert activation.shape == (2, 2)
    assert activation[0, 0] > activation[0, 1]
    assert activation[1, 1] > activation[1, 0]
    neural = bases['visual_neural_mass']
    assert neural.shape[1] == 3 and np.isfinite(neural).all()
    target = np.ones((3, neural.shape[0]))
    assert np.linalg.norm(project(target, neural, 1000.)) < np.linalg.norm(
        project(target, neural, .1))


def test_held_block_labels_do_not_change_its_predictions():
    ds = independent_data(ROOT, 'B2')
    bases, _ = make_bases()
    _, original_choices, original = evaluate_dataset(ds, bases)
    changed_labels = ds.y.copy()
    changed_labels[ds.blocks == 0] *= -1
    altered = ds.labels(changed_labels)
    _, changed_choices, second = evaluate_dataset(altered, bases)
    for model in bases:
        key = f'{model}_0_predicted'
        np.testing.assert_array_equal(original[key], second[key])
        before = next(v for v in original_choices if v['model'] == model and v['held_block'] == 0)
        after = next(v for v in changed_choices if v['model'] == model and v['held_block'] == 0)
        assert before['selected_lambda'] == after['selected_lambda']
        assert set(before['train_ids']).isdisjoint(before['test_ids'])
