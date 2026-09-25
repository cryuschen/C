"""Q1 pooled-template correction fitted inside every Q2 validation split.

The adapter preserves the current Q1 candidate-selection rule (which uses
TRAINING labels). Transform accepts EEG only. It is not fully unsupervised.
No stored Q1 corrected waveforms are consumed here.
"""
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
import importlib.util
import hashlib
import numpy as np
from .data import means
from .model import LAMBDAS, fit_erp, weighted_error


@lru_cache(maxsize=1)
def q1_module():
    path = Path(__file__).resolve().parents[2] / 'EEG_P300_artifact_correction_unsupervised.py'
    spec = importlib.util.spec_from_file_location('q1_pooled_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class FoldDenoiser:
    model: tuple
    candidate: str
    mean_delta: dict
    audit: dict

    def transform(self, x):
        """Apply one fixed pooled model; no labels or other trials are needed."""
        if x.ndim != 3 or x.shape[1:] != self.model[0][-1].shape:
            raise ValueError('Unexpected EEG shape')
        # Both keys contain exactly the same fitted quantities. A constant key
        # avoids exposing a direction-label argument on the inference API.
        q = q1_module()
        return q.apply_v7(x, np.ones(len(x), dtype=int), self.model,
                          self.candidate, self.mean_delta)

    def arrays(self):
        return dict(template=self.model[0][-1], scale_v=self.model[1][-1],
                    scale_h=self.model[2][-1], mean_delta=self.mean_delta[-1])


def fit_denoiser(training, seed=20260925):
    q = q1_module()
    x, y = training.x, training.y
    if len(x) < 12 or min(np.sum(y == c) for c in (-1, 1)) < 4:
        raise ValueError('Insufficient training data for the fixed Q1 selector')
    scores = q.detect_bad_trials(np.zeros_like(x), x)[1]
    model = q.build_unsupervised_reference_model(x, y, np.zeros(len(x), bool), scores)
    candidate, losses = q.select_candidate(x, y, seed)
    delta = q.training_mean_delta(x, y, model, candidate)
    for component in model[:3]:
        np.testing.assert_array_equal(component[-1], component[1])
    np.testing.assert_array_equal(delta[-1], delta[1])
    tr, va = q.train_test_split(np.arange(len(x)), test_size=.35,
                                stratify=y, random_state=seed)
    audit = dict(train_ids=training.ids.tolist(), train_blocks=np.unique(training.blocks).tolist(),
                 reference_ids=training.ids[model[3]].tolist(), candidate=candidate,
                 policy=q.selected_policy(candidate), seed=seed,
                 selector_train_ids=training.ids[tr].tolist(),
                 selector_validation_ids=training.ids[va].tolist(),
                 selector_losses=losses, training_labels_used=True,
                 inference_labels_used=False)
    return FoldDenoiser(model, candidate, delta, audit)


class DenoiserCache:
    """Reuse only an identical training set, including its labels and samples."""
    def __init__(self, seed=20260925):
        self.seed = seed
        self.models = {}

    def fit(self, training):
        digest = hashlib.sha256()
        for value in (training.ids, training.y, training.x):
            digest.update(np.ascontiguousarray(value).tobytes())
        key = digest.hexdigest()
        if key not in self.models:
            self.models[key] = fit_denoiser(training, self.seed)
        return self.models[key]


def transform_pair(training, validation, cache):
    if set(training.ids) & set(validation.ids):
        raise ValueError('Training and validation trials overlap')
    denoiser = cache.fit(training)
    # Preserve strict pre-stimulus controls and quality records. No new
    # stage-dependent rejection is allowed in this paired comparison.
    return (replace(training, x=denoiser.transform(training.x)),
            replace(validation, x=denoiser.transform(validation.x)), denoiser)


def fit_prepared_model(final_training, inner_pairs, quick=False):
    """Existing Q2 grid/refinement rule with explicitly prepared inner splits.

    Kept separate from model.train_model to preserve existing Q2 entry points
    and the user's in-progress edits. An equivalence regression checks that
    uncorrected pairs reproduce train_model exactly.
    """
    all_ids = set(final_training.ids)
    if len(inner_pairs) != len(np.unique(final_training.blocks)):
        raise ValueError('One inner validation split per training block required')
    seen = []
    for tr, va in inner_pairs:
        if set(tr.ids) & set(va.ids) or set(tr.ids) | set(va.ids) != all_ids:
            raise ValueError('Invalid inner split lineage')
        if len(np.unique(va.blocks)) != 1 or set(tr.blocks) & set(va.blocks):
            raise ValueError('Inner validation must hold out complete time blocks')
        seen.extend(va.ids.tolist())
    if sorted(seen) != sorted(all_ids):
        raise ValueError('Every training trial must be held out exactly once')
    rows, candidates = [], []
    for rank in (2, 3):
        for lam in LAMBDAS:
            losses = []
            for tr, va in inner_pairs:
                fit = fit_erp(means(tr.x, tr.y), np.median(tr.durations), rank, lam,
                              20., quick, refine=False)
                loss = weighted_error(means(va.x, va.y), fit.predict())
                losses.append(loss)
                rows.append(dict(rank=rank, lambda_A=lam,
                                 validation_block=int(va.blocks[0]), loss=loss,
                                 train_ids=tr.ids.tolist(), validation_ids=va.ids.tolist()))
            candidates.append((np.mean(losses), rank, lam,
                               float(np.std(losses, ddof=1) / np.sqrt(len(losses)))))
    minimum = min(c[0] for c in candidates)
    eligible = [c for c in candidates if c[0] <= minimum + max(1e-10, .01*abs(minimum))]
    chosen = min(eligible, key=lambda c: (c[1], -c[2], c[0]))
    fit = fit_erp(means(final_training.x, final_training.y), np.median(final_training.durations),
                  chosen[1], chosen[2], 20., quick, True)
    fit.diagnostics.update(inner_validation=rows, selection_candidates=candidates)
    return fit


def prepare_fold(ds, heldout_block, branch, cache):
    tr = ds.subset(ds.blocks != heldout_block)
    te = ds.subset(ds.blocks == heldout_block)
    if branch not in ('baseline', 'denoised'):
        raise ValueError('Unknown branch')
    pairs, inner_audits = [], []
    for block in np.unique(tr.blocks):
        it, iv = tr.subset(tr.blocks != block), tr.subset(tr.blocks == block)
        if branch == 'denoised':
            it, iv, d = transform_pair(it, iv, cache)
            inner_audits.append(dict(validation_block=int(block), validation_ids=iv.ids.tolist(),
                                     denoiser=d.audit))
        pairs.append((it, iv))
    if branch == 'denoised':
        tr, te, denoiser = transform_pair(tr, te, cache)
    else:
        denoiser = None
    return tr, te, pairs, denoiser, inner_audits
