# 问题二：形状编码、神经群体动力学与方向判别
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, label, maximum_filter

DOCX = '服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx'
MEDIA = 'word/media/image12.png'
def cue_images(root: Path):
    with ZipFile(Path(root) / DOCX) as archive:
        raw = archive.read(MEDIA)
    image = np.asarray(Image.open(BytesIO(raw)).convert('RGB')).astype(np.float64)
    blue = np.maximum(0., image[:, :, 2] - np.maximum(image[:, :, 0], image[:, :, 1])) / 255
    mask = blue > .2
    components, count = label(mask)
    pieces = sorted([(int(np.sum(components == j)), j) for j in range(1, count + 1)], reverse=True)
    if len(pieces) < 2 or pieces[0][0] < 80 or pieces[1][0] < 40:
        raise ValueError('Cannot identify the triangle and fixation circle in the problem image')
    top, bottom = sorted((pieces[0][1], pieces[1][1]),
                         key=lambda j: np.mean(np.nonzero(components == j)[0]))
    triangle = np.where(components == top, blue, 0.)
    circle = np.where(components == bottom, blue, 0.)
    center_x = int(round(np.mean(np.nonzero(circle)[1])))
    left_triangle = np.zeros_like(triangle)
    ys, xs = np.nonzero(triangle)
    reflected_x = 2 * center_x - xs
    inside = (reflected_x >= 0) & (reflected_x < triangle.shape[1])
    left_triangle[ys[inside], reflected_x[inside]] = triangle[ys[inside], xs[inside]]
    if not np.isclose(triangle.sum(), left_triangle.sum(), rtol=0, atol=1e-10):
        raise ValueError('Mirroring lost triangle pixels')
    return dict(right=triangle + circle, left=left_triangle + circle,
                right_triangle=triangle, left_triangle=left_triangle,
                circle=circle, center_x=center_x,
                source_size=image.shape[:2], source_media=MEDIA)


def oriented_edges(image: np.ndarray) -> np.ndarray:
    smooth = gaussian_filter(image, .8)
    gx = gaussian_filter(smooth, 1., order=(0, 1))
    gy = gaussian_filter(smooth, 1., order=(1, 0))
    angles = np.arange(8) * np.pi / 8
    return np.stack([np.abs(gx * np.cos(a) + gy * np.sin(a)) for a in angles])


def sites(template: np.ndarray, n: int = 18) -> list[tuple[int, int, int]]:
    response = oriented_edges(template)
    peak = response.max(0)
    ranked = np.argsort(peak.ravel())[::-1]
    selected = []
    width = template.shape[1]
    for flat in ranked:
        y, x = divmod(int(flat), width)
        if peak[y, x] < .025 * peak.max():
            break
        if any((y - yy) ** 2 + (x - xx) ** 2 < 9 for yy, xx, _ in selected):
            continue
        orientation = int(np.argmax(response[:, y, x]))
        selected.append((y, x, orientation))
        if len(selected) == n:
            break
    if len(selected) < n:
        raise ValueError('Too few template edge sites')
    return selected


def shape_inputs(cues: dict):
    names = ('left', 'right')
    triangle = (cues['left_triangle'], cues['right_triangle'])
    templates = [sites(item) for item in triangle]
    # One-pixel tolerance; two pixels would pool across much of this 13-pixel cue.
    pooled = [maximum_filter(oriented_edges(item), size=(1, 3, 3))
              for item in triangle]
    activation = np.zeros((2, 2))
    for i, response in enumerate(pooled):
        for j, selected in enumerate(templates):
            values = np.array([response[o, y, x] for y, x, o in selected])
            activation[i, j] = np.exp(np.mean(np.log(values + 1e-5)))
    normalization = np.maximum(np.diag(activation), 1e-10)
    activation /= normalization[None, :]
    # A shared, direction-neutral fixation input is supplied separately.
    common = np.ones(2)
    return dict(inputs=np.column_stack([activation, common]),
                activation=activation, sites=np.array(templates), names=names,
                selectivity=(activation[:, 1] - activation[:, 0]) /
                            (activation[:, 1] + activation[:, 0] + 1e-9))
# ---- 神经群体动力学与留出拟合 ----
from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import expit
from scipy.ndimage import maximum_filter
from .stimulus_shape import cue_images, oriented_edges, sites
from .data import TIMES, WEIGHTS, FS, preprocess, means

SCALES = (.75, 1., 1.25)
RECURRENCES = (0., .1, .2)
LAMBDAS = np.array([.1, .3, 1., 3., 10., 30., 100.])
CHANNELS = ('Fz', 'F3', 'F4')
STAGES = ('早期视觉', '形状整合', '额区响应')


def encode_shapes(root_path):
    cue = cue_images(root_path)
    left, right = cue['left_triangle'], cue['right_triangle']
    yy, xx = np.nonzero(right)
    missing = right.copy()
    missing[:, int(np.quantile(xx, .70)):] = 0
    # Permute narrow vertical strips within the triangle bounding box only.
    scrambled = right.copy()
    lo, hi = int(xx.min()), int(xx.max()) + 1
    strips = np.array_split(np.arange(lo, hi), 3)
    scrambled[:, lo:hi] = right[:, np.concatenate([strips[1], strips[2], strips[0]])]
    images = np.stack([left, right, missing, scrambled])
    templates = [sites(left), sites(right)]
    response = np.stack([maximum_filter(oriented_edges(im), size=(1, 3, 3)) for im in images])
    activation = np.array([[np.exp(np.mean(np.log([r[o, y, x] + 1e-5
                           for y, x, o in positions]))) for positions in templates]
                           for r in response])
    activation /= np.maximum(np.diag(activation[:2]), 1e-10)[None, :]
    return dict(images=images, inputs=np.c_[activation, np.ones(4)],
                names=np.array(['left', 'right', 'missing_tip', 'scrambled']),
                sites=np.array(templates), circle=cue['circle'],
                source_media=cue['source_media'])


def simulate(inputs, scale=1., recurrence=.1, duration=52/256, max_step=.002):
    """LGN (3), E/I (3 stages x 3 groups), two-stage synapses (36 states).

    Time constants and couplings are modelling assumptions. Excitatory and
    inhibitory postsynaptic filters give q = vE - .7 vI, not raw spike rates.
    """
    inputs = np.atleast_2d(inputs)
    tau_e = scale * np.array([.020, .040, .080])[:, None]
    tau_i = 2 * tau_e
    tau_se = scale * np.array([.025, .050, .100])[:, None]
    tau_si = 3 * tau_se
    W = (np.ones((3, 3)) - np.eye(3)) / 2
    def neural(g, e, i):
        drive = np.vstack([g, 3 * (e[:-1] - resting_e[:-1])])
        de = (-e + expit(e - i + recurrence * (e @ W.T) + drive - 2)) / tau_e
        di = (-i + expit(e - i - 2)) / tau_i
        return de, di
    # Resting state is identical in all stages; feedforward deviations vanish.
    eq = root(lambda z: np.r_[-z[:3] + expit(z[:3]-z[3:] + recurrence*W@z[:3]-2),
                              -z[3:] + expit(z[:3]-z[3:]-2)], np.full(6, .12))
    if not eq.success:
        raise RuntimeError('Resting-state solution failed')
    resting_e = np.tile(eq.x[:3], (3, 1))
    resting_i = np.tile(eq.x[3:], (3, 1))
    state0 = np.r_[np.zeros(3), resting_e.ravel(), resting_i.ravel(), np.zeros(36)]
    def derivative(t, state, feature):
        g = state[:3]; e = state[3:12].reshape(3, 3); i = state[12:21].reshape(3, 3)
        ue, ve, ui, vi = state[21:].reshape(4, 3, 3)
        de, di = neural(g, e, i)
        drive = 3 * feature if 0 <= t < duration else np.zeros(3)
        return np.r_[(-g + drive)/(.020*scale), de.ravel(), di.ravel(),
                     ((e-resting_e-ue)/tau_se).ravel(), ((ue-ve)/tau_se).ravel(),
                     ((i-resting_i-ui)/tau_si).ravel(), ((ui-vi)/tau_si).ravel()]
    eps = 1e-6
    jac = np.column_stack([(derivative(-1, state0 + eps*v, np.zeros(3)) -
                            derivative(-1, state0 - eps*v, np.zeros(3)))/(2*eps)
                           for v in np.eye(len(state0))])
    eigmax = float(np.linalg.eigvals(jac).real.max())
    if eigmax >= 0:
        raise RuntimeError('Unstable model configuration')
    t = np.arange(-250, 4001)/1000
    states = []
    # Integrate separately at stimulus discontinuities, avoiding adaptive-step skips.
    for feature in inputs:
        state = state0.copy(); parts = []
        for start, stop in ((-.25, 0.), (0., duration), (duration, 4.)):
            idx = np.flatnonzero((t >= start) & ((t < stop) if stop < 4 else (t <= stop)))
            sol = solve_ivp(lambda tt, z: derivative(tt, z, feature), (start, stop), state,
                            dense_output=True, max_step=max_step, rtol=1e-7, atol=1e-9)
            if not sol.success:
                raise RuntimeError(sol.message)
            parts.append(sol.sol(t[idx])); state = sol.y[:, -1]
        states.append(np.concatenate(parts, axis=1))
    states = np.stack(states)
    syn = states[:, 21:].reshape(len(inputs), 4, 3, 3, len(t))
    q = syn[:, 1] - .7*syn[:, 3]
    return dict(t=t, state=states, q=q, eigenmax=eigmax, scale=scale,
                recurrence=recurrence, duration=duration)


def observation_sources(sim):
    """Apply the EEG observation filter at the actual 256-Hz sampling rate.

    Zero padding represents resting activity, not extra fitted trials. Raw
    neural states remain causal; the observed zero-phase-filtered waves need not.
    """
    grid = np.arange(-32*FS, 36*FS)/FS
    q = sim['q'].reshape(len(sim['q']), 9, -1)
    full = np.array([[np.interp(grid, sim['t'], s, left=0, right=0) for s in cond] for cond in q])
    filtered = preprocess(full)
    ix = np.rint((TIMES-grid[0])*FS).astype(int)
    sampled = filtered[..., ix]
    sampled -= np.median(sampled[..., TIMES < 0], axis=-1, keepdims=True)
    return sampled.transpose(0, 2, 1)


def common_delta(a):
    return np.stack([a.mean(axis=0), a[1]-a[0]])


def condition_waves(cd):
    return np.stack([cd[0]-.5*cd[1], cd[0]+.5*cd[1]])


def select_contrast_shrinkage(bank, projected, ds, train, factors=(0., .25, .5, 1.)):
    """Select contrast attenuation using training blocks only.

    Each inner template is itself chosen without its validation block. Common
    response is unchanged. This is predictive regularization, not a new source.
    """
    errors=np.zeros(len(factors)); records=[]
    for held in np.unique(ds.blocks[train]):
        inner=train & (ds.blocks!=held); val=train & (ds.blocks==held)
        best,_=bank.select_prepared(projected,ds.y,ds.blocks,inner)
        coef=bank.inverse[best]@bank.prepared_stats(projected,ds.y,inner)[best]
        pred=bank.predict(coef,best); observed=means(ds.x[val],ds.y[val])
        for j,factor in enumerate(factors):
            cd=common_delta(pred); cd[1]*=factor
            loss=delta_metrics(observed,condition_waves(cd))['delta_MSE']
            errors[j]+=loss
        records.append(dict(inner_held=int(held),template_candidate=int(best),
                            train_ids=ds.ids[inner].tolist(),validation_ids=ds.ids[val].tolist()))
    errors/=len(records)
    return float(factors[int(np.argmin(errors))]),errors,records
# ---- 机制增强特征与块留出判别 ----
import numpy as np
import pandas as pd
from scipy.fft import dct
from sklearn.metrics import roc_auc_score

from q2model.data import NAMES, TIMES, independent_data, modes, save_csv, save_json


ROOT = Path(__file__).resolve().parents[1]
POST_MASK = (TIMES >= .05) & (TIMES < .75)
N_DCT = 6
SHRINK = .8  # fixed isotropic shrinkage of within-class covariance
PRE_TO_POST_RIDGE_FACTOR = 1.0  # penalty = number of training trials
MODELS = ('past_only', 'post_erp', 'post_given_past')
MODES = ('common', 'midline', 'lateral')


def dct_features(x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Orthonormal low-frequency coefficients, 3 scalp modes x 6 time modes."""
    if x.ndim != 3 or x.shape[1] != 3:
        raise ValueError('Expected trials x Fz/F3/F4 x time')
    segment = x if mask is None else x[..., mask]
    if segment.shape[-1] < N_DCT:
        raise ValueError('Not enough time samples for DCT')
    return dct(modes(segment), type=2, norm='ortho', axis=-1)[..., :N_DCT].reshape(len(x), -1)


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    center = train.mean(axis=0)
    scale = np.maximum(train.std(axis=0), 1e-8)
    return np.clip((train - center) / scale, -8, 8), np.clip((test - center) / scale, -8, 8)


def remove_past_prediction(pre_train: np.ndarray, pre_test: np.ndarray,
                           post_train: np.ndarray, post_test: np.ndarray
                           ) -> tuple[np.ndarray, np.ndarray]:
    """Label-blind, train-only ridge regression of post waveform on past EEG."""
    if pre_train.shape[0] != post_train.shape[0] or pre_test.shape[0] != post_test.shape[0]:
        raise ValueError('Paired EEG features required')
    penalty = PRE_TO_POST_RIDGE_FACTOR * len(pre_train)
    transform = np.linalg.solve(pre_train.T @ pre_train + penalty * np.eye(pre_train.shape[1]),
                                pre_train.T @ post_train)
    return post_train - pre_train @ transform, post_test - pre_test @ transform


def erp_score(train: np.ndarray, labels: np.ndarray, test: np.ndarray) -> np.ndarray:
    """Shrinkage-whitened right-minus-left ERP template, balanced prior."""
    if set(np.unique(labels)) != {-1, 1}:
        raise ValueError('Both cue directions required in training')
    left, right = train[labels == -1], train[labels == 1]
    mu_left, mu_right = left.mean(axis=0), right.mean(axis=0)
    residual = np.vstack((left - mu_left, right - mu_right))
    cov = residual.T @ residual / max(len(train) - 2, 1)
    identity_scale = np.trace(cov) / cov.shape[0]
    regularized = (1 - SHRINK) * cov + SHRINK * max(identity_scale, 1e-6) * np.eye(cov.shape[0])
    weight = np.linalg.solve(regularized, mu_right - mu_left)
    return (test - .5 * (mu_left + mu_right)) @ weight


def load_datasets() -> dict[str, dict[str, np.ndarray]]:
    out = {}
    for index, key in enumerate(NAMES):
        ds = independent_data(ROOT, key)
        out[key] = dict(pre=dct_features(ds.prestim_only), post=dct_features(ds.x, POST_MASK),
                        y=ds.y, blocks=ds.blocks, ids=ds.ids,
                        global_blocks=ds.blocks + 5 * index)
    return out


def decode_one(data: dict[str, np.ndarray], labels: np.ndarray,
               return_features: bool = False) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Five contiguous-block folds; no outer-test labels enter any fitting step."""
    scores = {name: np.full(len(labels), np.nan) for name in MODELS}
    features = {name: np.full_like(data['post'], np.nan) for name in MODELS} if return_features else {}
    for held in np.unique(data['blocks']):
        train, test = data['blocks'] != held, data['blocks'] == held
        pre_train, pre_test = standardize(data['pre'][train], data['pre'][test])
        post_train, post_test = standardize(data['post'][train], data['post'][test])
        residual_train, residual_test = remove_past_prediction(
            pre_train, pre_test, post_train, post_test)
        fold = {'past_only': (pre_train, pre_test), 'post_erp': (post_train, post_test),
                'post_given_past': (residual_train, residual_test)}
        for name, (x_train, x_test) in fold.items():
            scores[name][test] = erp_score(x_train, labels[train], x_test)
            if return_features:
                features[name][test] = x_test
    if any(not np.isfinite(arr).all() for arr in scores.values()):
        raise ValueError('Incomplete outer-fold predictions')
    return scores, features


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    pred = np.where(scores >= 0, 1, -1)
    return dict(n=int(len(y)), BA=float(.5 * ((pred[y == -1] == -1).mean() +
                                              (pred[y == 1] == 1).mean())),
                AUC=float(roc_auc_score(y, scores)),
                recall_left=float((pred[y == -1] == -1).mean()),
                recall_right=float((pred[y == 1] == 1).mean()))


def evaluate(data: dict[str, dict[str, np.ndarray]],
             labels: dict[str, np.ndarray], return_features: bool = False):
    score_map, feature_map = {}, {}
    for key, group in data.items():
        score_map[key], feature_map[key] = decode_one(group, labels[key], return_features)
    rows = []
    for key in ('pooled', *NAMES):
        for name in MODELS:
            yy = np.concatenate([labels[k] for k in NAMES]) if key == 'pooled' else labels[key]
            ss = np.concatenate([score_map[k][name] for k in NAMES]) if key == 'pooled' else score_map[key][name]
            rows.append(dict(dataset=key, model=name, **metrics(yy, ss)))
    return pd.DataFrame(rows), score_map, feature_map


def permute_within_blocks(data: dict[str, dict[str, np.ndarray]],
                          rng: np.random.Generator) -> dict[str, np.ndarray]:
    labels = {}
    for key, group in data.items():
        labels[key] = group['y'].copy()
        for block in np.unique(group['blocks']):
            index = np.flatnonzero(group['blocks'] == block)
            labels[key][index] = rng.permutation(labels[key][index])
    return labels


def block_bootstrap(data, score_map, n_boot: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for dataset in ('pooled', *NAMES):
        for model in MODELS:
            vals = []
            for _ in range(n_boot):
                yy, ss = [], []
                for key in NAMES if dataset == 'pooled' else (dataset,):
                    group = data[key]
                    blocks = np.unique(group['blocks'])
                    chosen = rng.choice(blocks, len(blocks), replace=True)
                    index = np.concatenate([np.flatnonzero(group['blocks'] == b) for b in chosen])
                    yy.append(group['y'][index]); ss.append(score_map[key][model][index])
                y = np.concatenate(yy); score = np.concatenate(ss)
                pred = np.where(score >= 0, 1, -1)
                vals.append(float(.5 * ((pred[y == -1] == -1).mean() +
                                           (pred[y == 1] == 1).mean())))
            low, high = np.quantile(vals, [.025, .975])
            rows.append(dict(dataset=dataset, model=model, BA_low=float(low), BA_high=float(high),
                             bootstraps=n_boot, resampling_unit='contiguous_time_block'))
    return pd.DataFrame(rows)

