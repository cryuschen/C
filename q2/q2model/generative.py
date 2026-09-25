"""Image-driven, three-stage neural population model and nested ERP fitting.

No anatomical inverse claim is made: fitted readout weights are effective scalp
mixing coefficients. Shape preference is not hemisphere or visual-field side.
"""
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


def empirical_basis(kind):
    t = np.maximum(TIMES, 0)
    if kind == 'gamma':
        curves = [(t/p)**r*np.exp(r*(1-t/p)) for p,r in
                  zip((.09,.14,.23,.35,.52,.68), (3,4,5,6,7,8))]
    elif kind == 'step':
        curves = [expit((TIMES-a)/.025) for a in (.08,.18,.32)]
    elif kind == 'ramp':
        curves = [np.maximum(TIMES-a, 0) for a in (0,.18,.35)]
    else:
        raise ValueError(kind)
    h = np.array(curves).T
    h[TIMES < 0] = 0
    p = h.shape[1]
    cd = np.zeros((2,len(TIMES),2*p)); cd[0,:,:p]=h; cd[1,:,p:]=h
    return condition_waves(cd)


@dataclass
class Candidate:
    model: str
    scale: float
    recurrence: float
    h: np.ndarray  # condition x time x sources; same readout for both directions


class ERPBank:
    """Precomputed linear algebra permits full nested re-fitting per permutation."""
    def __init__(self, candidates, lambdas=LAMBDAS, symmetric=False):
        self.candidates = candidates; self.lambdas=np.asarray(lambdas)
        self.symmetric = symmetric
        self.cd=[]; self.inverse=[]; self.gram=[]; self.reg=[]; self.meta=[]
        self.weight = np.tile(WEIGHTS, 2) / 2
        maxp=max(c.h.shape[-1] for c in candidates)
        for c in candidates:
            h=common_delta(c.h).reshape(-1,c.h.shape[-1])
            norm=np.sqrt((self.weight[:,None]*h*h).sum(0))
            h=h/np.maximum(norm,1e-10)
            h=np.pad(h,((0,0),(0,maxp-h.shape[1])))
            gram=h.T@(self.weight[:,None]*h)
            for lam in self.lambdas:
                self.cd.append(h); self.gram.append(gram)
                self.inverse.append(np.linalg.inv(gram+lam*np.eye(maxp)))
                self.reg.append(lam); self.meta.append(c)
        self.cd=np.stack(self.cd); self.inverse=np.stack(self.inverse); self.gram=np.stack(self.gram)
        self.weighted=self.cd*self.weight[None,:,None]

    def stats(self, erp):
        target=common_delta(erp).transpose(0,2,1).reshape(-1,3)
        if self.symmetric:
            target=target.copy(); target[:,1:]=target[:,1:].mean(1,keepdims=True)
        b=np.einsum('ntp,tc->npc',self.weighted,target,optimize=True)
        return b

    def fit(self, erp, index):
        return self.inverse[index]@self.stats(erp)[index]

    def predict(self, coef, index):
        cd=(self.cd[index]@coef).reshape(2,len(TIMES),3).transpose(0,2,1)
        return condition_waves(cd)

    def select(self, x, y, blocks):
        return self.select_prepared(self.prepare(x), y, blocks)

    def prepare(self, x):
        # All regularizers share the same basis: project once per shape candidate.
        h=self.weighted[::len(self.lambdas)].reshape(-1,2,len(TIMES),self.cd.shape[-1])
        z=x.copy()
        if self.symmetric:
            z[:,1:]=z[:,1:].mean(1,keepdims=True)
        return np.stack([
            np.einsum('nct,btp->nbpc',z,h[:,j],optimize=True) for j in range(2)])

    def prepared_stats(self, projected, labels, mask):
        left=projected[:,mask & (labels==-1)].mean(1)
        right=projected[:,mask & (labels==1)].mean(1)
        b=.5*(left[0]+right[0]) + right[1]-left[1]
        return np.repeat(b,len(self.lambdas),axis=0)

    def select_prepared(self, projected, y, blocks, subset=None):
        subset=np.ones(len(y),bool) if subset is None else subset
        losses=np.zeros(len(self.meta)); inner=[]
        for b in np.unique(blocks[subset]):
            tr=(blocks!=b)&subset; va=(blocks==b)&subset
            if any(np.sum(y[m]==c)==0 for m in (tr,va) for c in (-1,1)):
                raise ValueError('An inner split is missing a direction')
            tb=self.prepared_stats(projected,y,tr); vb=self.prepared_stats(projected,y,va)
            coef=np.einsum('nij,njc->nic',self.inverse,tb)
            # Omit target energy, which is identical for every candidate.
            score=(np.einsum('nic,nij,njc->n',coef,self.gram,coef)-
                   2*np.einsum('nic,nic->n',coef,vb))/3
            losses+=score; inner.append(score)
        losses/=len(inner)
        best=int(np.argmin(losses))
        return best, losses


def build_banks(shape):
    simulations={}; candidates=[]; no_shape=[]
    for scale in SCALES:
        for rec in RECURRENCES:
            sim=simulate(shape['inputs'][:2],scale,rec)
            h=observation_sources(sim)
            candidates.append(Candidate('full',scale,rec,h))
            simulations[f'{scale}_{rec}']=sim
            sim0=simulate(np.tile(shape['inputs'][:2].mean(0),(2,1)),scale,rec)
            no_shape.append(Candidate('no_shape',scale,rec,observation_sources(sim0)))
    full=ERPBank(candidates)
    banks={'full':full, 'no_shape':ERPBank(no_shape),
           'no_recurrence':ERPBank([c for c in candidates if c.recurrence==0]),
           'symmetric_readout':ERPBank(candidates,symmetric=True)}
    for kind in ('gamma','step','ramp'):
        banks[kind]=ERPBank([Candidate(kind,np.nan,np.nan,empirical_basis(kind))])
    common=[Candidate('zero',c.scale,c.recurrence,np.tile(c.h.mean(0),(2,1,1))) for c in candidates]
    banks['zero']=ERPBank(common)
    return banks, simulations


def wave_metrics(obs,pred):
    mask=WEIGHTS>0; o=obs[...,mask].ravel(); p=pred[...,mask].ravel()
    sse=np.sum((o-p)**2); sst=np.sum((o-o.mean())**2)
    corr=float(np.corrcoef(o,p)[0,1]) if np.std(o)>1e-12 and np.std(p)>1e-12 else np.nan
    return dict(RMSE=float(np.sqrt(np.mean((o-p)**2))),
                R2=float(1-sse/sst) if sst>1e-12 else np.nan,Corr=corr)


def delta_metrics(obs,pred):
    d=obs[1]-obs[0]; p=pred[1]-pred[0]
    mse=float(np.mean(np.sum((d-p)**2*WEIGHTS,axis=-1)))
    zero=float(np.mean(np.sum(d*d*WEIGHTS,axis=-1)))
    return dict(delta_MSE=mse,zero_MSE=zero,S_delta=1-mse/zero if zero>1e-12 else np.nan)
