"""Separable ridge regression; label-free kernel banks and supervised bounded refinement.

The inner search exhaustively profiles a fixed bounded grid. Each selected final
training fit is refined from multiple grid minima. Permutations run the SAME
search/refinement; only fixed basis banks may be reused.
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
from scipy.fft import rfft, irfft
from scipy.signal import freqz, sosfreqz, find_peaks
from scipy.optimize import minimize
from .data import FS, TIMES, B, WEIGHTS, FIT_MASK, filters, means, modes

LAMBDAS = (.01, .1, 1.)
BOUNDS = [(20.,150.), (20.,100.), (70.,250.), (150.,600.)]


class KernelFactory:
    def __init__(self, duration=.203125, tau_g=20., nfft=16384):
        self.duration = float(duration); self.tau_g=float(tau_g); self.nfft=nfft
        self.w = 2*np.pi*np.fft.rfftfreq(nfft)
        self.z = np.exp(-1j*self.w)
        b,a,sos=filters()
        _, hn = freqz(b,a,worN=self.w)
        _, hb = sosfreqz(sos,worN=self.w)
        impulse=np.zeros(nfft); width=int(round(duration*FS)); impulse[:width]=1.
        self.drive = rfft(impulse) * np.abs(hn*hb)**2 * self.exp_transfer(tau_g)
        self.indices = (np.arange(-64,205) % nfft)

    def exp_transfer(self, tau):
        a=np.exp(-1000/(FS*tau))
        return (1-a)/(1-a*self.z)

    def alpha_transfer(self, tau):
        a=np.exp(-1000/(FS*tau))
        return (1-a)**2*self.z/(1-a*self.z)**2

    def kernel(self, theta, rank):
        delay,te,ti=theta[:3]
        samples=delay*FS/1000; j=int(np.floor(samples)); frac=samples-j
        delayed=self.drive*self.z**j*((1-frac)+frac*self.z)
        he=self.alpha_transfer(te); hi=self.alpha_transfer(ti)
        spectra=[delayed*he, delayed*hi]
        if rank==3: spectra.append(delayed*he*self.exp_transfer(theta[3]))
        full=irfft(np.stack(spectra), n=self.nfft, axis=-1)
        h=full[:,self.indices].T
        h=h-np.median(h[:64],axis=0)
        norms=np.sqrt(np.einsum('tr,t,tr->r',h,WEIGHTS,h))
        if np.any(norms<1e-12): raise ValueError('Degenerate response kernel')
        return h/norms


def allowed(theta, rank):
    return theta[2]>=theta[1]+10 and (rank==2 or theta[3]>=theta[2]+10)


@lru_cache(maxsize=24)
def bank(duration_samples, rank, tau_g=20., quick=False):
    fac=KernelFactory(duration_samples/FS,tau_g)
    n=3 if quick else 4
    axes=[np.linspace(a,b,n) for a,b in BOUNDS[:rank+1]]
    mesh=np.array(np.meshgrid(*axes,indexing='ij')).reshape(rank+1,-1).T
    theta=mesh[[allowed(t,rank) for t in mesh]]
    h=np.stack([fac.kernel(t,rank) for t in theta])
    gram=np.einsum('ktr,t,kts->krs',h,WEIGHTS,h)
    # Precomputed maps are independent of every EEG sample and label.
    maps={}
    for lam in LAMBDAS:
        maps[lam]=tuple(np.linalg.inv(gram+v*np.eye(rank)) for v in (lam/2,5*lam))
    return fac,theta,h,gram,maps


def target_components(erp):
    z=np.einsum('cm,dct->dmt',B,erp)
    return (z[0]+z[1])/2,(z[1]-z[0])/2


def coefficients(erp, h, lam, direction=True):
    avg,diff=target_components(erp)
    gram=h.T@(WEIGHTS[:,None]*h)
    A=np.linalg.solve(gram+lam/2*np.eye(h.shape[1]),h.T@(WEIGHTS[:,None]*avg.T)).T
    D=np.linalg.solve(gram+5*lam*np.eye(h.shape[1]),h.T@(WEIGHTS[:,None]*diff.T)).T if direction else np.zeros_like(A)
    return A,D


def predict_erp(A,D,h):
    return np.stack([B@(A+c*D)@h.T for c in (-1,1)])


def weighted_error(target,pred):
    return float(np.einsum('dct,t,dct->',target-pred,WEIGHTS,target-pred)/6)


@dataclass
class Fit:
    rank: int
    lam: float
    theta: np.ndarray
    A: np.ndarray
    D: np.ndarray
    h: np.ndarray
    diagnostics: dict

    def predict(self): return predict_erp(self.A,self.D,self.h)

    def features(self,x):
        z=modes(x)
        # Weighted ridge projection matches the equal-window fitting metric.
        inv=np.linalg.inv(self.h.T@(WEIGHTS[:,None]*self.h)+.1*np.eye(self.rank))
        return np.einsum('nmt,tr,rs->nms',z,WEIGHTS[:,None]*self.h,inv).reshape(len(x),-1)


def profile_grid(erp, duration, rank, lam, tau_g=20., quick=False, direction=True):
    fac,theta,h,g,inv=bank(int(round(duration*FS)),rank,tau_g,quick)
    avg,diff=target_components(erp)
    pa=np.einsum('mt,t,ktr->kmr',avg,WEIGHTS,h,optimize=True)
    pd=np.einsum('mt,t,ktr->kmr',diff,WEIGHTS,h,optimize=True)
    ia,id_=inv[lam]
    A=np.einsum('kmr,krs->kms',pa,ia)
    D=np.einsum('kmr,krs->kms',pd,id_) if direction else np.zeros_like(A)
    # Variable-projection objective, including the ridge penalty, up to a constant.
    scores=-2*(np.einsum('kmr,kmr->k',A,pa)+np.einsum('kmr,kmr->k',D,pd))
    return fac,theta,h,g,A,D,scores


def fit_erp(erp,duration,rank=3,lam=.1,tau_g=20.,quick=False,refine=True,direction=True):
    fac,theta,h,g,A,D,scores=profile_grid(erp,duration,rank,lam,tau_g,quick,direction)
    idx=int(np.argmin(scores)); best_theta=theta[idx].copy(); best_h=h[idx]; ba=A[idx];bd=D[idx]
    def objective(v, return_fit=False):
        if not allowed(v,rank): return 1e15 if not return_fit else None
        hv=fac.kernel(v,rank); av,dv=coefficients(erp,hv,lam,direction)
        val=6*weighted_error(erp,predict_erp(av,dv,hv))+lam*np.sum(av**2)+10*lam*np.sum(dv**2)
        return (val,hv,av,dv) if return_fit else val
    initial=objective(best_theta); best=initial; traces=[]
    if refine:
        # Multiple deterministic, distinct good initial points, not random re-use of fits.
        starts=[]
        for j in np.argsort(scores):
            v=theta[j]
            if not starts or min(np.linalg.norm((v-s)/np.array([b-a for a,b in BOUNDS[:rank+1]])) for s in starts)>.18:
                starts.append(v)
            if len(starts)>=(2 if quick else 3):break
        cons=[{'type':'ineq','fun':lambda v:v[2]-v[1]-10}]
        if rank==3:cons.append({'type':'ineq','fun':lambda v:v[3]-v[2]-10})
        scale=max(initial,1.)
        lows=np.array([a for a,b in BOUNDS[:rank+1]])
        spans=np.array([b-a for a,b in BOUNDS[:rank+1]])
        ucons=[{'type':'ineq','fun':lambda u: (lows+u*spans)[2]-(lows+u*spans)[1]-10}]
        if rank==3:ucons.append({'type':'ineq','fun':lambda u:(lows+u*spans)[3]-(lows+u*spans)[2]-10})
        for start in starts:
            result=minimize(lambda u:objective(lows+u*spans)/scale,(start-lows)/spans,method='SLSQP',
                            bounds=[(0,1)]*(rank+1),constraints=ucons,
                            options={'maxiter':35 if quick else 65,'ftol':1e-8,'eps':1e-4})
            result.x=lows+result.x*spans
            value=objective(result.x)
            traces.append(dict(theta=result.x.tolist(),objective=value,success=bool(result.success),
                               message=str(result.message),nfev=int(result.nfev)))
            if result.success and np.isfinite(value) and value<best:
                best_theta=result.x.copy();best=value
        best,best_h,ba,bd=objective(best_theta,True)
    condition=float(np.linalg.cond(best_h[FIT_MASK]*np.sqrt(WEIGHTS[FIT_MASK,None])))
    hits=[i for i,(a,b) in enumerate(BOUNDS[:rank+1]) if min(abs(best_theta[i]-a),abs(best_theta[i]-b))<.5]
    return Fit(rank,lam,best_theta,ba,bd,best_h,dict(objective=best,grid_objective=initial,
        condition=condition,bound_hits=hits,starts=traces,refinement_success=any(t['success'] for t in traces),
        numerical_status='grid_optimum' if not refine else ('refined' if best<initial else 'grid_retained'),
        duration_ms=duration*1000,tau_g_ms=tau_g))


def train_model(ds,quick=False,rank_constraint=None,tau_g=20.,direction=True):
    """Inner leave-block selection; outer/test data are absent from this interface."""
    blocks=np.unique(ds.blocks); rows=[]; candidates=[]
    ranks=(rank_constraint,) if rank_constraint else (2,3)
    if len(blocks)<2: raise ValueError('At least two training blocks required')
    for rank in ranks:
        for lam in LAMBDAS:
            losses=[]
            for block in blocks:
                tr=ds.subset(ds.blocks!=block);va=ds.subset(ds.blocks==block)
                erp=means(tr.x,tr.y); obs=means(va.x,va.y)
                f=fit_erp(erp,np.median(tr.durations),rank,lam,tau_g,quick,refine=False,direction=direction)
                loss=weighted_error(obs,f.predict());losses.append(loss)
                rows.append(dict(rank=rank,lambda_A=lam,validation_block=int(block),loss=loss,
                                 train_ids=tr.ids.tolist(),validation_ids=va.ids.tolist()))
            candidates.append((np.mean(losses),rank,lam,float(np.std(losses,ddof=1)/np.sqrt(len(losses)))))
    # Fixed 1%-relative tolerance for near ties; otherwise lowest inner validation loss.
    minimum=min(v[0] for v in candidates)
    eligible=[v for v in candidates if v[0]<=minimum+max(1e-10,.01*abs(minimum))]
    chosen=min(eligible,key=lambda v:(v[1],-v[2],v[0]))
    f=fit_erp(means(ds.x,ds.y),np.median(ds.durations),chosen[1],chosen[2],tau_g,quick,True,direction)
    f.diagnostics.update(inner_validation=rows,selection_candidates=candidates)
    return f


def waveform_metrics(obs,pred):
    rows=[]
    for d,cue in enumerate((-1,1)):
        for c,name in enumerate(('Fz','F3','F4')):
            y=obs[d,c,FIT_MASK];p=pred[d,c,FIT_MASK];den=np.sum((y-y.mean())**2)
            peaks,_=find_peaks(pred[d,c]);peaks=peaks[(TIMES[peaks]>=.25)&(TIMES[peaks]<=.5)&(pred[d,c,peaks]>0)]
            peak=int(peaks[np.argmax(pred[d,c,peaks])]) if len(peaks) else None
            rows.append(dict(cue=cue,channel=name,RMSE=float(np.sqrt(np.mean((y-p)**2))),
                R2=float(1-np.sum((y-p)**2)/den) if den>1e-12 else np.nan,
                positive_peak_latency_ms=float(TIMES[peak]*1000) if peak is not None else np.nan,
                positive_peak_amplitude=float(pred[d,c,peak]) if peak is not None else np.nan))
    delta=obs[1]-obs[0];dp=pred[1]-pred[0]
    energy=float(np.sum(delta[:,FIT_MASK]**2));err=float(np.sum((delta-dp)[:,FIT_MASK]**2))
    return rows,dict(contrast_RMSE=float(np.sqrt(err/(3*FIT_MASK.sum()))),
        S_delta=float(1-err/energy) if energy>1e-10 else np.nan,delta_energy=energy,
        weighted_MSE=weighted_error(obs,pred))
