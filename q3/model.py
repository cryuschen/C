"""Q2 E/I visual pathway plus stable, delayed memory/cognition mean states."""
from dataclasses import dataclass, asdict
from itertools import product
from functools import lru_cache
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root
from scipy.special import expit
from scipy.signal import sosfilt, sosfiltfilt
from q2.q2model.generative import encode_shapes
from .data import ROOT, FS, TIME, filter_sos

MODELS=('M0','M1','M2','M3','M2_no_feedback','M2_no_direct_memory')


@dataclass(frozen=True)
class Candidate:
    model: str
    tau_m: float=1.
    tau_p: float=.3
    delay: float=.05
    feedback: float=0.
    beta: float=1.
    trend_tau: float=1.

    def record(self): return asdict(self)


def candidates(task, quick=False):
    tm=(.5,1.,2.,4.) if not quick else (1.,2.)
    tp=(.1,.3,.6) if not quick else (.3,)
    delta=(0.,.05,.1) if not quick else (.05,)
    feedback=(0.,.1,.3) if not quick else (0.,.1)
    betas=(.5,1.,2.) if task==2 and not quick else (1.,)
    out=[Candidate('M0')]
    out += [Candidate('M1',tau_p=p) for p in tp]
    out += [Candidate('M3',trend_tau=s) for s in (.3,1.,2.)]
    for m,p,d,k,b in product(tm,tp,delta,feedback,betas):
        out.append(Candidate('M2',m,p,d,k,b))
        if k==0: out.append(Candidate('M2_no_feedback',m,p,d,k,b))
        out.append(Candidate('M2_no_direct_memory',m,p,d,k,b))
    return out


@lru_cache(maxsize=1)
def shape_inputs():
    return encode_shapes(ROOT)['inputs'][:2]


@lru_cache(maxsize=128)
def visual_response(cue, duration_samples, target_samples):
    """Same 57-state equations as Q2, now two actual-onset input pulses.

    The second pulse is a fixed 200-ms common input, NOT target sign/response.
    Q2 local scale=1 and recurrence=.1 are fixed assumptions, not refitted.
    """
    te=np.array([.02,.04,.08])[:,None];ti=2*te;se=np.array([.025,.05,.1])[:,None];si=3*se
    W=(np.ones((3,3))-np.eye(3))/2
    eq=root(lambda z:np.r_[-z[:3]+expit(z[:3]-z[3:]+.1*W@z[:3]-2),
                            -z[3:]+expit(z[:3]-z[3:]-2)],np.full(6,.12))
    if not eq.success: raise RuntimeError('Q2 rest state failed')
    er=np.tile(eq.x[:3],(3,1));ir=np.tile(eq.x[3:],(3,1))
    initial=np.r_[np.zeros(3),er.ravel(),ir.ravel(),np.zeros(36)]
    def rhs(state,drive):
        g=state[:3];e=state[3:12].reshape(3,3);i=state[12:21].reshape(3,3)
        ue,ve,ui,vi=state[21:].reshape(4,3,3)
        ff=np.vstack([g,3*(e[:-1]-er[:-1])])
        return np.r_[(-g+drive)/.02,
          ((-e+expit(e-i+.1*(e@W.T)+ff-2))/te).ravel(),
          ((-i+expit(e-i-2))/ti).ravel(),
          ((e-er-ue)/se).ravel(),((ue-ve)/se).ravel(),
          ((i-ir-ui)/si).ravel(),((ui-vi)/si).ravel()]
    duration=duration_samples/FS;target=target_samples/FS
    breaks=sorted(set([0.,duration,target,target+.2,float(TIME[-1])]))
    states=np.tile(initial[:,None],(1,len(TIME)));state=initial.copy()
    for start,stop in zip(breaks[:-1],breaks[1:]):
        if stop<=start:continue
        mid=(start+stop)/2
        drive=3*shape_inputs()[0 if cue==-1 else 1] if mid<duration else np.zeros(3)
        if target<=mid<target+.2:drive=drive+np.array([0.,0.,3.])
        sol=solve_ivp(lambda t,s:rhs(s,drive),(start,stop),state,dense_output=True,
                       max_step=1/FS,rtol=1e-6,atol=1e-8)
        if not sol.success:raise RuntimeError(sol.message)
        idx=np.flatnonzero((TIME>=start)&(TIME<=stop))
        states[:,idx]=sol.sol(TIME[idx]);state=sol.y[:,-1]
    syn=states[21:].reshape(4,3,3,len(TIME))
    # Three shape/common groups at the final visual stage, centered at rest.
    return (syn[1,2]-.7*syn[3,2]).T


def stimulus_key(event):
    return (int(event['cue']),int(round(event['cue_duration_s']*FS)),int(round(event['target_s']*FS)))


def slow_states(v, params, target, times=TIME):
    """Vectorized exponential Euler, fractional-delay interpolation; no future values."""
    n=len(params);m=np.zeros((n,len(times),3));p=np.zeros_like(m)
    dt=float(times[1]-times[0])
    tm=np.array([c.tau_m for c in params]);tp=np.array([c.tau_p for c in params])
    k=np.array([c.feedback for c in params]);b=np.array([c.beta for c in params])
    if np.any(k*b>=1):raise ValueError('Feedback violates sufficient delay-independent stability bound')
    delay=np.array([c.delay for c in params])/dt;whole=np.floor(delay).astype(int);frac=delay-whole
    am=np.exp(-dt/tm)[:,None];ap=np.exp(-dt/tp)[:,None];ix=np.arange(n)
    for t in range(1,len(times)):
        past=t-1-whole
        md=m[ix,np.maximum(past,0)]*(1-frac[:,None])+m[ix,np.maximum(past-1,0)]*frac[:,None]
        pd=p[ix,np.maximum(past,0)]*(1-frac[:,None])+p[ix,np.maximum(past-1,0)]*frac[:,None]
        gate=float(times[t-1]>=target)
        m[:,t]=am*m[:,t-1]+(1-am)*(v[t-1]+k[:,None]*pd)
        p[:,t]=ap*p[:,t-1]+(1-ap)*(v[t-1]+b[:,None]*gate*md)
    return m,p


def design_bank(params,event,mode='causal'):
    v=visual_response(*stimulus_key(event));m,p=slow_states(v,params,event['target_s'])
    bank=np.zeros((len(params),len(TIME),9))
    for j,c in enumerate(params):
        bank[j,:,:3]=v
        if c.model=='M1':
            # No hidden memory drive: ordinary low-pass integration of visual response.
            a=np.exp(-1/(FS*c.tau_p));z=np.zeros_like(v)
            for t in range(1,len(TIME)):z[t]=a*z[t-1]+(1-a)*v[t-1]
            bank[j,:,3:6]=z
        elif c.model.startswith('M2'):
            if c.model!='M2_no_direct_memory':bank[j,:,3:6]=m[j]
            bank[j,:,6:9]=p[j]
        elif c.model=='M3':
            tt=np.maximum(TIME,0);after=np.maximum(TIME-event['target_s'],0)
            g=(TIME>=0).astype(float);gt=(TIME>=event['target_s']).astype(float)
            slow=np.stack([g*(1-np.exp(-tt/c.trend_tau)),g*tt/(1+tt),
                           gt*(1-np.exp(-after/c.trend_tau))],axis=1)
            # Common + cue-signed smooth terms; six extra columns, same maximum size as M2.
            bank[j,:,3:6]=slow;bank[j,:,6:9]=slow*event['cue']
    sos=filter_sos()
    if mode=='causal': bank=sosfilt(sos,bank,axis=1)
    else:
        # Zero padding is rest, never a response-conditioned waveform extension.
        pad=np.pad(bank,((0,0),(24*FS,24*FS),(0,0)))
        bank=sosfiltfilt(sos,pad,axis=1)[:,24*FS:24*FS+len(TIME)]
    bank-=np.median(bank[:,TIME<0],axis=1)[:,None,:]
    return bank.astype(np.float32)


def predict(event,candidate,coef,mode='causal'):
    # Only explicitly whitelisted stimulus fields enter the generated design.
    public={k:event[k] for k in ('cue','cue_duration_s','target_s')}
    return design_bank([candidate],public,mode)[0]@coef
