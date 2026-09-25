"""Separated cue/target pathways and an encoding-only memory state.

All target input is common: no clicked side, response time or correctness enters.
Direction shrinkage and target pulse width are selected in inner time blocks.
"""
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.signal import sosfilt
from q2.q2model.generative import simulate
from q3.model import shape_inputs
from q3.data import FS, TIME, filter_sos


@dataclass(frozen=True)
class Candidate:
    model: str
    tau_m: float = 1.
    tau_p: float = .3
    target_duration: float = .2
    direction_penalty: float = 1.
    delay: float = .05

    def record(self):
        return asdict(self)


def candidates():
    ans=[]
    for duration,ratio in product((.1,.2,.4),(1.,10.)):
        ans.append(Candidate('N0',target_duration=duration,direction_penalty=ratio))
        for tp in (.1,.3,.6):
            for model in ('N1','N3'):
                ans.append(Candidate(model,tau_p=tp,target_duration=duration,direction_penalty=ratio))
            for tm in (.5,1.,2.,4.):
                for model in ('N2','N2_no_projection','N2_no_retrieval'):
                    ans.append(Candidate(model,tm,tp,duration,ratio))
    ans.append(Candidate('B0'))
    return ans


def stimulus_key(e):
    return int(e['cue']),int(round(e['cue_duration_s']*FS)),int(round(e['target_s']*FS))


@lru_cache(maxsize=24)
def pulse(duration, common_target=False):
    inputs=np.array([[0.,0.,1.]]) if common_target else shape_inputs()
    result=simulate(inputs,duration=duration,max_step=1/FS)
    # Retain Q2's last visual stage. Its source is sampled on a fixed horizon.
    q=result['q'][:,2]
    return result['t'],q


@lru_cache(maxsize=128)
def visual_parts(key,duration):
    cue,ns,ts=key
    t,q=pulse(ns/FS)
    left=np.stack([np.interp(TIME,t,a,left=0,right=0) for a in q[0]],axis=1)
    right=np.stack([np.interp(TIME,t,a,left=0,right=0) for a in q[1]],axis=1)
    # Common activity and antisymmetric shape contrast have distinct readouts.
    common=(left.mean(1)+right.mean(1))/2
    contrast=(right[:,1]-left[:,1])/2
    tc,qt=pulse(duration,True)
    target=np.interp(TIME-ts/FS,tc,qt[0,2],left=0,right=0)
    return np.stack([common,cue*contrast,target],axis=1)


def lowpass(x,tau):
    # Exponential Euler: state at t uses input at t-1, hence is causal.
    from scipy.signal import lfilter
    a=np.exp(-1/(FS*tau))
    return lfilter([0.,1-a],[1.,-a],x,axis=0)


def states(e,c):
    v=visual_parts(stimulus_key(e),c.target_duration)
    m=lowpass(v[:,:2],c.tau_m)  # Target activity NEVER enters stored cue memory.
    delayed=np.stack([np.interp(TIME-c.delay,TIME,m[:,j],left=0,right=0) for j in range(2)],axis=1)
    gate=(TIME>=e['target_s'])[:,None]
    gain=0. if c.model=='N2_no_retrieval' else 1.
    drive=v.copy()
    if c.model.startswith('N2'):
        drive[:,:2]+=gain*gate*delayed
    p=lowpass(drive,c.tau_p)
    return v,m,p


@lru_cache(maxsize=4096)
def neural_design(key,c):
    e=dict(cue=key[0],cue_duration_s=key[1]/FS,target_s=key[2]/FS)
    v,m,p=states(e,c);h=np.zeros((len(TIME),9));h[:,:3]=v
    if c.model=='B0':
        h[:]=0
    elif c.model=='N1':
        h[:,6:9]=p
    elif c.model.startswith('N2'):
        if c.model!='N2_no_projection':h[:,3:5]=m
        h[:,6:9]=p
    elif c.model=='N3':
        t=np.maximum(TIME,0);a=np.maximum(TIME-e['target_s'],0)
        s=np.stack([(TIME>=0)*(1-np.exp(-t/c.tau_p)),t/(1+t),
                    (TIME>=e['target_s'])*(1-np.exp(-a/c.tau_p))],axis=1)
        h[:,3:6]=s;h[:,6:9]=s*e['cue']
    h=sosfilt(filter_sos(),h,axis=0)
    h-=np.median(h[TIME<0],axis=0)
    return h.astype(np.float32)


def penalties(c,dim):
    p=np.ones(dim)
    directional=[1,6,7,8] if c.model=='N3' else [1,4,7]
    p[directional]=c.direction_penalty
    return p


def background_design(e):
    """Past-only slope continuation with three fixed decay horizons per channel.

    The input is computed BEFORE cue onset; this is a separately labelled
    baseline-conditioned prediction setting, not a stimulus-only prediction.
    """
    t=np.maximum(TIME,0)
    slopes=np.asarray(e['past_slopes'])  # 3 horizons x 3 channels
    response=np.stack([tau*(1-np.exp(-t/tau)) for tau in (.25,1.,4.)],axis=1)
    return (response[:,:,None]*slopes[None,:,:]).reshape(len(TIME),9)


def design_bank(params,event,mode='causal',context=False):
    if mode!='causal':raise ValueError('V2 primary analyses use causal filtering')
    h=np.stack([neural_design(stimulus_key(event),c) for c in params])
    if context:
        bg=np.broadcast_to(background_design(event),(len(params),len(TIME),9))
        h=np.concatenate([h,bg],axis=2)
    return h


def predict(event,candidate,coef,context=False):
    return design_bank([candidate],event,context=context)[0]@coef
