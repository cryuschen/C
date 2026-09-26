"""First-passage decisions driven exclusively by V2 cognition states."""
from dataclasses import replace
import numpy as np
from .common import FS, TIME
from .model import Candidate, states

REFERENCE=Candidate('N2',tau_m=2.,tau_p=.3,target_duration=.2)
EVENT=dict(cue=1,cue_duration_s=52/FS,target_s=568/FS)
SCENARIOS=(
    ('nominal','基准',REFERENCE,1.5,.6,2.),
    ('weak_memory','短保持',replace(REFERENCE,tau_m=.5),1.5,.6,2.),
    ('retrieval_off','关闭提取',replace(REFERENCE,model='N2_no_retrieval'),1.5,.6,2.),
    ('high_noise','高噪声',REFERENCE,1.5,1.3,2.),
    ('low_evidence','低证据增益',REFERENCE,.15,.6,2.),
    ('short_deadline','短仿真截止',REFERENCE,1.5,.6,1.2),
)


def aligned_evidence(candidate,event=EVENT,deadline=2.,dt=1/FS):
    t=np.arange(int(np.floor(deadline/dt)))*dt
    if not len(t) or event['target_s']+deadline>TIME[-1]:
        raise ValueError('Decision horizon exceeds neural simulation')
    _,_,p=states(event,candidate)
    return t,event['cue']*np.interp(event['target_s']+t,TIME,p[:,1])


def evidence_normalization():
    _,a=aligned_evidence(REFERENCE)
    return max(float(np.max(abs(a))),1e-10)


def target_match(cues,layouts):
    """Signed right-minus-left similarity; no correctness labels are accepted.

    Shapes are encoded by the same Q2 image encoder. This extension represents
    simulated Task-2 layouts only; real per-trial layouts remain unavailable.
    """
    from .model import shape_inputs
    features=shape_inputs()[:,:2]
    features=features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-10)
    cues=np.asarray(cues);layouts=np.asarray(layouts)
    if layouts.shape!=(len(cues),2) or not np.isin(layouts,[-1,1]).all():
        raise ValueError('Supply left/right target shapes for every simulated trial')
    memory=features[(cues==1).astype(int)]
    targets=features[(layouts==1).astype(int)]
    similarities=np.einsum('nd,nkd->nk',memory,targets)
    scale=max(1-float(features[0]@features[1]),1e-10)
    return (similarities[:,1]-similarities[:,0])/scale


def simulate(candidate=REFERENCE,gamma=1.5,sigma=.6,deadline=2.,n=6000,
             seed=202609263,dt=1/FS,boundary=1.,event=EVENT,evidence_mode='conditional'):
    if n%4 or n<4 or sigma<0 or boundary<=0:
        raise ValueError('Use balanced four-way simulation and valid noise/boundary')
    time,evidence=aligned_evidence(candidate,event,deadline,dt)
    horizon=len(time)*dt
    drift=gamma*evidence/evidence_normalization()
    # Ground-truth layout exists ONLY here. Cue shape and target side are distinct.
    cue=np.tile([-1,-1,1,1],n//4);correct_side=np.tile([-1,1,-1,1],n//4)
    layouts=np.column_stack([np.tile([-1,1,-1,1],n//4),np.tile([1,-1,1,-1],n//4)])
    if evidence_mode=='matching':
        direction=target_match(cue,layouts)
        correct_side=np.where(layouts[:,1]==cue,1,-1)  # scoring only
    elif evidence_mode=='conditional':direction=correct_side.astype(float)
    else:raise ValueError('Unknown evidence mode')
    rng=np.random.default_rng(seed);z=np.zeros(n);choice=np.zeros(n,int)
    decision=np.full(n,np.nan);history=np.zeros((n,len(time)),np.float32)
    for j in range(len(time)):
        noise=rng.standard_normal(n)  # common random numbers across scenarios
        active=choice==0
        z[active]+=direction[active]*drift[j]*dt+sigma*np.sqrt(dt)*noise[active]
        hit=active&(abs(z)>=boundary)
        choice[hit]=np.sign(z[hit]).astype(int);decision[hit]=(j+1)*dt
        history[:,j]=z
    outcome=np.where(choice==0,'unresolved',np.where(choice==correct_side,'correct','error'))
    rows=[dict(simulation_id=i+1,cue=int(cue[i]),correct_side=int(correct_side[i]),
          target_left_shape=int(layouts[i,0]) if evidence_mode=='matching' else None,
          target_right_shape=int(layouts[i,1]) if evidence_mode=='matching' else None,
          signed_match=float(direction[i]),evidence_mode=evidence_mode,
          choice=int(choice[i]),outcome=outcome[i],decision_time_s=float(decision[i]),
          observation_time_s=float(decision[i]) if choice[i] else horizon,
          event_observed=int(choice[i]!=0),right_censored=bool(choice[i]==0)) for i in range(n)]
    observed_times=np.where(choice!=0,decision,horizon)
    resolved=np.sort(decision[np.isfinite(decision)])
    median_index=int(np.ceil(n/2))-1
    survival_median=float(resolved[median_index]) if len(resolved)>median_index else None
    summary=dict(n=n,evidence_mode=evidence_mode,tau_m=candidate.tau_m,tau_p=candidate.tau_p,
                 neural_model=candidate.model,target_duration=candidate.target_duration,
                 gamma=gamma,sigma=sigma,boundary=boundary,deadline_s=horizon,dt=dt,
                 correct_rate=float(np.mean(outcome=='correct')),error_rate=float(np.mean(outcome=='error')),
                 unresolved_rate=float(np.mean(outcome=='unresolved')),
                 median_decision_time_s=float(np.nanmedian(decision)) if np.isfinite(decision).any() else None,
                 conditional_median_scope='resolved_trials_only',
                 censor_aware_median_s=survival_median,
                 restricted_mean_decision_time_s=float(observed_times.mean()),
                 interpretation='simulation_only_no_observed_accuracy_fit')
    for name in ('correct','error','unresolved'):
        p=summary[name+'_rate'];k=p*n;zz=1.96**2
        center=(p+zz/(2*n))/(1+zz/n);half=1.96*np.sqrt(p*(1-p)/n+zz/(4*n*n))/(1+zz/n)
        summary[name+'_mc_low']=float(center-half);summary[name+'_mc_high']=float(center+half)
    examples=[]
    for name in ('correct','error','unresolved'):
        examples.extend(np.flatnonzero(outcome==name)[:3])
    paths=dict(time=time+dt,drift=drift,evidence=evidence,trajectories=history[examples],
               outcome=outcome[examples],correct_side=correct_side[examples],ids=np.array(examples)+1)
    # Competing outcomes reconcile at every time, without assigning RT to censored rows.
    curves=[]
    for t in np.r_[0.,time+dt]:
        c=float(np.mean((outcome=='correct')&(decision<=t)))
        e=float(np.mean((outcome=='error')&(decision<=t)))
        curves.append(dict(time_s=float(t),correct_cdf=c,error_cdf=e,unresolved_survival=1-c-e))
    return summary,rows,paths,curves
