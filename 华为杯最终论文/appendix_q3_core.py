# 问题三：事件审计、认知状态拟合与决策模拟
import numpy as np
from .common import FS, TIME, ROOT, Trials, load_raw, runs, causal_filter, quality


def audit_events(key, data):
    cues, markers = runs(data[7]), runs(data[8])
    rows = []
    for i, (on, off, cue) in enumerate(cues):
        nxt = cues[i+1][0] if i+1 < len(cues) else data.shape[1]
        ev = [(s, e, v) for s, e, v in markers if on <= s < nxt]
        platforms = [(s, e, v) for s, e, v in ev if abs(v) == 1]
        clicks = [(s, e, v) for s, e, v in ev if abs(v) == 2]
        ts, te, sign = platforms[0] if len(platforms) == 1 else (None, None, None)
        target_valid = ts is not None and on < ts < nxt
        marker_valid = len(clicks) <= 1
        click = clicks[0][0] if len(clicks) == 1 else None
        if click is not None and (not target_valid or not ts < click < nxt):
            marker_valid = False
        response, status = None, 'unobserved'
        if marker_valid and click is not None:
            response, status = click, 'explicit_click'
        elif marker_valid and key.endswith('1') and target_valid and ts < te < nxt:
            response, status = te, 'platform_end_proxy'
        row = dict(dataset=key, trial_id=i+1, block=i//20, cue=int(cue),
            cue_sample=on, cue_end_sample=off, cue_duration_s=(off-on)/FS,
            next_cue_sample=nxt, record_stop_sample=data.shape[1],
            target_sample=ts, target_s=(ts-on)/FS if target_valid else None,
            target_status='platform_onset_hypothesis' if target_valid else 'unknown',
            platform_end_sample=te, platform_sign=sign,
            click_sample=click, click_side=int(np.sign(clicks[0][2])) if len(clicks)==1 else None,
            response_sample=response, response_s=(response-on)/FS if response is not None else None,
            rt_s=(response-ts)/FS if response is not None else None, response_status=status,
            correctness=None, correctness_reason='target_ground_truth_unavailable',
            timeout=None, timeout_reason='experimental_deadline_unavailable',
            n_platforms=len(platforms), n_clicks=len(clicks),
            cue_platform_disagree=bool(sign!=cue) if sign is not None else None,
            valid_events=bool(target_valid and marker_valid),
            event_issue='none' if target_valid and marker_valid else 'ambiguous_or_invalid_markers')
        rows.append(row)
    return rows


def window_policy(event, horizon=2., gap=.1):
    """All correctness values are eligible. Administrative cutoff != timeout.

    Missing-marker records have a finite EEG window but are not automatically
    valid right-censored behavior: ascertainment of the response is unknown.
    """
    if horizon <= 0 or gap < 0:
        raise ValueError('Invalid window policy')
    e = dict(event)
    e.update(horizon_s=float(horizon), gap_s=gap, retained=False,
             end_s=None, behavior_duration_s=None, behavior_event=None,
             behavior_survival_eligible=False, eeg_administrative_cutoff=False)
    if not e['valid_events']:
        e['window_status']='unusable_events'
        return e
    target = e['target_s']
    administrative = target+horizon
    available = min((e['next_cue_sample']-e['cue_sample'])/FS,
                    (e['record_stop_sample']-e['cue_sample'])/FS, float(TIME[-1]))
    bound = min(administrative, available)
    response = e['response_s']
    # Gap is attached to an actual/proxy response, not subtracted from censoring.
    e['end_s'] = min(bound, response-gap) if response is not None else bound
    e['eeg_administrative_cutoff'] = response is None or bound < response-gap
    e['window_status'] = ('response_minus_gap' if not e['eeg_administrative_cutoff']
                          else 'bounded_observation')
    e['bound_reason'] = 'configured_horizon' if administrative <= available else 'record_or_next_trial'
    duration = max(0., bound-target)
    if e['response_status']=='explicit_click':
        e['behavior_survival_eligible']=duration>0
        e['behavior_event']=int(response <= bound)
        e['behavior_duration_s']=min(response-target,duration)
        e['behavior_status']='observed_click' if e['behavior_event'] else 'administratively_right_censored'
    elif e['response_status']=='platform_end_proxy':
        e['behavior_status']='proxy_only_not_observed_click'
    else:
        e['behavior_status']='unknown_response_or_missing_marker'
    if e['end_s'] <= target:
        e['window_status']='no_posttarget_observation'
    return e
def build_trials(key, horizon=2., gap=.1, raw=None, root=ROOT):
    data = load_raw(key, root)[0] if raw is None else raw
    audited = audit_events(key,data)
    ons=[e['cue_sample'] for e in audited]
    edges=[0]+[(ons[j-1]+ons[j])//2 for j in (20,40,60,80)]+[data.shape[1]]
    pieces={b:causal_filter(data[:3,a:z]) for b,(a,z) in enumerate(zip(edges[:-1],edges[1:]))}
    kept=[];arrays=[];scales=[];rows=[]
    for event in audited:
        e=window_policy(event,horizon,gap);b=e['block'];on=e['cue_sample'];a,z=edges[b:b+2]
        e.update(filter_start_sample=a,filter_stop_sample=z,filter_mode='causal',reason='invalid_events')
        if e['end_s'] is not None and e['end_s']>e['target_s']:
            end=on+int(np.ceil(e['end_s']*FS));length=end-(on-64)
            e.update(epoch_start_sample=on-64,epoch_end_exclusive=end)
            if on-64<a+24*FS or end>z-24*FS:
                e['reason']='block_boundary_24s_buffer'
            else:
                epoch=pieces[b][:,on-64-a:end-a].copy()
                epoch-=np.median(epoch[:,:64],axis=1,keepdims=True)
                bad,measures=quality(epoch,data[:3,on-64:end],np.arange(length)>=64)
                e.update(measures,retained=not bad,reason='quality' if bad else 'retained')
                if not bad:
                    x=np.full((len(TIME),3),np.nan);x[:length]=epoch.T
                    arrays.append(x);kept.append(e)
                    scales.append(np.maximum(1.4826*np.median(abs(epoch[:,:64]),axis=1),1.))
        rows.append(e)
    if not kept: raise ValueError(f'No valid EEG trials for {key}')
    return Trials(key,kept,np.stack(arrays),np.stack(scales),'causal',gap),rows
# ---- 记忆与认知状态 ----
from dataclasses import dataclass, asdict
from functools import lru_cache
from itertools import product
import numpy as np
from scipy.signal import sosfilt
from q2.q2model.generative import simulate, encode_shapes
from .common import FS, TIME, ROOT, filter_sos

@lru_cache(maxsize=1)
def shape_inputs():
    return encode_shapes(ROOT)['inputs'][:2]


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
# ---- 分块选择与稳健拟合 ----
import numpy as np
from scipy.linalg import solve
from .model import design_bank, penalties

LAMBDAS=np.array([.001,.01,.1,1.,10.])


class Bank:
    def __init__(self,ds,params,context=False,design_function=None):
        self.ds=ds;self.params=params;self.context=context;self.dim=18 if context else 9
        self.design=design_bank if design_function is None else design_function
        self.fit_cache={};self.robust_cache={};self.weights=ds.weights
        self.penalty=np.stack([penalties(c,self.dim) for c in params])
        n=len(params);d=self.dim
        self.G=np.zeros((5,n,d,d));self.B=np.zeros((5,n,d,3));self.Y=np.zeros((5,3));self.N=np.zeros(5)
        # A design can include pre-cue context, so sufficient statistics are
        # accumulated per trial (never grouped while dropping its context).
        for i,e in enumerate(ds.events):
            block=e['block'];h=self.design(params,e,context=context).astype(float)
            w=self.weights[i];x=np.nan_to_num(ds.x[i])
            self.G[block]+=np.matmul(h.transpose(0,2,1),h*w[None,:,None])
            self.B[block]+=np.matmul(h.transpose(0,2,1),w[:,None]*x)
            self.Y[block]+=np.sum(w[:,None]*x*x,axis=0)
            self.N[block]+=1


    def ridge_all(self,blocks):
        ix=list(blocks);count=self.N[ix].sum()
        if count<=0:raise ValueError('Empty training blocks')
        g=self.G[ix].sum(0)/count;b=self.B[ix].sum(0)/count
        scale=np.maximum(np.sqrt(np.diagonal(g,axis1=1,axis2=2)),1e-8)
        gn=g/scale[:,:,None]/scale[:,None,:];bn=b/scale[:,:,None]
        systems=gn[None]+LAMBDAS[:,None,None,None]*np.eye(self.dim)[None,None]*self.penalty[None,:,None,:]
        coef=np.linalg.solve(systems,np.broadcast_to(bn,systems.shape[:2]+(self.dim,3)))
        return coef/scale[None,:,:,None],scale,np.maximum(self.Y[ix].sum(0)/count,1.)

    def validation_errors(self,coef,block,channel_var):
        # lambda, candidate, channel
        err=(np.einsum('lkpc,kpq,lkqc->lkc',coef,self.G[block],coef,optimize=True)
             -2*np.einsum('lkpc,kpc->lkc',coef,self.B[block],optimize=True)+self.Y[block])
        return np.mean(err/channel_var[None,None,:],axis=-1)/self.N[block]

    def matrix(self,indices,candidate_index):
        hs=[];ys=[];ws=[]
        c=self.params[candidate_index]
        for i in indices:
            h=self.design([c],self.ds.events[i],context=self.context)[0].astype(float)
            mask=self.weights[i]>0
            hs.append(h[mask]);ys.append(self.ds.x[i,mask]);ws.append(self.weights[i,mask])
        return np.concatenate(hs),np.concatenate(ys),np.concatenate(ws)/len(indices)

    def robust_fit(self,blocks,ci,lam,iterations=7):
        cache_key=(tuple(sorted(blocks)),ci,lam,iterations)
        if cache_key in self.robust_cache:return self.robust_cache[cache_key]
        idx=np.flatnonzero(np.isin(self.ds.blocks,list(blocks)))
        h,y,w=self.matrix(idx,ci)
        source_scale=np.maximum(np.sqrt(np.sum(w[:,None]*h*h,axis=0)),1e-8)
        channel_scale=np.maximum(np.sqrt(np.sum(w[:,None]*y*y,axis=0)),1.)
        z=h/source_scale;target=y/channel_scale
        gram=z.T@(w[:,None]*z)
        co=solve(gram+lam*np.diag(self.penalty[ci]),z.T@(w[:,None]*target),assume_a='pos')
        # Huber M-estimation on trial/stage-balanced, channel-normalized residuals.
        convergence=[]
        for _ in range(iterations):
            residual=target-z@co;robust=np.minimum(1.,1.5/np.maximum(abs(residual),1e-12))
            new=np.empty_like(co)
            for channel in range(3):
                wc=w*robust[:,channel]
                new[:,channel]=solve(z.T@(wc[:,None]*z)+lam*np.diag(self.penalty[ci]),z.T@(wc*target[:,channel]),assume_a='pos')
            change=float(np.max(abs(new-co)));convergence.append(change);co=new
            if change<1e-6:break
        result=co/source_scale[:,None]*channel_scale[None,:]
        edf=float(np.trace(np.linalg.solve(gram+lam*np.diag(self.penalty[ci]),gram)))
        singular=np.linalg.svd(np.sqrt(w[:,None])*z,compute_uv=False)
        answer=result,dict(source_scale=source_scale,channel_scale=channel_scale,edf=edf,
                          design_singular_values=singular,irls_changes=convergence)
        self.robust_cache[cache_key]=answer
        return answer
# ---- 首达边界决策过程 ----
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

