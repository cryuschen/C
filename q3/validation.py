"""Paired-block uncertainty, semi-synthetic identifiability, decision simulation."""
import numpy as np
import pandas as pd
from .data import SEED, TIME, FS, Trials, STAGES, stage_masks
from .model import Candidate, design_bank, visual_response, slow_states
from .fit import Bank, LAMBDAS


def metric_summary(rows):
    frame=pd.DataFrame(rows)
    group=frame.groupby(['dataset','model','stage'],sort=False)
    summary=group[['MSE','zero_MSE','baseline_MSE','normalized_MSE']].mean().reset_index()
    summary['RMSE']=np.sqrt(summary.MSE);summary['NRMSE']=np.sqrt(summary.normalized_MSE)
    summary['S_zero']=1-summary.MSE/summary.zero_MSE
    summary['S_train_baseline']=1-summary.MSE/summary.baseline_MSE
    summary['n_trials']=group.trial_id.nunique().values
    return summary


def cognitive_intervals(rows,n_boot=2000):
    rng=np.random.default_rng(SEED+2);df=pd.DataFrame(rows);results=[]
    # Average channels and trials within blocks. Five blocks/record, not N independent people.
    block=df.groupby(['dataset','block','model','stage']).MSE.mean().unstack('model')
    for (dataset,stage),part in block.groupby(level=['dataset','stage']):
        for control in ('M0','M1','M3','M2_no_feedback','M2_no_direct_memory'):
            if control not in part or 'M2' not in part:continue
            a=part.M2.to_numpy();b=part[control].to_numpy()
            idx=rng.integers(0,len(a),(n_boot,len(a)))
            bootstrap=1-a[idx].mean(1)/b[idx].mean(1)
            lo,hi=np.quantile(bootstrap,[.025,.975])
            results.append(dict(dataset=dataset,stage=stage,control=control,
                S_cog=float(1-a.mean()/b.mean()),ci_low=float(lo),ci_high=float(hi),
                n_blocks=len(a),n_bootstraps=n_boot))
    return results


def contrast_metrics(ds,predictions):
    rows=[]
    for held in range(5):
        for stage,name in enumerate(STAGES):
            left=np.flatnonzero((ds.blocks==held)&(ds.cues==-1));right=np.flatnonzero((ds.blocks==held)&(ds.cues==1))
            if not len(left) or not len(right):continue
            # Pointwise group means on identical observed/predicted trial masks.
            masks=np.stack([stage_masks(e['target_s'],e['end_s'])[stage] for e in ds.events])
            countL=masks[left].sum(0);countR=masks[right].sum(0)
            valid=(countL>=2)&(countR>=2)
            if not valid.any():continue
            def average(a,ix):
                return np.nansum(np.where(masks[ix,:,None],a[ix],np.nan),axis=0)/np.maximum(masks[ix].sum(0),1)[:,None]
            obs=average(ds.x,right)-average(ds.x,left)
            for model,pred in predictions.items():
                pdiff=average(pred,right)-average(pred,left)
                mse=float(np.mean((obs[valid]-pdiff[valid])**2));zero=float(np.mean(obs[valid]**2))
                rows.append(dict(dataset=ds.key,block=held,model=model,stage=name,MSE=mse,zero_MSE=zero,
                                 S_delta=1-mse/zero if zero>0 else np.nan,n_timepoints=int(valid.sum())))
    return rows


def recovery_experiment(ds,params,quick=False,progress=None):
    """Known generator + real residual backgrounds; do not mistake this for source truth."""
    rng=np.random.default_rng(SEED+3)
    true=Candidate('M2',1.,.3,.05,.1,1.)
    coef=rng.normal(size=(9,3));h0=design_bank([true],ds.events[0])[0]
    norms=np.maximum(np.sqrt(np.mean(h0[TIME>=0]**2,axis=0)),1e-6)
    coef=coef/norms[:,None]*5
    signal=np.stack([design_bank([true],e)[0]@coef for e in ds.events])
    mask=np.isfinite(ds.x);signal[~mask]=np.nan
    # Permute real trial backgrounds and zero outside the destination window.
    raw=np.nan_to_num(ds.x);raw-=raw.mean(axis=0,keepdims=True)
    background=raw[rng.permutation(len(raw))]
    sig_rms=np.sqrt(np.nanmean(signal**2));background*=sig_rms/max(np.sqrt(np.mean(background**2)),1e-9)
    rows=[];profiles=[]
    for level in (0.,.5,1.):
        x=signal+level*background;x[~mask]=np.nan
        synthetic=Trials(ds.key,ds.events,x,ds.baseline_scale,'causal',ds.gap)
        bank=Bank(synthetic,params)
        fit=bank.select((0,1,2,3),'M2');c=fit['candidate']
        test=ds.blocks==4;pred=np.stack([design_bank([c],e)[0]@fit['coef'] for e in ds.events])
        rmse=float(np.sqrt(np.nanmean(np.where(test[:,None,None],(signal-pred)**2,np.nan))))
        coef_all,_,_=bank.ridge_all((0,1,2,3))
        score=bank.validation_errors(coef_all,4,np.maximum(np.nanmean(signal[test]**2,axis=(0,1)),1.))
        ci=[i for i,p in enumerate(params) if p.model=='M2'];best_by_candidate=score[:,ci].min(0)
        order=np.argsort(best_by_candidate);minimum=float(best_by_candidate[order[0]])
        # Profile is diagnostic on the synthetic holdout, NOT another selection step.
        for local in order[:10]:
            j=ci[local];profiles.append(dict(noise_level=level,**params[j].record(),test_normalized_MSE=float(best_by_candidate[local])))
        close=int(np.sum(best_by_candidate<=minimum+max(.01*abs(minimum),1e-5)))
        rows.append(dict(noise_level=level,true_parameters=true.record(),selected=c.record(),
            signal_RMSE=rmse,signal_RMS=float(sig_rms),NRMSE=rmse/sig_rms,
            exact_dynamic_recovery=c==true,near_equivalent_candidates=close,
            design_singular_values=fit['meta']['design_singular_values'],
            caveat='backgrounds_are_real_EEG_not_clean_ground_truth'))
        if progress:progress(f'semi-synthetic recovery: noise {level:g} complete')
    return rows,profiles


def decision_simulation(n=5000):
    rng=np.random.default_rng(SEED+4);target=568/FS
    v=visual_response(1,52,568);time=np.arange(0,2.,1/FS)
    nominal=Candidate('M2',2.,.3,.05,.1,1.)
    _,pn=slow_states(v,[nominal],target)
    # Semantic right-minus-left evidence; true mapping exists ONLY in simulation.
    reference=np.interp(target+time,TIME,pn[0,:,1]-pn[0,:,0]);normalization=max(np.max(abs(reference)),1e-8)
    scenarios=[('nominal',2.,1.5,.6),('weak_memory',.5,1.5,.6),
               ('high_noise',2.,1.5,1.3),('low_evidence',2.,.15,.4)]
    rows=[];paths={};details=[]
    for name,tm,gamma,sigma in scenarios:
        _,p=slow_states(v,[Candidate('M2',tm,.3,.05,.1,1.)],target)
        drift=gamma*np.interp(target+time,TIME,p[0,:,1]-p[0,:,0])/normalization
        z=np.zeros(n);choice=np.zeros(n,dtype=int);rt=np.full(n,np.nan);history=np.zeros((n,len(time)),dtype=np.float32)
        for j,t in enumerate(time):
            active=choice==0;z[active]+=drift[j]/FS+sigma/np.sqrt(FS)*rng.standard_normal(active.sum())
            reached=active&(abs(z)>=1);choice[reached]=np.sign(z[reached]).astype(int);rt[reached]=t+1/FS
            history[:,j]=z
        examples=[]
        for outcome in (1,-1,0):
            examples.extend(np.flatnonzero(choice==outcome)[:3])
        paths[name]=dict(time=time,drift=drift,trajectories=history[examples],choice=choice[examples])
        rows.append(dict(scenario=name,n=n,tau_m=tm,gamma=gamma,sigma=sigma,boundary=1.,deadline_s=2.,
            correct_rate=float(np.mean(choice==1)),error_rate=float(np.mean(choice==-1)),
            timeout_rate=float(np.mean(choice==0)),median_decision_time_s=float(np.nanmedian(rt)),
            interpretation='simulation_only_not_fitted_behavior'))
        for i in range(n):details.append(dict(scenario=name,simulation_id=i+1,choice=int(choice[i]),
            decision_time_s=float(rt[i]),correct=bool(choice[i]==1),timeout=bool(choice[i]==0)))
    return rows,paths,details
