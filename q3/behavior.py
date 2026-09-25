"""Strict-prefix RT validation with fully nested unsupervised EEG feature fitting."""
import numpy as np
from scipy.stats import spearmanr
from .data import TIME, Trials, prefix_quality_trials
from .model import design_bank, slow_states, visual_response, stimulus_key, candidates
from .fit import Bank, LAMBDAS


def prefix_dataset(key):
    events,x,audit=prefix_quality_trials(key)
    # Only records with an observed post-prefix response can be RT-scored.
    selected=[i for i,e in enumerate(events) if e['response_status']=='explicit_click' and e['response_s']>e['prefix_end_s']]
    chosen=[]
    for i in selected:
        e=events[i].copy();e['end_s']=e['prefix_end_s'];chosen.append(e)
    return Trials(key,chosen,x[selected],np.ones((len(chosen),3)),'causal',.1),audit


def ordinary_features(ds):
    rows=[]
    for e,x in zip(ds.events,ds.x):
        masks=[(TIME>=0)&(TIME<.8),(TIME>=.8)&(TIME<e['target_s']),
               (TIME>=e['target_s'])&(TIME<e['end_s'])]
        rows.append(np.concatenate([x[mask].mean(0) for mask in masks]))
    return np.array(rows)


def state_features(ds,fit):
    """Fit nine bounded-by-ridge source amplitudes using only a fixed EEG prefix.

    The readout/dynamics are frozen from training. No response, RT, full-window
    quality flag, or post-prefix sample enters a feature.
    """
    c=fit['candidate'];coef=fit['coef'];scale=fit['meta']['channel_scale'];rows=[];audit=[]
    for e,x in zip(ds.events,ds.x):
        h=design_bank([c],e)[0].astype(float)
        mask=(TIME>=0)&(TIME<e['end_s'])
        # t, channel, source; sum sources reproduces model prediction.
        contributions=h[mask,:,None]*coef[None,:,:]
        D=(contributions/scale[None,None,:]).transpose(0,2,1).reshape(-1,9)
        y=(x[mask]/scale).ravel();n=len(y)
        norm=np.maximum(np.sqrt(np.mean(D*D,axis=0)),1e-6)
        Z=D/norm
        # Prior a=1; regularization in normalized-source coordinates.
        correction=np.linalg.solve(Z.T@Z/n+np.eye(9),Z.T@(y-D.sum(1))/n)
        amplitude=1+correction/norm
        v=visual_response(*stimulus_key(e));m,p=slow_states(v,[c],e['target_s'])
        last=np.flatnonzero(mask)[-1];previous=max(last-13,0)
        features=np.r_[m[0,last]*amplitude[3:6],p[0,last]*amplitude[6:9],
                       (p[0,last]-p[0,previous])/(TIME[last]-TIME[previous])*amplitude[6:9]]
        rows.append(features)
        audit.append(dict(trial_id=e['trial_id'],prefix_end_s=e['end_s'],amplitudes=amplitude))
    return np.array(rows),audit


def smoother(features,train,test,lam):
    """Linear prediction map including train-only standardization and intercept."""
    x=features[train];z=features[test];mu=x.mean(0);sd=np.maximum(x.std(0),1e-9)
    x=(x-mu)/sd;z=(z-mu)/sd
    return np.ones((len(test),len(train)))/len(train) + z@np.linalg.solve(
        x.T@x/len(train)+lam*np.eye(x.shape[1]),x.T/len(train))


def build_prediction_maps(ds,bank,progress=None):
    blocks=ds.blocks;ordinary=ordinary_features(ds);maps={};feature_rows=[];choices=[]
    for held in range(5):
        train=np.flatnonzero(blocks!=held);test=np.flatnonzero(blocks==held);train_blocks=sorted(set(blocks[train]))
        fit=bank.select(train_blocks,'M2');features,audit=state_features(ds,fit)
        choices.append(dict(dataset=ds.key,held_block=held,parameters=fit['candidate'].record(),
                            selection=fit['selection'],train_ids=fit['train_ids'],prefix_only=True))
        for i in test:
            feature_rows.append(dict(dataset=ds.key,trial_id=int(ds.ids[i]),block=held,
              prefix_end_s=ds.events[i]['end_s'],**{f'cognitive_{j}':float(v) for j,v in enumerate(features[i])},
              **{f'ordinary_{j}':float(v) for j,v in enumerate(ordinary[i])}))
        outer_features={'cognitive':features,'ordinary':ordinary}
        inner_features={}
        for inner in train_blocks:
            subset=[b for b in train_blocks if b!=inner]
            # EEG hyperparameters and readout reselected WITHOUT inner validation block.
            inner_fit=bank.select(subset,'M2')
            inner_features[inner]=state_features(ds,inner_fit)[0]
        for name,full in outer_features.items():
            final=np.stack([smoother(full,train,test,lam) for lam in LAMBDAS])
            inner_maps=[]
            for inner in train_blocks:
                tr=np.flatnonzero((blocks!=held)&(blocks!=inner));va=np.flatnonzero(blocks==inner)
                feat=inner_features[inner] if name=='cognitive' else ordinary
                inner_maps.append((tr,va,np.stack([smoother(feat,tr,va,lam) for lam in LAMBDAS])))
            maps[(held,name)]=(train,test,final,inner_maps)
        if progress:progress(f'{ds.key} RT: prefix-only outer block {held+1}/5 completed')
    return maps,feature_rows,choices


def predict_all_labels(ds,maps,log_targets):
    """Rows are true/permuted targets. Redo supervised inner selection per row."""
    nsets=len(log_targets);pred={k:np.empty((nsets,len(ds.events))) for k in ('median','ordinary','cognitive')}
    selected={k:np.empty((nsets,5),dtype=int) for k in ('ordinary','cognitive')}
    rt=np.exp(log_targets)
    for held in range(5):
        train=np.flatnonzero(ds.blocks!=held);test=np.flatnonzero(ds.blocks==held)
        pred['median'][:,test]=np.median(rt[:,train],axis=1)[:,None]
        for name in ('ordinary','cognitive'):
            tr,te,final,inner=maps[(held,name)]
            loss=np.zeros((nsets,len(LAMBDAS)))
            for ti,vi,h in inner:
                py=np.einsum('lvt,st->slv',h,log_targets[:,ti],optimize=True)
                loss+=np.mean((py-log_targets[:,None,vi])**2,axis=2)/len(inner)
            choice=np.argmin(loss,axis=1);selected[name][:,held]=choice
            all_pred=np.einsum('lvt,st->slv',final,log_targets[:,tr],optimize=True)
            # Numerical safety only; exp clipping not a data-tuned RT bound.
            pred[name][:,te]=np.exp(np.clip(all_pred[np.arange(nsets),choice],-10,10))
    return pred,selected


def rt_metrics(y,p,baseline):
    return dict(n=len(y),MAE=float(np.mean(abs(y-p))),RMSE=float(np.sqrt(np.mean((y-p)**2))),
        R2=float(1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)),
        S_vs_train_median=float(1-np.sum((y-p)**2)/np.sum((y-baseline)**2)),
        spearman=float(spearmanr(y,p).statistic))


def holm(values):
    p=np.asarray(values);order=np.argsort(p);out=np.empty(len(p));prev=0.
    for i,j in enumerate(order):
        prev=max(prev,(len(p)-i)*p[j]);out[j]=min(prev,1.)
    return out


def run_behavior(quick=False,n_perm=1999,n_boot=2000,progress=None):
    from .data import SEED
    rng=np.random.default_rng(SEED+7);predrows=[];feature_rows=[];choice_rows=[];auditrows=[];metrics=[];nullrows=[];intervals=[]
    pooled=[]
    for key in ('A2','B2'):
        ds,audit=prefix_dataset(key);auditrows+=audit
        if progress:progress(f'{key} RT: building prefix candidate bank ({len(ds.events)} trials)')
        bank=Bank(ds,candidates(2,quick));maps,features,choices=build_prediction_maps(ds,bank,progress)
        feature_rows+=features;choice_rows+=choices
        y=np.array([e['rt_s'] for e in ds.events]);log=np.tile(np.log(y),(n_perm+1,1))
        for rep in range(1,n_perm+1):
            for block in range(5):
                idx=np.flatnonzero(ds.blocks==block);log[rep,idx]=rng.permutation(log[rep,idx])
        preds,selections=predict_all_labels(ds,maps,log)
        for i,e in enumerate(ds.events):
            predrows.append(dict(dataset=key,trial_id=e['trial_id'],block=e['block'],rt_s=y[i],
               **{name:float(values[0,i]) for name,values in preds.items()},
               **{name+'_lambda':float(LAMBDAS[selections[name][0,e['block']]]) for name in selections}))
        for name,p in preds.items():metrics.append(dict(dataset=key,model=name,**rt_metrics(y,p[0],preds['median'][0])))
        pooled.append((key,ds.blocks,y,preds,np.exp(log)))
        del bank
    pooled.append(('pooled',np.concatenate([a[1]+5*j for j,a in enumerate(pooled)]),
       np.concatenate([a[2] for a in pooled]),{name:np.concatenate([a[3][name] for a in pooled],axis=1) for name in preds},
       np.concatenate([a[4] for a in pooled],axis=1)))
    for key,blocks,y,preds,ys in pooled:
        if key=='pooled':
            for name,p in preds.items():metrics.append(dict(dataset=key,model=name,**rt_metrics(y,p[0],preds['median'][0])))
        for baseline in ('median','ordinary'):
            improvement=np.mean(abs(ys-preds[baseline])-abs(ys-preds['cognitive']),axis=1)
            p=(1+np.sum(improvement[1:]>=improvement[0]))/(n_perm+1)
            nullrows.append(dict(dataset=key,contrast='cognitive_vs_'+baseline,MAE_gain=float(improvement[0]),
                                 p_raw=float(p),n_permutations=n_perm,null_mean=float(improvement[1:].mean())))
            unique=np.unique(blocks);boot=[]
            for _ in range(n_boot):
                # Stratify by recording for pooled inference; blocks stay intact.
                ix=[]
                for record in np.unique(blocks//5):
                    ub=unique[unique//5==record]
                    ix.extend(np.concatenate([np.flatnonzero(blocks==b) for b in rng.choice(ub,len(ub))]))
                ix=np.asarray(ix,dtype=int)
                boot.append(float(np.mean(abs(y[ix]-preds[baseline][0,ix])-abs(y[ix]-preds['cognitive'][0,ix]))))
            lo,hi=np.quantile(boot,[.025,.975])
            intervals.append(dict(dataset=key,contrast='cognitive_vs_'+baseline,MAE_gain=float(improvement[0]),
                                  ci_low=float(lo),ci_high=float(hi),n_bootstraps=n_boot))
    for row,p in zip(nullrows,holm([r['p_raw'] for r in nullrows])):row['p_Holm_6']=float(p)
    return dict(predictions=predrows,features=feature_rows,choices=choice_rows,audit=auditrows,
                metrics=metrics,permutation=nullrows,intervals=intervals,
                null_distribution={key:{b:np.mean(abs(ys-pr[b])-abs(ys-pr['cognitive']),axis=1)
                                       for b in ('median','ordinary')} for key,_,_,pr,ys in pooled})
