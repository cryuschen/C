"""Nested, label-safe feature extraction, LDA, uncertainty and null inference."""
import warnings
import numpy as np
import pandas as pd
from scipy.fft import dct
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
from .data import TIMES, WEIGHTS, B, means

FEATURE_SETS=('amplitude','peak_latency','spatial','mechanism','covariance','projection6','direction3','past_only','previous_cue','post_given_past')
PRIMARY=FEATURE_SETS[:7]
MASK=(TIMES>=.25)&(TIMES<.5)
POST=WEIGHTS>0


def basic_features(x):
    amp=x[...,MASK].mean(-1); peak=np.full_like(amp,np.nan); latency=peak.copy()
    for n in range(len(x)):
        for c in range(3):
            a=x[n,c,MASK]; sd=np.std(x[n,c,TIMES<0])
            ids,info=find_peaks(a,prominence=max(.5*sd,1e-8))
            ids=ids[a[ids]>0]
            if len(ids):
                j=ids[np.argmax(a[ids])]; peak[n,c]=a[j]; latency[n,c]=TIMES[MASK][j]*1000
    return np.c_[amp,peak,latency,np.isnan(peak).astype(float)]


def covariance_features(x):
    z=np.einsum('cm,nct->nmt',B,x); out=[]
    for lo,hi in ((.05,.25),(.25,.5),(.5,.75),(.05,.75)):
        a=z[...,(TIMES>=lo)&(TIMES<hi)]; a=a-a.mean(-1,keepdims=True)
        cov=np.einsum('nmt,nkt->nmk',a,a)/a.shape[-1]
        sd=np.sqrt(np.maximum(np.diagonal(cov,axis1=1,axis2=2),1e-12))
        out.extend([np.log(sd[:,j]**2) for j in range(3)])
        out.extend([cov[:,j,k]/(sd[:,j]*sd[:,k]) for j,k in ((0,1),(0,2),(1,2))])
    return np.column_stack(out)


def past_features(x):
    return dct(np.einsum('cm,nct->nmt',B,x),norm='ortho',axis=-1)[...,:6].reshape(len(x),-1)


def standardize(train,test):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        fill=np.nanmedian(train,axis=0)
    fill=np.nan_to_num(fill)
    a=np.where(np.isfinite(train),train,fill); b=np.where(np.isfinite(test),test,fill)
    mu=a.mean(0); sd=np.maximum(a.std(0),1e-8)
    return np.clip((a-mu)/sd,-8,8),np.clip((b-mu)/sd,-8,8)


def lda(train,y,test):
    a,b=standardize(train,test)
    l=a[y==-1]; r=a[y==1]
    if not len(l) or not len(r):
        raise ValueError('LDA requires both directions')
    ml,mr=l.mean(0),r.mean(0)
    res=np.r_[l-ml,r-mr]; cov=res.T@res/max(len(a)-2,1)
    cov=.2*cov+.8*max(np.trace(cov)/len(cov),1e-6)*np.eye(len(cov))
    w=np.linalg.solve(cov,mr-ml)
    # Training-only scale puts scores from different outer folds on a similar scale.
    scale=max(np.sqrt(w@cov@w),1e-8)
    return (b-.5*(ml+mr))@w/scale


def spatial_features(base,x,train):
    a=base[:,:3]
    denom=np.abs(a[:,1])+np.abs(a[:,2])
    floor=max(float(np.quantile(denom[train],.1)),1e-8)
    return np.c_[base,(a[:,2]-a[:,1])/np.maximum(denom,floor),a[:,2]-a[:,1]],floor


def mechanism_features(x,pred):
    """Two fitted forward templates per channel; no test condition is consulted."""
    templates=np.stack([pred.mean(0),pred[1]-pred[0]],axis=1)[...,POST]
    a=x[...,POST]; columns=[]; residual=[]
    for c in range(3):
        h=templates[c].T
        h=h/np.maximum(np.linalg.norm(h,axis=0),1e-10)
        coef=a[:,c]@h@np.linalg.inv(h.T@h+1e-3*np.eye(2))
        columns.append(coef)
        residual.append(np.sqrt(np.mean((a[:,c]-coef@h.T)**2,axis=1)))
    return np.c_[*columns,np.column_stack(residual)]


def prepare_decoder(ds,bank,previous):
    return dict(ds=ds,base=basic_features(ds.x),cov=covariance_features(ds.x),
                past=past_features(ds.prestim_only),previous=np.asarray(previous)[:,None],
                projected=bank.prepare(ds.x))


def feature_fold(data,bank,y,train,test):
    ds=data['ds']; best,loss=bank.select_prepared(data['projected'],y,ds.blocks,train)
    coef=bank.inverse[best]@bank.prepared_stats(data['projected'],y,train)[best]
    pred=bank.predict(coef,best)
    spatial,floor=spatial_features(data['base'],ds.x,train)
    projections=mechanism_features(ds.x,pred)
    mech=np.c_[spatial,projections]
    choices={'amplitude':data['base'][:,:3],'peak_latency':data['base'],
             'spatial':spatial,'mechanism':mech,'covariance':data['cov'],
             'projection6':projections[:,:6], 'direction3':projections[:,[1,3,5]],
             'past_only':data['past'],'previous_cue':data['previous']}
    # Incremental response after a training-only, label-blind regression on past EEG.
    pretr,prete=standardize(data['past'][train],data['past'][test])
    posttr,postte=standardize(mech[train],mech[test])
    coefpast=np.linalg.solve(pretr.T@pretr+len(pretr)*np.eye(pretr.shape[1]),pretr.T@posttr)
    residual=(posttr-pretr@coefpast,postte-prete@coefpast)
    result={name:(a[train],a[test]) for name,a in choices.items()}
    result['post_given_past']=residual
    meta=dict(candidate=best,scale=bank.meta[best].scale,recurrence=bank.meta[best].recurrence,
              ridge=float(bank.reg[best]),li_floor=floor,inner_loss=loss.tolist())
    return result,meta


def decode(data,bank,labels=None,details=False):
    ds=data['ds']; y=ds.y if labels is None else labels
    scores={k:np.full(len(y),np.nan) for k in FEATURE_SETS}; features={}; choices=[]
    for held in np.unique(ds.blocks):
        tr=ds.blocks!=held; te=~tr
        f,meta=feature_fold(data,bank,y,tr,te)
        for name,(a,b) in f.items():
            scores[name][te]=lda(a,y[tr],b)
            if details:
                if name not in features:
                    features[name]=np.full((len(y),b.shape[1]),np.nan)
                features[name][te]=b
        if details:
            choices.append(dict(dataset=ds.key,held_block=int(held),
                                train_ids=ds.ids[tr].tolist(),test_ids=ds.ids[te].tolist(),**meta))
    if any(not np.isfinite(v).all() for v in scores.values()):
        raise ValueError('Incomplete out-of-fold prediction')
    return scores,features,choices


def metrics(y,scores):
    p=np.where(scores>=0,1,-1); tp=int(np.sum((p==1)&(y==1))); tn=int(np.sum((p==-1)&(y==-1)))
    fp=int(np.sum((p==1)&(y==-1))); fn=int(np.sum((p==-1)&(y==1)))
    return dict(n=len(y),Accuracy=float(np.mean(p==y)),BA=.5*(tp/max(tp+fn,1)+tn/max(tn+fp,1)),
                Precision=tp/max(tp+fp,1),Recall=tp/max(tp+fn,1),F1=2*tp/max(2*tp+fp+fn,1),
                AUC=float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else np.nan,
                TN=tn,FP=fp,FN=fn,TP=tp)


def evaluate_all(prepared,bank,labels=None,details=False):
    outputs={}; featuremap={}; choice=[]
    for key,data in prepared.items():
        current=data
        if labels is not None and 'all_labels' in data:
            all_y=data['all_labels'].copy(); all_y[data['ds'].ids-1]=labels[key]
            current=dict(data,previous=np.r_[0,all_y[:-1]][data['ds'].ids-1,None])
        outputs[key],featuremap[key],ch=decode(current,bank,None if labels is None else labels[key],details)
        choice.extend(ch)
    rows=[]
    for key in (*prepared,'pooled'):
        keys=list(prepared) if key=='pooled' else [key]
        y=np.concatenate([prepared[k]['ds'].y if labels is None else labels[k] for k in keys])
        for name in FEATURE_SETS:
            rows.append(dict(dataset=key,features=name,**metrics(y,np.concatenate([outputs[k][name] for k in keys]))))
    return pd.DataFrame(rows),outputs,featuremap,choice


def null_labels(prepared,rng,kind):
    result={}
    for key,data in prepared.items():
        ds=data['ds']; y=ds.y.copy()
        if kind=='block_permutation':
            for block in np.unique(ds.blocks):
                ix=np.flatnonzero(ds.blocks==block); y[ix]=rng.permutation(y[ix])
        elif kind=='circular_shift':
            y=np.roll(y,int(rng.integers(1,len(y))))
        else:
            raise ValueError(kind)
        result[key]=y
    return result


def bootstrap_metrics(prepared,outputs,n_boot,seed):
    rng=np.random.default_rng(seed); rows=[]
    for key in (*prepared,'pooled'):
        keys=list(prepared) if key=='pooled' else [key]
        values={name:[] for name in FEATURE_SETS}; auc={name:[] for name in FEATURE_SETS}
        gains=[]; space=[]; past=[]; reduced={n:[] for n in ('projection6','direction3')}
        for _ in range(n_boot):
            ys=[]; ss={name:[] for name in FEATURE_SETS}
            for k in keys:
                ds=prepared[k]['ds']; blocks=np.unique(ds.blocks)
                ix=np.concatenate([np.flatnonzero(ds.blocks==b) for b in rng.choice(blocks,len(blocks))])
                ys.append(ds.y[ix])
                for name in FEATURE_SETS:
                    ss[name].append(outputs[k][name][ix])
            y=np.concatenate(ys)
            for name in FEATURE_SETS:
                m=metrics(y,np.concatenate(ss[name])); values[name].append(m['BA']); auc[name].append(m['AUC'])
            gains.append(values['mechanism'][-1]-values['spatial'][-1])
            space.append(values['spatial'][-1]-values['peak_latency'][-1])
            past.append(values['post_given_past'][-1]-values['past_only'][-1])
            for name in reduced:
                reduced[name].append(values[name][-1]-values['amplitude'][-1])
        for name in FEATURE_SETS:
            rows.append(dict(dataset=key,features=name,BA_low=np.quantile(values[name],.025),
                             BA_high=np.quantile(values[name],.975),AUC_low=np.quantile(auc[name],.025),
                             AUC_high=np.quantile(auc[name],.975)))
        for name,vals in [('mechanism_minus_spatial',gains),('spatial_minus_peak',space),('residual_minus_past',past)]+[(n+'_minus_amplitude',v) for n,v in reduced.items()]:
            rows.append(dict(dataset=key,features=name,BA_low=np.quantile(vals,.025),BA_high=np.quantile(vals,.975)))
    return pd.DataFrame(rows)


def transfer(train_data,test_data,bank):
    src=train_data['ds']; dst=test_data['ds']
    best,_=bank.select_prepared(train_data['projected'],src.y,src.blocks)
    coef=bank.fit(means(src.x,src.y),best); pred=bank.predict(coef,best)
    base=np.r_[train_data['base'],test_data['base']]; x=np.r_[src.x,dst.x]
    tr=np.arange(len(x))<len(src.x); te=~tr
    spatial,_=spatial_features(base,x,tr); projection=mechanism_features(x,pred); mech=np.c_[spatial,projection]
    variants={'amplitude':base[:,:3],'peak_latency':base,'spatial':spatial,'mechanism':mech,
              'covariance':np.r_[train_data['cov'],test_data['cov']],
              'projection6':projection[:,:6], 'direction3':projection[:,[1,3,5]]}
    return {name:lda(a[tr],src.y,a[te]) for name,a in variants.items()},best
