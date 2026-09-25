"""Supervised nested validation, complete permutations and uncertainty diagnostics."""
from pathlib import Path
import json
import numpy as np
from scipy.signal import savgol_filter
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix
from joblib import Parallel, delayed
from .data import TIMES, FS, FIT_MASK, WEIGHTS, B, means, modes, window_features, save_json, save_csv
from .model import train_model, fit_erp, waveform_metrics, KernelFactory, coefficients, weighted_error, BOUNDS


def classify(train_features,y,test_features):
    if not np.isfinite(train_features).all() or not np.isfinite(test_features).all():
        raise ValueError('Nonfinite features cannot be silently imputed')
    if len(np.unique(y))!=2: raise ValueError('Training fold has only one class')
    scaler=StandardScaler().fit(train_features)
    lda=LinearDiscriminantAnalysis(solver='lsqr',shrinkage='auto',priors=[.5,.5])
    lda.fit(scaler.transform(train_features),y)
    p=lda.predict_proba(scaler.transform(test_features))[:,list(lda.classes_).index(1)]
    return np.where(p>=.5,1,-1),p,scaler,lda


def metrics(y,pred,score):
    cm=confusion_matrix(y,pred,labels=[-1,1]);den=cm.sum(1)
    recall=np.divide(cm.diagonal(),den,out=np.full(2,np.nan),where=den!=0)
    both=np.all(den>0)
    return dict(n=len(y), BA=float(recall.mean()) if both else np.nan,
        AUC=float(roc_auc_score(y,score)) if both else np.nan,
        recall_left=recall[0],recall_right=recall[1],TN=int(cm[0,0]),FP=int(cm[0,1]),
        FN=int(cm[1,0]),TP=int(cm[1,1]))


def fit_record(f):
    return dict(rank=f.rank,lambda_A=f.lam,theta=f.theta.tolist(),A=f.A.tolist(),D=f.D.tolist(),
                diagnostics=f.diagnostics)


def extra_features(ds):
    prewindows=[(-.25,-.167),(-.167,-.083),(-.083,0)]
    pre=window_features(ds.x, windows=prewindows)
    strict=window_features(ds.prestim_only, times=TIMES[:64],windows=prewindows)
    return {'window_mean_9':window_features(ds.x),'prestim_zero_phase':pre,
            'prestim_past_only':strict,'trial_order':ds.ids[:,None].astype(float),
            'artifact_quality':np.log1p(ds.artifact)}


def evaluate_split(tr,te,quick=False,extras=True,tau_g=20.):
    f=train_model(tr,quick=quick,tau_g=tau_g)
    ft=f.features(tr.x);fe=f.features(te.x)
    pred,score,scaler,lda=classify(ft,tr.y,fe)
    predictions={'mechanism_selected':(pred,score)}
    fit_models={'M1':f}
    obs=means(te.x,te.y)
    templates=means(tr.x,tr.y)
    waveform={'M1':f.predict(),'M2_template':templates}
    f3=None
    if extras:
        f0=train_model(tr,quick=quick,tau_g=tau_g,direction=False);fit_models['M0']=f0;waveform['M0']=f0.predict()
        for rank in (2,3):
            fr=f if rank==f.rank else train_model(tr,quick=quick,rank_constraint=rank,tau_g=tau_g)
            fit_models[f'R{rank}']=fr
            if rank==3:f3=fr
            predictions[f'kernel_R{rank}']=classify(fr.features(tr.x),tr.y,fr.features(te.x))[:2]
        for name,mode in [('common_only',0),('lateral_only',2)]:
            predictions[name]=classify(ft.reshape(len(tr.x),3,f.rank)[:,mode],tr.y,
                                       fe.reshape(len(te.x),3,f.rank)[:,mode])[:2]
        tf=extra_features(tr);vf=extra_features(te)
        for name in tf: predictions[name]=classify(tf[name],tr.y,vf[name])[:2]
    wave_rows=[];wave_detail=[]
    for name,p in waveform.items():
        details,summary=waveform_metrics(obs,p)
        wave_rows.append(dict(model=name,**summary));wave_detail.extend(dict(model=name,**r) for r in details)
    return dict(fit=f,models=fit_models,predictions=predictions,obs=obs,waveform=waveform,
                wave_rows=wave_rows,wave_detail=wave_detail,features=fe,fit3=f3,
                classifier=dict(mean=scaler.mean_,scale=scaler.scale_,coef=lda.coef_,intercept=lda.intercept_))


def nested_cv(ds,quick=False,extras=True,out=None,tag='blocked',tau_g=20.):
    rows=[];waves=[];details=[];folds=[];features=[]
    for block in np.unique(ds.blocks):
        tr=ds.subset(ds.blocks!=block);te=ds.subset(ds.blocks==block)
        r=evaluate_split(tr,te,quick,extras,tau_g)
        for model,(pred,score) in r['predictions'].items():
            rows.extend(dict(dataset=ds.key,validation=tag,fold=int(block),trial_id=int(i),truth=int(y),
                             model=model,prediction=int(p),score=float(s)) for i,y,p,s in zip(te.ids,te.y,pred,score))
        waves.extend(dict(dataset=ds.key,validation=tag,fold=int(block),**v) for v in r['wave_rows'])
        details.extend(dict(dataset=ds.key,validation=tag,fold=int(block),**v) for v in r['wave_detail'])
        rec=dict(fold=int(block),train_ids=tr.ids.tolist(),test_ids=te.ids.tolist(),fit=fit_record(r['fit']))
        if extras:rec['models']={name:fit_record(model) for name,model in r['models'].items()}
        folds.append(rec)
        if out:
            dest=Path(out)/'validation'/tag/ds.key;dest.mkdir(parents=True,exist_ok=True)
            save_json(rec,dest/f'fold_{block}.json')
            np.savez_compressed(dest/f'fold_{block}.npz',test_ids=te.ids,truth=te.y,obs=r['obs'],
                **{f'prediction_{k}':v for k,v in r['waveform'].items()},h=r['fit'].h,
                A=r['fit'].A,D=r['fit'].D,theta=r['fit'].theta,features=r['features'],
                **r['classifier'])
            if r['fit3'] is not None:
                f3=r['fit3']; train3=f3.features(tr.x);test3=f3.features(te.x)
                standardized=StandardScaler().fit(train3).transform(test3)
                for j,i in enumerate(te.ids):
                    features.append(dict(dataset=ds.key,fold=int(block),trial_id=int(i),cue=int(te.y[j]),
                        **{f'coef_{k+1}':float(v) for k,v in enumerate(test3[j])},
                        **{f'z_{k+1}':float(v) for k,v in enumerate(standardized[j])}))
    if out:
        save_csv(rows,Path(out)/'validation'/tag/f'{ds.key}_predictions.csv')
        save_csv(waves,Path(out)/'validation'/tag/f'{ds.key}_waveform.csv')
        save_csv(details,Path(out)/'validation'/tag/f'{ds.key}_waveform_channels.csv')
        if features:save_csv(features,Path(out)/'features'/f'{ds.key}_oof_nine_features.csv')
    return rows,waves,folds


def block_intervals(rows,n_boot=2000,seed=42):
    import pandas as pd
    df=pd.DataFrame(rows);results=[];rng=np.random.default_rng(seed)
    for model,g in df.groupby('model',sort=True):
        base=metrics(g.truth.to_numpy(),g.prediction.to_numpy(),g.score.to_numpy())
        blocks=g.fold.unique(); draws=[]
        for _ in range(n_boot):
            sampled=pd.concat([g[g.fold==b] for b in rng.choice(blocks,len(blocks))],ignore_index=True)
            m=metrics(sampled.truth.to_numpy(),sampled.prediction.to_numpy(),sampled.score.to_numpy())
            draws.append([m['BA'],m['AUC']])
        interval=np.nanquantile(draws,[.025,.975],axis=0)
        results.append(dict(dataset=g.dataset.iloc[0],model=model,**base,BA_low=interval[0,0],
            BA_high=interval[1,0],AUC_low=interval[0,1],AUC_high=interval[1,1],n_boot=n_boot,
            interval='fixed_OOF_models_time_block_bootstrap'))
    return results


def permutation_job(ds,index,seed,quick,out):
    path=Path(out)/'checkpoints'/'permutations'/ds.key/f'{index:04d}.json'
    if path.exists():
        data=json.loads(path.read_text())
        if data.get('status')=='complete':return data
    rng=np.random.default_rng(np.random.SeedSequence([seed,ord(ds.key[0]),int(ds.key[1]),index]))
    y=ds.y.copy()
    for b in np.unique(ds.blocks):
        mask=np.flatnonzero(ds.blocks==b);y[mask]=rng.permutation(y[mask])
    try:
        rows,_,folds=nested_cv(ds.labels(y),quick,extras=False)
        truth=np.array([r['truth'] for r in rows]);pred=np.array([r['prediction'] for r in rows]);scores=np.array([r['score'] for r in rows])
        result=dict(index=index,status='complete',**metrics(truth,pred,scores),
            trial_ids=[r['trial_id'] for r in rows],permuted_truth=truth,predictions=pred,scores=scores,
            folds=[dict(fold=f['fold'],rank=f['fit']['rank'],lambda_A=f['fit']['lambda_A'],
                        theta=f['fit']['theta'],starts=f['fit']['diagnostics']['starts'],
                        inner_candidates=f['fit']['diagnostics']['selection_candidates']) for f in folds])
        save_json(result,path);return result
    except Exception as exc:
        save_json(dict(index=index,status='failed',error=repr(exc)),path)
        raise


def run_permutations(ds,observed,n,seed,quick,out,n_jobs):
    jobs=Parallel(n_jobs=n_jobs,return_as='generator_unordered',batch_size=1)(
        delayed(permutation_job)(ds,i,seed,quick,out) for i in range(n))
    values=[]
    for k,result in enumerate(jobs,1):
        values.append(result)
        if k%25==0 or k==n: print(f'PERM {ds.key} {k}/{n}',flush=True)
    vals=np.array([r['BA'] for r in values]);p=(1+np.sum(vals>=observed))/(n+1)
    save_csv([dict(index=r['index'],BA=r['BA'],AUC=r['AUC']) for r in sorted(values,key=lambda v:v['index'])],
             Path(out)/'validation'/'permutations'/f'{ds.key}.csv')
    return dict(dataset=ds.key,observed_BA=observed,permutations=n,p_raw=float(p))


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p);result=np.empty_like(p)
    result[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return result


def pooled_correction(tr,te):
    # Fixed conservative sensitivity, fitted WITHOUT directions, never selected by BA.
    score=np.log1p(np.ptp(tr.x,axis=-1).max(1))
    tpl=np.median(tr.x[np.argsort(score)[:max(8,len(score)//2)]],axis=0)
    res=tr.x-tpl
    scale=np.maximum(np.median(np.abs(res-np.median(res,axis=0)),axis=0)*1.4826,1e-6)
    def apply(x):
        residual=x-tpl
        smooth=savgol_filter(residual,31,3,axis=-1)
        gate=np.clip((np.abs(smooth)/scale-3.5)/3.5,0,1)
        y=x-.15*smooth*gate
        return y-np.median(y[:,:,:64],axis=-1,keepdims=True)
    from dataclasses import replace
    return replace(tr,x=apply(tr.x)),replace(te,x=apply(te.x))


def supplemental(datasets,root,out,quick=False):
    from .data import independent_data
    rows=[];waves=[]
    def run(tr,te,tag):
        r=evaluate_split(tr,te,quick,extras=False)
        pred,score=r['predictions']['mechanism_selected']
        rows.append(dict(source=tr.key,target=te.key,validation=tag,**metrics(te.y,pred,score)))
        save_json(dict(train_ids=tr.ids,test_ids=te.ids,fit=fit_record(r['fit']),truth=te.y,
                       predictions=pred,scores=score),Path(out)/'validation'/'supplemental'/f'{tag}_{tr.key}_{te.key}.json')
        waves.extend(dict(source=tr.key,target=te.key,validation=tag,**w) for w in r['wave_rows'])
    for ds in datasets.values():
        run(ds.subset(ds.blocks<3),ds.subset(ds.blocks>=3),'forward_3_to_2')
        ds32=independent_data(root,ds.key,32,out)
        pr,wr,_=nested_cv(ds32,quick,False,out,'guard32')
        import pandas as pd
        df=pd.DataFrame(pr);m=metrics(df.truth.to_numpy(),df.prediction.to_numpy(),df.score.to_numpy())
        rows.append(dict(source=ds.key,target=ds.key,validation='guard32',**m))
        pooled=[]
        for b in range(5):
            tr,te=pooled_correction(ds.subset(ds.blocks!=b),ds.subset(ds.blocks==b))
            r=evaluate_split(tr,te,quick,False);pred,score=r['predictions']['mechanism_selected']
            pooled.extend(dict(fold=b,trial_id=int(i),truth=int(y),prediction=int(p),score=float(s)) for i,y,p,s in zip(te.ids,te.y,pred,score))
        g=pd.DataFrame(pooled)
        rows.append(dict(source=ds.key,target=ds.key,validation='pooled_label_free_correction',
            **metrics(g.truth.to_numpy(),g.prediction.to_numpy(),g.score.to_numpy())))
        save_csv(pooled,Path(out)/'validation'/'supplemental'/f'{ds.key}_pooled_correction.csv')
    for source,tr in datasets.items():
        for target,te in datasets.items():
            if source!=target and (source[0]==target[0] or source[1]==target[1]):
                run(tr,te,'cross_project' if source[0]==target[0] else 'cross_subject')
    save_csv(rows,Path(out)/'validation'/'supplemental_metrics.csv')
    save_csv(waves,Path(out)/'validation'/'supplemental_waveforms.csv')
    return rows


def bootstrap_fit_job(ds,index,seed,quick,out):
    p=Path(out)/'checkpoints'/'bootstrap'/ds.key/f'{index:04d}.json'
    if p.exists():
        r=json.loads(p.read_text())
        if r.get('status')=='complete':return r
    rng=np.random.default_rng(np.random.SeedSequence([seed,99,ord(ds.key[0]),int(ds.key[1]),index]))
    ids=[]
    for b in np.unique(ds.blocks):
        for c in (-1,1):
            s=np.flatnonzero((ds.blocks==b)&(ds.y==c));ids.extend(rng.choice(s,len(s)))
    try:
        f=train_model(ds.subset(np.array(ids)),quick)
        r=dict(status='complete',index=index,rank=f.rank,lambda_A=f.lam,theta=f.theta,A=f.A,D=f.D,
               condition=f.diagnostics['condition'],bound_hits=f.diagnostics['bound_hits'],
               fit_status=f.diagnostics['numerical_status'])
        save_json(r,p);return r
    except Exception as exc:
        save_json(dict(status='failed',index=index,error=repr(exc)),p);raise


def stability(ds,out,quick=False,n_boot=200,n_jobs=1,seed=42):
    f=train_model(ds,quick);dest=Path(out)/'stability'/ds.key;dest.mkdir(parents=True,exist_ok=True)
    save_json(fit_record(f),dest/'full_independent_fit.json')
    np.savez_compressed(dest/'full_independent_fit.npz',h=f.h,A=f.A,D=f.D,theta=f.theta)
    erp=means(ds.x,ds.y);profile=[]
    # Profile over a fixed coordinate; re-fit all other time parameters, A and D.
    from scipy.optimize import minimize
    for j,(lo,hi) in enumerate(BOUNDS[:f.rank+1]):
        for value in np.linspace(lo,hi,11):
            start=f.theta.copy();start[j]=value
            if start[2]<start[1]+10:
                if j==1:start[2]=min(250.,start[1]+10)
                else:start[1]=max(20.,start[2]-10)
            if f.rank==3 and start[3]<start[2]+10:
                if j==3:start[2]=max(70.,start[3]-10)
                else:start[3]=min(600.,start[2]+10)
            fac=KernelFactory(np.median(ds.durations))
            def obj(t):
                h=fac.kernel(t,f.rank);a,d=coefficients(erp,h,f.lam)
                from .model import predict_erp
                return 6*weighted_error(erp,predict_erp(a,d,h))+f.lam*np.sum(a*a)+10*f.lam*np.sum(d*d)
            cons=[{'type':'ineq','fun':lambda t:t[2]-t[1]-10}]
            if f.rank==3:cons.append({'type':'ineq','fun':lambda t:t[3]-t[2]-10})
            lows=np.array([a for a,b in BOUNDS[:f.rank+1]]);spans=np.array([b-a for a,b in BOUNDS[:f.rank+1]])
            bounds=[(0.,1.)]*(f.rank+1);fixed=(value-lows[j])/spans[j];bounds[j]=(fixed,fixed)
            cons=[{'type':'ineq','fun':lambda u:(lows+u*spans)[2]-(lows+u*spans)[1]-10}]
            if f.rank==3:cons.append({'type':'ineq','fun':lambda u:(lows+u*spans)[3]-(lows+u*spans)[2]-10})
            fit=minimize(lambda u:obj(lows+u*spans)/max(f.diagnostics['objective'],1),(start-lows)/spans,
                         method='SLSQP',bounds=bounds,constraints=cons,options={'maxiter':65,'eps':1e-4,'ftol':1e-8})
            solution=lows+fit.x*spans
            profile.append(dict(parameter=j,value_ms=value,objective=obj(solution),success=bool(fit.success),theta=solution.tolist()))
    save_csv(profile,dest/'parameter_profiles.csv')
    sens=[]
    for tg in (10.,20.,40.):
        ff=train_model(ds,quick,tau_g=tg);sens.append(dict(tau_G=tg,rank=ff.rank,lambda_A=ff.lam,
                theta=ff.theta.tolist(),condition=ff.diagnostics['condition'],objective=ff.diagnostics['objective']))
        save_json(fit_record(ff),dest/f'tauG_{tg:g}.json')
    for lam in (.01,.1,1.):
        ff=fit_erp(erp,np.median(ds.durations),f.rank,lam,quick=quick)
        sens.append(dict(tau_G=20,rank=ff.rank,lambda_A=lam,theta=ff.theta.tolist(),
                         condition=ff.diagnostics['condition'],objective=ff.diagnostics['objective']))
    save_csv(sens,dest/'sensitivity.csv')
    jobs=Parallel(n_jobs=n_jobs,return_as='generator_unordered',batch_size=1)(delayed(bootstrap_fit_job)(ds,i,seed,quick,out) for i in range(n_boot))
    rows=[]
    for k,r in enumerate(jobs,1):
        rows.append(r)
        if k%25==0 or k==n_boot:print(f'BOOT {ds.key} {k}/{n_boot}',flush=True)
    save_json(sorted(rows,key=lambda r:r['index']),dest/'full_refit_bootstrap.json')
    return f
