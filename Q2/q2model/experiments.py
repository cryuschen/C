"""Descriptive fits, joint-time-scale comparison, and model/misspecification checks."""
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from .data import NAMES, FS, TIMES, B, FIT_MASK, WEIGHTS, Dataset, descriptive_data, means, save_json, save_csv
from .model import (LAMBDAS, BOUNDS, KernelFactory, Fit, train_model, fit_erp, profile_grid,
                    coefficients, predict_erp, weighted_error, waveform_metrics, allowed)
from .validation import fit_record, nested_cv, metrics


def descriptive(root,out,quick=False):
    all_metrics=[];params=[]
    for key in NAMES:
        for stage in ('before','v7'):
            ds=descriptive_data(root,key,stage); f=train_model(ds,quick)
            f3=f if f.rank==3 else train_model(ds,quick,rank_constraint=3)
            dest=Path(out)/'fits'/key/stage;dest.mkdir(parents=True,exist_ok=True)
            obs=means(ds.x,ds.y);pred=f.predict()
            save_json(fit_record(f),dest/'model.json')
            save_json(fit_record(f3),dest/'nine_feature_model.json')
            np.savez_compressed(dest/'waveforms.npz',observed=obs,predicted=pred,residual=obs-pred,
                                 h=f.h,A=f.A,D=f.D,theta=f.theta,times=TIMES,trial_ids=ds.ids,cues=ds.y)
            details,summary=waveform_metrics(obs,pred)
            all_metrics.extend(dict(dataset=key,stage=stage,**r) for r in details)
            params.append(dict(dataset=key,stage=stage,rank=f.rank,lambda_A=f.lam,
                delta_ms=f.theta[0],tau_E_ms=f.theta[1],tau_I_ms=f.theta[2],
                tau_S_ms=f.theta[3] if f.rank==3 else np.nan,condition=f.diagnostics['condition'],
                bound_hits=str(f.diagnostics['bound_hits']),**summary))
            coef=f3.features(ds.x)
            save_csv([dict(trial_id=int(i),cue=int(y),**{f'coef_{k+1}':v for k,v in enumerate(c)})
                      for i,y,c in zip(ds.ids,ds.y,coef)],dest/'descriptive_nine_features.csv')
            print(f'DESCRIPTION {key} {stage} R={f.rank}',flush=True)
    save_csv(all_metrics,Path(out)/'fits'/'channel_metrics.csv')
    save_csv(params,Path(out)/'fits'/'parameters.csv')


def fit_shared(datasets,rank,lam,quick=False,refine=False):
    erps=[means(d.x,d.y) for d in datasets]
    durations=[np.median(d.durations) for d in datasets]
    prof=[profile_grid(e,t,rank,lam,quick=quick) for e,t in zip(erps,durations)]
    theta=prof[0][1];scores=sum(p[-1] for p in prof);idx=int(np.argmin(scores))
    factories=[p[0] for p in prof]
    def solve(t):
        fits=[];loss=0.
        for erp,fac in zip(erps,factories):
            h=fac.kernel(t,rank);a,d=coefficients(erp,h,lam)
            loss+=6*weighted_error(erp,predict_erp(a,d,h))+lam*np.sum(a*a)+10*lam*np.sum(d*d)
            fits.append(Fit(rank,lam,t,a,d,h,{}))
        return loss,fits
    best_theta=theta[idx];best,fs=solve(best_theta)
    if refine:
        lows=np.array([a for a,b in BOUNDS[:rank+1]]);spans=np.array([b-a for a,b in BOUNDS[:rank+1]])
        con=[{'type':'ineq','fun':lambda u:(lows+u*spans)[2]-(lows+u*spans)[1]-10}]
        if rank==3:con.append({'type':'ineq','fun':lambda u:(lows+u*spans)[3]-(lows+u*spans)[2]-10})
        scale=max(best,1.)
        for index in np.argsort(scores)[:3]:
            opt=minimize(lambda u:solve(lows+u*spans)[0]/scale,(theta[index]-lows)/spans,
                method='SLSQP',bounds=[(0,1)]*(rank+1),constraints=con,
                options={'maxiter':65,'eps':1e-4,'ftol':1e-8})
            val,ff=solve(lows+opt.x*spans)
            if opt.success and val<best:best=val;fs=ff
    return fs


def shared_project_test(datasets,out,quick=False):
    rows=[]
    for subject in ('A','B'):
        pair=[datasets[subject+'1'],datasets[subject+'2']]
        for block in range(5):
            tr=[d.subset(d.blocks!=block) for d in pair];te=[d.subset(d.blocks==block) for d in pair]
            candidates=[]
            for rank in (2,3):
                for lam in LAMBDAS:
                    losses=[]
                    for inner in np.unique(tr[0].blocks):
                        itr=[d.subset(d.blocks!=inner) for d in tr];iva=[d.subset(d.blocks==inner) for d in tr]
                        fits=fit_shared(itr,rank,lam,quick)
                        losses.append(np.mean([weighted_error(means(d.x,d.y),f.predict()) for d,f in zip(iva,fits)]))
                    candidates.append((np.mean(losses),rank,lam))
            minimum=min(x[0] for x in candidates)
            chosen=min([x for x in candidates if x[0]<=minimum+max(1e-10,.01*abs(minimum))],key=lambda x:(x[1],-x[2]))
            shared=fit_shared(tr,chosen[1],chosen[2],quick,True)
            for train,test,joint in zip(tr,te,shared):
                independent=train_model(train,quick)
                for name,f in [('shared_theta',joint),('independent_theta',independent)]:
                    _,m=waveform_metrics(means(test.x,test.y),f.predict())
                    rows.append(dict(subject=subject,dataset=test.key,fold=block,model=name,
                                     rank=f.rank,lambda_A=f.lam,theta=f.theta.tolist(),**m))
    save_csv(rows,Path(out)/'validation'/'shared_project_parameters.csv')
    return rows


def synthetic_tests(out,quick=False,seed=42):
    rng=np.random.default_rng(seed);rank=3;theta=np.array([65.,50.,135.,350.])
    fac=KernelFactory();h=fac.kernel(theta,rank)
    A=np.array([[20,10,15],[5,10,2],[3,4,3.]])
    D=np.array([[4,1,2],[2,-2,1],[3,4,1.]])
    truth=predict_erp(A,D,h)
    a,d=coefficients(truth,h,0)
    closure=dict(A_max_error=float(np.max(abs(A-a))),D_max_error=float(np.max(abs(D-d))),
                 waveform_max_error=float(np.max(abs(truth-predict_erp(a,d,h)))))
    n=100;blocks=np.repeat(np.arange(5),20)
    y=np.concatenate([rng.permutation(np.tile([-1,1],10)) for _ in range(5)])
    base=np.stack([truth[(yy+1)//2] for yy in y])
    noise=rng.normal(0,8,size=base.shape)
    from scipy.ndimage import gaussian_filter1d
    noise=gaussian_filter1d(noise,2,axis=-1)
    noise-=np.median(noise[:,:,:64],axis=-1,keepdims=True)
    eye=(np.tanh((TIMES-.12)/.04)-np.tanh((TIMES-.68)/.06))/2
    eye=eye-eye[:64].mean()
    scenarios={'model_assumed':base+noise,
        'eye_only_no_neural_direction':np.repeat((B@A@h.T)[None,:,:],n,axis=0)+noise+
            y[:,None,None]*np.array([.1,1,-1])[None,:,None]*eye[None,None,:]*30,
        'background_drift':base+noise+np.linspace(-35,35,n)[:,None,None]*np.ones((1,3,1))*TIMES[None,None,:],
        'time_parameter_drift':np.stack([predict_erp(A,D,fac.kernel(theta+np.array([0,0,0,120*(i/(n-1)-.5)]),3))[(yy+1)//2] for i,yy in enumerate(y)])+noise}
    results=[];dest=Path(out)/'synthetic';dest.mkdir(parents=True,exist_ok=True)
    save_json(closure,dest/'exact_projection_closure.json')
    # Noiseless full time-parameter fit tests the actual optimizer, not just projection.
    fitted=fit_erp(truth,.203125,3,.01,quick=quick)
    save_json(dict(**fit_record(fitted),true_theta=theta,true_A=A,true_D=D,
        prediction_RMSE=float(np.sqrt(np.mean((truth-fitted.predict())**2)))),dest/'noiseless_parameter_recovery.json')
    for name,x in scenarios.items():
        ds=Dataset(name,x,y,np.arange(1,n+1),blocks,np.full(n,.203125),np.arange(n)*2200,
                   np.zeros((n,3)),x[:,:,:64])
        rows,_,_=nested_cv(ds,quick,False)
        yy=np.array([r['truth'] for r in rows]);pp=np.array([r['prediction'] for r in rows]);ss=np.array([r['score'] for r in rows])
        results.append(dict(scenario=name,**metrics(yy,pp,ss)))
        save_csv(rows,dest/f'{name}_predictions.csv')
    np.savez_compressed(dest/'ground_truth.npz',theta=theta,A=A,D=D,h=h,truth=truth,
                        y=y,**scenarios)
    save_csv(results,dest/'scenario_metrics.csv')
    return results
