#!/usr/bin/env python3
"""Exploratory failure diagnosis. Does not modify the frozen primary analysis.

Fixed-kernel counterfactuals isolate coefficient shrinkage, temporal span, and
affine trends. They are not a newly selected, confirmatory model.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/q2-diagnostic-mpl')
from pathlib import Path
import json
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from q2model.data import (B,FS,TIMES,WEIGHTS,FIT_MASK,NAMES,Dataset,means,modes,
                         descriptive_data,save_csv,save_json,sha,window_features)
from q2model.model import KernelFactory,coefficients,predict_erp,weighted_error,waveform_metrics
from q2model.validation import classify,metrics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'Q2_result'


def load_ds(key):
    with np.load(BASE/'audit'/f'{key}_independent_guard24.npz') as w:
        return Dataset(key,*[w[k].copy() for k in ('x','y','ids','blocks','durations','onset','artifact','prestim_only')])


def span_fit(erp,h,lam_a=0.,lam_d=0.):
    """Penalty convention identical to production: SSE + la|A|² + ld|D|²."""
    z=np.einsum('cm,dct->dmt',B,erp)
    targets=[(z[0]+z[1])/2,(z[1]-z[0])/2]
    gram=h.T@(WEIGHTS[:,None]*h)
    out=[]
    for t,penalty in zip(targets,(lam_a,lam_d)):
        out.append(np.linalg.lstsq(gram+penalty/2*np.eye(h.shape[1]),h.T@(WEIGHTS[:,None]*t.T),rcond=1e-12)[0].T)
    return predict_erp(*out,h),out


def affine_basis():
    h=np.column_stack([np.ones(len(TIMES)),TIMES])
    return h/np.sqrt(np.sum(WEIGHTS[:,None]*h*h,axis=0))


def feature_projection(x,h,ridge=.1):
    inv=np.linalg.pinv(h.T@(WEIGHTS[:,None]*h)+ridge*np.eye(h.shape[1]),rcond=1e-12)
    return np.einsum('nmt,tr,rs->nms',modes(x),WEIGHTS[:,None]*h,inv).reshape(len(x),-1)


def detrend(x,prestim=False):
    idx=TIMES<0 if prestim else np.ones(len(TIMES),bool)
    design=np.column_stack([np.ones(len(TIMES)),TIMES])
    co=np.einsum('st,nct->ncs',np.linalg.pinv(design[idx]),x[...,idx])
    return x-np.einsum('ts,ncs->nct',design,co)


def detail(key,stage,name,obs,pred):
    rows,m=waveform_metrics(obs,pred)
    m.update(dataset=key,stage=stage,variant=name,
             negative_R2=sum(r['R2']<0 for r in rows),mean_R2=np.mean([r['R2'] for r in rows]))
    delta=obs[1]-obs[0]; dp=pred[1]-pred[0]
    m['weighted_contrast_MSE']=float(np.sum(WEIGHTS[None,:]*(delta-dp)**2)/3)
    return m


def broader_fit(obs,duration,theta,rank,seed=41):
    """Same rank, no ridge; broaden engineering limits only. In-sample diagnostic."""
    fac=KernelFactory(duration)
    low=np.array([0.,10.,30.,80.][:rank+1]);hi=np.array([200.,300.,1200.,2400.][:rank+1])
    span=hi-low
    def loss(u):
        t=low+u*span
        h=fac.kernel(t,rank);p,_=span_fit(obs,h)
        return weighted_error(obs,p)
    con=[{'type':'ineq','fun':lambda u:(low+u*span)[2]-(low+u*span)[1]-10}]
    if rank==3:con.append({'type':'ineq','fun':lambda u:(low+u*span)[3]-(low+u*span)[2]-10})
    rng=np.random.default_rng(seed)
    starts=[(theta-low)/span]
    while len(starts)<7:
        t=low+rng.random(len(low))*span
        if t[2]>=t[1]+10 and (rank==2 or t[3]>=t[2]+10):starts.append((t-low)/span)
    scale=max(loss(starts[0]),1);traces=[]
    best=(loss(starts[0]),theta.copy())
    for s in starts:
        r=minimize(lambda u:loss(u)/scale,s,method='SLSQP',bounds=[(0,1)]*len(low),constraints=con,
                   options={'maxiter':150,'ftol':1e-9,'eps':1e-5})
        t=low+r.x*span;v=loss(r.x)
        traces.append(dict(success=bool(r.success),loss=v,theta=t.tolist(),message=str(r.message)))
        if r.success and v<best[0]:best=(v,t)
    h=fac.kernel(best[1],rank);p,_=span_fit(obs,h)
    return p,dict(theta=best[1],starts=traces,any_converged=any(t['success'] for t in traces),
                  status='converged' if any(t['success'] for t in traces) else 'all_failed_original_theta_retained',
                  condition=np.linalg.cond(h*np.sqrt(WEIGHTS[:,None])))


def descriptive_diagnosis(out):
    rows=[];traces={};checks=[]
    for key in NAMES:
        for stage in ('before','v7'):
            path=BASE/'fits'/key/stage
            meta=json.loads((path/'model.json').read_text())
            w=np.load(path/'waveforms.npz');obs=w['observed'];h=w['h'];lam=meta['lambda_A']
            # Independent least-squares solution of the normal equations.
            original,_=span_fit(obs,h,lam,10*lam)
            checks.append(dict(dataset=key,stage=stage,independent_solver_max_error=float(np.max(abs(original-w['predicted'])))))
            variants={'original':w['predicted'],'same_kernel_no_ridge':span_fit(obs,h)[0],
                      'same_kernel_equal_direction_penalty':span_fit(obs,h,lam,lam)[0],
                      'affine_only':span_fit(obs,affine_basis())[0],
                      'same_kernel_plus_affine':span_fit(obs,np.column_stack([h,affine_basis()]))[0]}
            wide,trace=broader_fit(obs,meta['diagnostics']['duration_ms']/1000,w['theta'],meta['rank'])
            variants['same_rank_broader_times_no_ridge']=wide;traces[f'{key}_{stage}']=trace
            for name,pred in variants.items():
                record=detail(key,stage,name,obs,pred)
                record['optimization_status']=trace['status'] if name=='same_rank_broader_times_no_ridge' else 'analytic'
                rows.append(record)
            np.savez_compressed(out/f'{key}_{stage}_counterfactual.npz',observed=obs,**variants)
            print('descriptive',key,stage,flush=True)
    save_csv(rows,out/'descriptive_counterfactuals.csv');save_json(traces,out/'broader_time_optimization.json')
    save_csv(checks,out/'independent_normal_equations.csv')


def temporal_diagnosis(ds,out):
    rows=[];block_delta=[];signal=[]
    for block in range(5):
        sub=ds.subset(ds.blocks==block);obs=means(sub.x,sub.y);delta=obs[1]-obs[0];block_delta.append(delta)
        var=[np.var(sub.x[sub.y==d],axis=0,ddof=1)/sum(sub.y==d) for d in (-1,1)]
        energy=float(np.sum(delta**2*WEIGHTS)/3);noise=float(np.sum((var[0]+var[1])*WEIGHTS)/3)
        signal.append(dict(dataset=ds.key,block=block,n_left=int(sum(sub.y==-1)),n_right=int(sum(sub.y==1)),
                           observed_delta_energy=energy,estimated_sampling_noise_energy=noise,
                           noise_to_observed_energy=noise/energy))
    dz=np.einsum('cm,dct->dmt',B,np.stack(block_delta))
    corr=[]
    for i in range(5):
        for j in range(i+1,5):
            for k,label in enumerate(['common','midline','lateral']):
                a=dz[i,k]*np.sqrt(WEIGHTS);b=dz[j,k]*np.sqrt(WEIGHTS)
                corr.append(dict(dataset=ds.key,block_i=i,block_j=j,mode=label,weighted_cosine=float(a@b/np.sqrt((a@a)*(b@b)))))
    return signal,corr


def heldout_diagnosis(out):
    waves=[];preds=[];stability=[];noise=[];solves=[];fold_fits=[];budgets=[];influence=[]
    for key in NAMES:
        ds=load_ds(key);n,c=temporal_diagnosis(ds,out);noise+=n;stability+=c
        for block in range(5):
            tr=ds.subset(ds.blocks!=block);te=ds.subset(ds.blocks==block)
            path=BASE/'validation/blocked'/key
            meta=json.loads((path/f'fold_{block}.json').read_text())['fit'];w=np.load(path/f'fold_{block}.npz')
            h=w['h'];lam=meta['lambda_A'];train=means(tr.x,tr.y);test=means(te.x,te.y)
            pred,ad=span_fit(train,h,lam,10*lam)
            solves.append(dict(dataset=key,fold=block,max_error=float(np.max(abs(pred-w['prediction_M1'])))))
            oracle,_=span_fit(test,h)
            total=weighted_error(test,pred);approx=weighted_error(test,oracle);coef=weighted_error(oracle,pred)
            budgets.append(dict(dataset=key,fold=block,total_error=total,span_approximation_error=approx,
                                coefficient_transfer_error=coef,pythagorean_gap=total-approx-coef,
                                coefficient_transfer_share=coef/total))
            # Influence on the TEST ERP is descriptive only, never trial exclusion.
            for j,(x,y,i) in enumerate(zip(te.x,te.y,te.ids)):
                d=(int(y)+1)//2;count=sum(te.y==y);change=(x-test[d])/(count-1)
                influence.append(dict(dataset=key,fold=block,trial_id=int(i),cue=int(y),
                    leave_one_out_ERP_change_RMS=float(np.sqrt(np.sum(change**2*WEIGHTS)/3)),
                    condition_ERP_RMS=float(np.sqrt(np.sum(test[d]**2*WEIGHTS)/3)),
                    PTP=float(te.artifact[j,1])))
            # A controlled change to the basis/coefficient fit; no held-out tuning.
            configs={'original':(h,lam,10*lam),'same_kernel_no_ridge':(h,0,0),
                     'same_kernel_equal_direction_penalty':(h,lam,lam),
                     'same_kernel_plus_affine':(np.column_stack([h,affine_basis()]),lam,10*lam),
                     'affine_only':(affine_basis(),lam,10*lam)}
            for name,(basis,la,ld) in configs.items():
                p,_=span_fit(train,basis,la,ld)
                for split,target in [('train',train),('test',test)]:
                    waves.append(dict(fold=block,split=split,**detail(key,'independent',name,target,p)))
            # Test-label-informed projection ONLY to quantify the basis approximation
            # floor. Never fed to feature learning, selection, or a predictor.
            for name,basis in [('test_oracle_same_span',h),('test_oracle_span_plus_affine',np.column_stack([h,affine_basis()]))]:
                p,_=span_fit(test,basis)
                waves.append(dict(fold=block,split='oracle_not_prediction',**detail(key,'independent',name,test,p)))
            variants={
                'original':(feature_projection(tr.x,h),feature_projection(te.x,h)),
                'projection_no_ridge':(feature_projection(tr.x,h,0),feature_projection(te.x,h,0)),
                'window_means':(window_features(tr.x),window_features(te.x)),
                'full_epoch_detrended_kernel':(feature_projection(detrend(tr.x),h),feature_projection(detrend(te.x),h)),
                'prestim_detrended_windows':(window_features(detrend(tr.x,True)),window_features(detrend(te.x,True))),
                'full_epoch_detrended_windows':(window_features(detrend(tr.x)),window_features(detrend(te.x))),
                'early_window_only':(window_features(tr.x,windows=[(.05,.25)]),window_features(te.x,windows=[(.05,.25)]))}
            for name,(ft,fe) in variants.items():
                p,s,_,_=classify(ft,tr.y,fe)
                preds.extend(dict(dataset=key,fold=block,trial_id=int(i),truth=int(y),variant=name,
                                  prediction=int(v),score=float(z)) for i,y,v,z in zip(te.ids,te.y,p,s))
            fold_fits.append(dict(dataset=key,fold=block,rank=meta['rank'],lam=lam,
                                 condition=meta['diagnostics']['condition'],theta=meta['theta'],
                                 bound_hits=meta['diagnostics']['bound_hits'],
                                 relative_refinement_gain=(meta['diagnostics']['grid_objective']-meta['diagnostics']['objective'])/meta['diagnostics']['grid_objective']))
        print('heldout',key,flush=True)
    save_csv(waves,out/'heldout_waveforms.csv');save_csv(preds,out/'counterfactual_predictions.csv')
    save_csv(stability,out/'block_direction_similarity.csv');save_csv(noise,out/'block_noise_budget.csv')
    save_csv(solves,out/'heldout_reproduction.csv');save_json(fold_fits,out/'original_fold_parameters.json')
    save_csv(budgets,out/'prediction_error_decomposition.csv');save_csv(influence,out/'trial_influence.csv')
    df=pd.DataFrame(preds)
    save_csv([dict(dataset=k,variant=v,**metrics(g.truth,g.prediction,g.score)) for (k,v),g in df.groupby(['dataset','variant'])],out/'counterfactual_classification.csv')


def make_figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from q2model.report import BLUE,ORANGE,GRAY
    names={'original':'原模型','same_kernel_no_ridge':'同核去收缩','same_rank_broader_times_no_ridge':'同阶放宽时间范围','same_kernel_plus_affine':'同核加截距和趋势','affine_only':'仅截距和趋势'}
    d=pd.read_csv(out/'descriptive_counterfactuals.csv');d=d[d.stage=='v7']
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key in zip(axes.flat,NAMES):
        g=d[d.dataset==key].set_index('variant');base=g.loc['original','weighted_MSE']
        vals=[g.loc[n,'weighted_MSE']/base for n in names]
        invalid=g.loc['same_rank_broader_times_no_ridge','optimization_status']!='converged'
        if invalid:vals[2]=np.nan
        ax.barh(list(names.values()),vals,color=[GRAY,BLUE,BLUE,ORANGE,ORANGE]);ax.invert_yaxis();ax.set_title(NAMES[key]);ax.set_xlim(0,1.1);ax.set_xlabel('加权拟合误差 / 原模型误差')
        if invalid:ax.text(.03,2,'优化未收敛，暂不评价',va='center',color=GRAY)
    fig.suptitle('描述性拟合失败的受控拆解\n同一组 V7 条件 ERP；增加自由度的训练内收益不代表泛化提升')
    for ext in ('png','pdf'):fig.savefig(out/f'01_拟合原因拆解.{ext}',dpi=170)
    plt.close(fig)
    w=pd.read_csv(out/'heldout_waveforms.csv');fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,metric,label in zip(axes,['weighted_MSE','weighted_contrast_MSE'],['整体波形','右减左差分']):
        for offset,(variant,title,color,marker) in enumerate([('same_kernel_no_ridge','同核去收缩',BLUE,'o'),('same_kernel_plus_affine','同核加趋势',ORANGE,'s'),('affine_only','仅趋势',GRAY,'^')]):
            a=w[(w.split=='test')&(w.variant==variant)].groupby('dataset')[metric].mean()
            b=w[(w.split=='test')&(w.variant=='original')].groupby('dataset')[metric].mean()
            ax.scatter(np.arange(4)+(offset-1)*.14,a/b,label=title,color=color,marker=marker)
        ax.set_xticks(range(4),list(NAMES))
        ax.axhline(1,color=GRAY,ls='--');ax.set_title(label);ax.set_ylabel('测试误差 / 原模型测试误差');ax.legend()
    fig.suptitle('训练内拟合改善能否迁移到留出时段？\n固定原训练核、相同测试试次；探索性对照，未据此更换主检验')
    for ext in ('png','pdf'):fig.savefig(out/f'02_留出误差对照.{ext}',dpi=170)
    plt.close(fig)
    s=pd.read_csv(out/'block_direction_similarity.csv');fig,axes=plt.subplots(1,4,figsize=(15,4),layout='constrained')
    for ax,key in zip(axes,NAMES):
        z=np.eye(5)
        for r in s[(s.dataset==key)&(s['mode']=='lateral')].itertuples():z[r.block_i,r.block_j]=z[r.block_j,r.block_i]=r.weighted_cosine
        from matplotlib.colors import LinearSegmentedColormap
        im=ax.imshow(z,vmin=-1,vmax=1,cmap=LinearSegmentedColormap.from_list('direction',[BLUE,'#ffffff',ORANGE]))
        for i in range(5):
            for j in range(5):ax.text(j,i,f'{z[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(z[i,j])>.6 else 'black')
        ax.set_title(NAMES[key]);ax.set_xticks(range(5),range(1,6));ax.set_yticks(range(5),range(1,6));ax.set_xlabel('时间块');ax.set_ylabel('时间块')
    fig.colorbar(im,ax=axes,shrink=.7,label='加权余弦相似度')
    fig.suptitle('两侧差分模态中的方向 ERP：跨时间块一致性\n负值表示整体方向反转；小样本估计噪声也可导致反转')
    for ext in ('png','pdf'):fig.savefig(out/f'03_方向差异稳定性.{ext}',dpi=170)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=BASE/'diagnosis');a=p.parse_args();out=a.output
    out.mkdir(parents=True,exist_ok=True)
    descriptive_diagnosis(out);heldout_diagnosis(out);make_figures(out)
    baseline_manifest=json.loads((BASE/'manifest.json').read_text())
    save_json(dict(status='complete',purpose='exploratory_diagnosis_not_confirmatory_replacement',
        frozen_primary_signature=baseline_manifest['signature'],source_sha256=sha(Path(__file__)),
        numerical_source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'Q2/q2model').glob('*.py'))},
        limitations=['Counterfactuals are not prespecified primary tests','Oracle fits use test labels for approximation diagnosis only','Full-epoch detrending is offline and can remove neural slow activity'],
        output_hashes={p.name:sha(p) for p in out.iterdir() if p.is_file() and p.name!='manifest.json'}),out/'manifest.json')
    print('DONE',out,flush=True)


if __name__=='__main__':main()
