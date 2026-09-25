"""V2 fixed-prefix behavior with nested EEG selection and label permutations."""
import numpy as np
import pandas as pd
from q3.data import TIME, ROOT, save_csv, save_json
from q3.behavior import prefix_dataset, ordinary_features, smoother, predict_all_labels, rt_metrics, holm
from q3.fit import LAMBDAS
from q3.v2.fit import Bank
from q3.v2.model import candidates, design_bank, states


def state_features(ds,fit):
    c=fit['candidate'];coef=fit['coef'];scale=fit['meta']['channel_scale'];rows=[];audit=[]
    for e,x in zip(ds.events,ds.x):
        h=design_bank([c],e)[0].astype(float)
        mask=(TIME>=0)&(TIME<e['end_s'])
        contributions=h[mask,:,None]*coef[None,:,:]
        D=(contributions/scale[None,None,:]).transpose(0,2,1).reshape(-1,9)
        y=(x[mask]/scale).ravel();n=len(y)
        norm=np.maximum(np.sqrt(np.mean(D*D,axis=0)),1e-6);Z=D/norm
        correction=np.linalg.solve(Z.T@Z/n+np.eye(9),Z.T@(y-D.sum(1))/n)
        amplitude=1+correction/norm
        _,m,p=states(e,c);last=np.flatnonzero(mask)[-1];previous=max(last-13,0)
        at_target=np.searchsorted(TIME,e['target_s'])
        feat=np.r_[m[last]*amplitude[3:5],p[last]*amplitude[6:9],
            (p[last]-p[previous])/(TIME[last]-TIME[previous])*amplitude[6:9],
            (m[last,0]-m[at_target,0])*amplitude[3]]
        rows.append(feat);audit.append(dict(trial_id=e['trial_id'],prefix_end_s=e['end_s'],amplitudes=amplitude))
    return np.array(rows),audit


def build_prediction_maps(ds,bank,progress=None):
    blocks=ds.blocks;ordinary=ordinary_features(ds);maps={};feature_rows=[];choices=[]
    for held in range(5):
        train=np.flatnonzero(blocks!=held);test=np.flatnonzero(blocks==held);train_blocks=sorted(set(blocks[train]))
        fit=bank.select(train_blocks,'N2');features,audit=state_features(ds,fit)
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
            inner_fit=bank.select(subset,'N2')
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


def run_behavior(quick=False,n_perm=1999,n_boot=2000,progress=None):
    from q3.data import SEED
    rng=np.random.default_rng(SEED+7);predrows=[];feature_rows=[];choice_rows=[];auditrows=[];metrics=[];nullrows=[];intervals=[]
    pooled=[]
    for key in ('A2','B2'):
        ds,audit=prefix_dataset(key);auditrows+=audit
        if progress:progress(f'{key} RT: building prefix candidate bank ({len(ds.events)} trials)')
        bank=Bank(ds,[c for c in candidates() if c.model=='N2']);maps,features,choices=build_prediction_maps(ds,bank,progress)
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


def run(out,progress):
    result=run_behavior(False,1999,2000,progress)
    for key,value in result.items():
        if key in ('choices','null_distribution'):save_json(out/f'behavior_{key}.json',value)
        else:save_csv(out/f'behavior_{key}.csv',value)
    # Paired V1/V2 comparison uses the identical prefix cohort and observed RT.
    old=pd.read_csv(ROOT/'q3_result/behavior_predictions.csv')
    new=pd.DataFrame(result['predictions'])
    merged=new.merge(old,on=['dataset','trial_id','block'],suffixes=('_v2','_v1'),validate='one_to_one')
    assert len(merged)==len(new)==len(old)
    np.testing.assert_allclose(merged.rt_s_v1,merged.rt_s_v2)
    rng=np.random.default_rng(202609254);comparisons=[]
    for key in ('A2','B2','pooled'):
        df=merged if key=='pooled' else merged[merged.dataset==key]
        a=abs(df.rt_s_v2-df.cognitive_v2).to_numpy();b=abs(df.rt_s_v1-df.cognitive_v1).to_numpy()
        bs=[]
        for _ in range(2000):
            ix=[]
            for record in df.dataset.unique():
                for block in rng.integers(0,5,5):
                    ix.extend(np.flatnonzero((df.dataset==record)&(df.block==block)))
            bs.append(np.mean((b-a)[ix]))
        lo,hi=np.quantile(bs,[.025,.975])
        comparisons.append(dict(dataset=key,V1_MAE=b.mean(),V2_MAE=a.mean(),gain=(b-a).mean(),ci_low=lo,ci_high=hi))
    save_csv(out/'behavior_v1_comparison.csv',comparisons)
