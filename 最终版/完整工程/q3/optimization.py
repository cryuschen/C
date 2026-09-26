"""New conditional parameter profiles, refits and robustness checks for Q3.

Frozen V2 predictions remain separate. All new fits use real raw-data V3 windows,
training-only parameter selection, and stimulus-only information.
"""
from pathlib import Path
from itertools import product
import json,hashlib
import numpy as np
import pandas as pd
from .common import ROOT,KEYS,TIME,Trials,save_csv,save_json,stage_masks
from .events import build_trials
from .model import Candidate,design_bank,predict,neural_design
from .fit import Bank
from .decision import simulate,SCENARIOS

TAUS=(.25,.5,1.,2.,4.,8.)


def records(ds,pred,model,**fields):
    ans=[]
    for i,e in enumerate(ds.events):
        for stage,mask in zip(('encoding','maintenance','retrieval'),stage_masks(e['target_s'],e['end_s'])):
            if mask.any():ans.append(dict(dataset=ds.key,trial_id=int(ds.ids[i]),block=int(ds.blocks[i]),model=model,
                stage=stage,n_samples=int(mask.sum()),MSE=float(np.mean((ds.x[i,mask]-pred[i,mask])**2)),**fields))
    return ans


def paired(frame,groupcols):
    rows=[];rng=np.random.default_rng(20260926)
    for keys,g in frame.groupby(groupcols):
        keys=keys if isinstance(keys,tuple) else (keys,)
        block=g.groupby(['block','model']).MSE.mean().unstack()
        a=block.N2.to_numpy();b=block.N3.to_numpy();ix=rng.integers(0,len(a),(2000,len(a)))
        s=1-a[ix].mean(1)/b[ix].mean(1)
        rows.append(dict(zip(groupcols,keys),S=1-a.mean()/b.mean(),ci_low=float(np.quantile(s,.025)),ci_high=float(np.quantile(s,.975)),n_blocks=len(a)))
    return rows


def real_evidence(out):
    profiles=[];metrics=[];choices=[];spectra=[];gaps=[];cohorts=[];arrays={}
    params=[Candidate('N2',tm,tp,d,r) for tm,tp,d,r in product(TAUS,(.1,.3,.6),(.1,.2,.4),(1.,10.))]
    params += [Candidate('N3',tau_p=tp,target_duration=d,direction_penalty=r) for tp,d,r in product((.1,.3,.6),(.1,.2,.4),(1.,10.))]
    for key in KEYS:
        ds,_=build_trials(key);bank=Bank(ds,params,design_function=design_bank)
        predictions={m:np.full_like(ds.x,np.nan) for m in ('N2','N3')}
        for held in range(5):
            train=tuple(b for b in range(5) if b!=held);te=np.flatnonzero(ds.blocks==held)
            for tm in TAUS:
                fit=bank.select(train,'N2',tau_m=tm)
                pred=np.stack([predict(ds.events[i],fit['candidate'],fit['coef']) for i in te])
                mse=np.mean([np.mean((ds.x[i,stage_masks(ds.events[i]['target_s'],ds.events[i]['end_s'])[2]]-pred[j,stage_masks(ds.events[i]['target_s'],ds.events[i]['end_s'])[2]])**2) for j,i in enumerate(te)])
                h,y,w=bank.matrix(np.flatnonzero(ds.blocks!=held),fit['selection']['candidate_index'])
                train_loss=float(np.sum(w[:,None]*(y-h@fit['coef'])**2)/3)
                profiles.append(dict(dataset=key,held_block=held,tau_m=tm,train_MSE=train_loss,held_retrieval_MSE=mse,
                    inner_huber=fit['selection']['inner_huber'],**{k:v for k,v in fit['candidate'].record().items() if k not in ('model','tau_m')}))
            for model in ('N2','N3'):
                fit=bank.select(train,model)
                for i in te:predictions[model][i]=predict(ds.events[i],fit['candidate'],fit['coef'])
                choices.append(dict(dataset=key,held_block=held,model=model,parameters=fit['candidate'].record(),
                    coef=fit['coef'],selection=fit['selection'],train_ids=fit['train_ids'],test_ids=ds.ids[te].tolist()))
                sv=fit['meta']['design_singular_values'];positive=sv[sv>1e-8]
                h,_,w=bank.matrix(np.flatnonzero(ds.blocks!=held),fit['selection']['candidate_index'])
                gram=h.T@(w[:,None]*h);sd=np.sqrt(np.maximum(np.diag(gram),1e-20));corr=gram/sd[:,None]/sd[None,:]
                active=sd>1e-8;off=abs(corr[np.ix_(active,active)]-np.eye(active.sum()))
                spectra.append(dict(dataset=key,held_block=held,model=model,active_rank=len(positive),
                    effective_condition=float(positive[0]/positive[-1]),max_column_cosine=float(off.max()),
                    edf=fit['meta']['edf'],singular_values=json.dumps(sv.tolist())))
            print('Q3 new fits',key,'block',held+1,flush=True)
        for model,p in predictions.items():metrics+=records(ds,p,model,identity='new_nested_refit')
        arrays[key+'_x']=ds.x;arrays[key+'_ids']=ds.ids;arrays[key+'_blocks']=ds.blocks
        for model,p in predictions.items():arrays[key+'_'+model]=p
        # Frozen original N2/N3 refit parameters score different endpoints: no re-selection.
        fits=json.loads((out/'reference'/f'{key}_stimulus_choices.json').read_text())
        gap_sets={gap:build_trials(key,gap=gap)[0] for gap in (.05,.1,.15)}
        common=set(ds.ids)
        for other in gap_sets.values():common &= set(other.ids)
        for gap,other in gap_sets.items():
            cohorts.append(dict(dataset=key,gap_s=gap,n_available=len(other.ids),n_common=len(common),excluded_from_common=sorted(set(other.ids)-common)))
            ix=np.array([i for i,t in enumerate(other.ids) if t in common]);ev=[other.events[i] for i in ix]
            subset=Trials(key,ev,other.x[ix],other.baseline_scale[ix],other.mode,gap)
            for model in ('N2','N3'):
                pp=[]
                for e in ev:
                    f=next(f for f in fits if f['model']==model and f['held_block']==e['block'])
                    assert e['trial_id'] not in f['train_ids']
                    pp.append(predict(e,Candidate(**f['parameters']),np.array(f['coef'])))
                gaps+=records(subset,np.array(pp),model,gap_s=gap,identity='frozen_V2_new_scoring_window')
        del bank;neural_design.cache_clear()
    save_csv(out/'优化_参数剖面.csv',profiles);save_csv(out/'优化_设计诊断.csv',spectra)
    save_csv(out/'优化_新拟合逐试次.csv',metrics);save_json(out/'优化_新拟合选择.json',choices)
    save_csv(out/'优化_新拟合区间.csv',paired(pd.DataFrame(metrics),['dataset','stage']))
    save_csv(out/'优化_截窗逐试次.csv',gaps);save_csv(out/'优化_截窗区间.csv',paired(pd.DataFrame(gaps),['dataset','stage','gap_s']))
    save_json(out/'优化_截窗队列.json',cohorts)
    np.savez_compressed(out/'优化_新拟合波形.npz',time=TIME,**arrays)


def recovery(out):
    ds,_=build_trials('A2')
    ix=np.concatenate([np.flatnonzero(ds.blocks==b)[:8] for b in range(5)])
    ds=Trials('synthetic_A2_timing',[ds.events[i] for i in ix],ds.x[ix],ds.baseline_scale[ix],ds.mode,ds.gap)
    params=[Candidate('N2',tm,tp,.2,r) for tm,tp,r in product(TAUS,(.1,.3,.6),(1.,10.))]
    rows=[];truth_arrays={};choices=[]
    for tm,seed in product((.5,1.,2.),(31,47,83)):
        rng=np.random.default_rng(seed);truth=Candidate('N2',tm,.3,.2)
        h=np.stack([design_bank([truth],e)[0] for e in ds.events]);coef=rng.normal(size=(9,3));coef[5]=0
        clean=h@coef;scale=100/np.sqrt(np.mean(clean[np.isfinite(ds.x)]**2));clean*=scale;coef*=scale
        white=rng.normal(size=clean.shape)
        from scipy.signal import lfilter
        noise=lfilter([1],[1,-.95],white,axis=1);noise*=100/np.sqrt(np.mean(noise[np.isfinite(ds.x)]**2))
        truth_arrays[f'tau{tm}_seed{seed}_clean']=clean;truth_arrays[f'tau{tm}_seed{seed}_coef']=coef
        for level in (0.,.5,1.):
            x=clean+level*noise;x[~np.isfinite(ds.x)]=np.nan
            synthetic=Trials(ds.key,ds.events,x,ds.baseline_scale,ds.mode,ds.gap)
            bank=Bank(synthetic,params,design_function=design_bank);fit=bank.select((0,1,2,3),'N2')
            te=np.flatnonzero(ds.blocks==4);p=np.stack([predict(ds.events[i],fit['candidate'],fit['coef']) for i in te])
            mask=np.isfinite(ds.x[te])&(TIME[None,:,None]>=0)
            error=np.sqrt(np.mean((p[mask]-clean[te][mask])**2))/np.sqrt(np.mean(clean[te][mask]**2))
            rows.append(dict(true_tau_m=tm,seed=seed,noise=level,tau_m=fit['candidate'].tau_m,tau_p=fit['candidate'].tau_p,
                NRMSE=error,tau_error=fit['candidate'].tau_m-tm,exact_tau=fit['candidate'].tau_m==tm,
                exact_dynamics=fit['candidate'].tau_m==tm and fit['candidate'].tau_p==.3))
            choices.append(dict(true_tau_m=tm,seed=seed,noise=level,parameters=fit['candidate'].record(),
                train_ids=fit['train_ids'],test_ids=ds.ids[te].tolist(),selection=fit['selection']))
            del bank
        print('Q3 recovery truth tau',tm,'seed',seed,flush=True)
    save_csv(out/'优化_恢复网格.csv',rows);save_json(out/'优化_恢复选择.json',choices)
    np.savez_compressed(out/'优化_恢复真值.npz',time=TIME,ids=ds.ids,blocks=ds.blocks,**truth_arrays)
    neural_design.cache_clear()


def matching(out):
    rows=[];trials=[]
    for name,label,c,g,s,d in SCENARIOS:
        sm,rr,_,_=simulate(c,gamma=g,sigma=s,deadline=d,evidence_mode='matching')
        rows.append(dict(scenario=name,label=label,**sm))
        trials.extend(dict(scenario=name,**r) for r in rr)
    save_csv(out/'优化_目标匹配仿真.csv',rows);save_csv(out/'优化_目标匹配试次.csv',trials)


def render(out):
    from q2.q2model.reporting import setup
    from .report import table
    import matplotlib.pyplot as plt
    setup();p=pd.read_csv(out/'优化_参数剖面.csv');fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,key in zip(axes.flat,KEYS):
        for held,g in p[p.dataset==key].groupby('held_block'):
            ax.plot(g.tau_m,g.held_retrieval_MSE,'o-',label=f'块{held+1}')
        ax.set(xscale='log',title=key,xlabel='固定记忆时间常数 / s',ylabel='留出提取阶段 MSE')
        from matplotlib.ticker import NullLocator
        ax.set_xticks(TAUS,[str(t) for t in TAUS]);ax.xaxis.set_minor_locator(NullLocator());ax.legend(fontsize=7)
    fig.savefig(out/'优化_参数剖面.png',dpi=180);plt.close(fig)
    r=pd.read_csv(out/'优化_恢复网格.csv');s=pd.read_csv(out/'优化_新拟合区间.csv');gap=pd.read_csv(out/'优化_截窗区间.csv')
    recovery_summary=r.groupby('noise')[['exact_tau','exact_dynamics','NRMSE']].mean().reset_index()
    # Compare new and frozen N2 on exactly matched trials and stages.
    fresh=pd.read_csv(out/'优化_新拟合逐试次.csv')
    old=pd.read_csv(out/'reference/trial_metrics.csv')
    old=old[(old.setting=='stimulus')&(old.model=='N2')].groupby(['dataset','trial_id','block','stage']).MSE.mean().reset_index(name='old_MSE')
    same=fresh[fresh.model=='N2'].merge(old,on=['dataset','trial_id','block','stage'],validate='one_to_one')
    block=same.groupby(['dataset','stage','block'])[['MSE','old_MSE']].mean().reset_index()
    comparisons=[];rng=np.random.default_rng(20260926)
    for (key,stage),g in block.groupby(['dataset','stage']):
        a=g.MSE.to_numpy();b=g.old_MSE.to_numpy();ix=rng.integers(0,len(a),(2000,len(a)))
        boot=1-a[ix].mean(1)/b[ix].mean(1)
        comparisons.append(dict(dataset=key,stage=stage,S=1-a.mean()/b.mean(),ci_low=np.quantile(boot,.025),ci_high=np.quantile(boot,.975)))
    comparisons=pd.DataFrame(comparisons);save_csv(out/'优化_新旧N2比较.csv',comparisons)
    report='# 第三问新增实测检验与参数辨识\n\n本报告为新实验，与V3冻结V2重建分开。记忆时间常数网格扩展至0.25–8秒；其余参数和读出在每个外层训练集内选择。\n\n'
    report+='## 新拟合的主要比较\n\n刺激输入N2相对普通慢趋势N3，目标后阶段；正值表示MSE减少。区间按当前记录的时间块重采样，不包含训练不确定性。\n\n'+table(s[s.stage=='retrieval'])
    report+='\n\n新N2相对原V2 N2在共同试次上的比较：\n\n'+table(comparisons[comparisons.stage=='retrieval'])+'\n\n扩大网格未带来稳定改善，保留原V2/V3主EEG结果，新拟合作为敏感性对照。'
    report+='\n\n## 窗口敏感性\n\n使用原V2固定参数，三种截窗共同试次，窗口改变的是评分目标，不是重新训练。\n\n'+table(gap[gap.stage=='retrieval'])
    report+='\n\n## 参数恢复\n\n3个生成记忆常数×3噪声强度×3随机种子，共27场景。使用A2的40个实测时刻作为合成设计，不是新增EEG样本；目标脉冲固定为已知0.2秒，因此属于条件恢复。匹配网格中的无噪声可恢复也不证明生理来源。\n\n'+table(recovery_summary)
    report+='\n\n参数剖面逐块报告，不能以测试损失最低的记忆常数事后代替训练选择；损失曲线不是参数置信区间。设计矩阵奇异值与列相似性见优化_设计诊断.csv。\n\n'
    report+='## 目标匹配扩展\n\n新增Task-2模拟左右形状，通过Q2形状编码计算记忆与两目标的相似度差，再驱动累积。正确目标侧只由模拟布局导出作评分，不进入匹配函数。实测布局仍缺失，不是实测行为验证。低证据场景只改变增益，噪声保持0.6。\n'
    (out/'优化_机制验证报告.md').write_text(report,encoding='utf8')


def run(out):
    out=Path(out);real_evidence(out);recovery(out);matching(out);render(out)
    save_json(out/'优化_清单.json',dict(status='complete',identity='new_refits_and_conditional_recovery',
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__),ROOT/'q3/model.py',ROOT/'q3/fit.py',ROOT/'q3/decision.py']},
        original_V2_results_changed=False,real_correctness_available=False,
        configuration='优化计划/本轮实验配置.json'))
