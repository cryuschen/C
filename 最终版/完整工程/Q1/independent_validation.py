"""Blocked truth-known recovery tests and training-selected conservative correction.

All four methods see identical train/test trials and contamination. Synthetic
and semisynthetic targets have separate identities; neither proves brain sources.
"""
from pathlib import Path
from dataclasses import replace
import json,hashlib
import numpy as np
import pandas as pd
import EEG_P300_artifact_correction_v7 as v7
from q2.q2model.data import independent_data, NAMES

ROOT=Path(__file__).resolve().parents[1]
STRENGTHS=(0.,.05,.10,.15,.25)
GATES=(0.,1.,2.)
SEEDS=(20260927,20260928,20260929)
LEVELS=(0.,.5,1.,2.)


def reference(x,y):
    scores=v7.detect_bad_trials(np.zeros_like(x),x)[1]
    return v7.build_clean_reference_model(x,y,np.zeros(len(x),bool),scores)


def deviation(x,y,model):
    tpl,sv,sh,_=model
    return np.array([np.sqrt(np.mean(((a-tpl[c]) / np.maximum(sv[c][None,:],1.))**2)) for a,c in zip(x,y)])


def conservative_output(x,y,model,candidate,strength,gate,threshold,candidate_output=None):
    if strength==0:return x.copy()
    candidate_output=v7.apply_model(x,y,model,candidate) if candidate_output is None else candidate_output
    score=deviation(x,y,model)
    weight=np.maximum(0.,1-gate*threshold/np.maximum(score,1e-12)) if gate else np.ones(len(x))
    return x+strength*weight[:,None,None]*(candidate_output-x)


def shrink(x,y,model,alpha):
    return (1-alpha)*x+alpha*np.stack([model[0][c] for c in y])


def fit(x,y,blocks,seed):
    # One fixed inner validation block (last available), never outer test samples.
    val=blocks==np.max(blocks);train=~val
    model=reference(x[train],y[train]);candidate,_=v7.select_candidate(x[train],y[train],seed)
    threshold=float(np.quantile(deviation(x[train],y[train],model),.9))
    options=[('shrink',a,0.) for a in STRENGTHS]+[('gated',a,g) for a in STRENGTHS for g in GATES]
    rows=[]
    for level,weight in ((0.,.5),(1.,.25),(2.,.25)):
        noisy=v7.inject(x[val],y[val],seed+700,level)
        corrected=v7.apply_model(noisy,y[val],model,candidate)
        for method,alpha,gate in options:
            out=shrink(noisy,y[val],model,alpha) if method=='shrink' else conservative_output(noisy,y[val],model,candidate,alpha,gate,threshold,corrected)
            err=v7.known_errors(x[val],out,y[val])
            scale=max(np.sqrt(np.mean(x[val]**2)),1e-9)
            rows.append(dict(method=method,alpha=alpha,gate=gate,level=level,weight=weight,
                             contrast_normalized=err['contrast_RMSE']/scale,**err))
    frame=pd.DataFrame(rows);selections={}
    for method in ('shrink','gated'):
        sub=frame[frame.method==method]
        losses=sub.assign(loss=sub.weight*sub.objective).groupby(['alpha','gate']).loss.sum()
        if method=='gated':
            zero=sub[sub.level==0].set_index(['alpha','gate'])
            # Engineering guardrail fixed before any outer recovery results.
            permitted=(zero.normalized_RMSE<=.05)&(zero.contrast_normalized<=.05)
            losses=losses[permitted.reindex(losses.index)]
        alpha,gate=min(losses.index,key=lambda k:(losses.loc[k],k[0],-k[1]))
        selections[method]=dict(alpha=float(alpha),gate=float(gate),inner_loss=float(losses.loc[(alpha,gate)]))
    fullmodel=reference(x,y);fullcandidate,_=v7.select_candidate(x,y,seed)
    return dict(model=fullmodel,candidate=fullcandidate,threshold=float(np.quantile(deviation(x,y,fullmodel),.9)),
                mean_delta=v7.training_mean_delta(x,y,fullmodel,fullcandidate),selected=selections,
                inner_records=rows,inner_validation_block=int(np.max(blocks)))


def synthetic(y,seed):
    rng=np.random.default_rng(seed);t=v7.T/1000;x=[]
    for cue in y:
        latency=.34+rng.normal(0,.025)
        common=(45+rng.normal(0,5))*np.exp(-.5*((t-latency)/.065)**2)
        contrast=cue*15*np.exp(-.5*((t-.41)/.11)**2)
        early=-12*np.exp(-.5*((t-.13)/.035)**2)
        wave=np.array([1.,.8,.9])[:,None]*(common+early)+np.array([.1,-1,1])[:,None]*contrast
        # This small smooth component is part of the known target, not injected noise.
        wave+=rng.normal(size=(3,1))*3*np.sin(2*np.pi*3*t)
        x.append(wave)
    return v7.robust_baseline_correct(np.array(x))


def unseen_contamination(x,y,seed,level,family):
    if level==0:return x.copy()
    rng=np.random.default_rng(seed);t=v7.T/1000;out=x.copy()
    scale=max(float(np.median(np.std(x,axis=-1))),1.)
    for i,cue in enumerate(y):
        center=rng.uniform(.02,.45);u=t-center;width=rng.uniform(.06,.2)
        if family=='asymmetric_burst':
            envelope=np.where(u>=0,np.exp(-np.maximum(u,0)/width),0.)
            temporal=envelope*(1+.45*np.sin(2*np.pi*19*u))
            spatial=rng.uniform(-1.5,1.5,3);sign=rng.choice([-1,1])
        elif family=='direction_step':
            temporal=np.clip(u/.04,0,1)*np.exp(-np.maximum(u,0)/.4)
            spatial=np.array([.25,-1,1])*rng.uniform(.7,1.3,3);sign=cue
        else:raise ValueError(family)
        out[i]+=level*scale*rng.uniform(2,5)*sign*spatial[:,None]*temporal
    return v7.robust_baseline_correct(out)


def run(out):
    out=Path(out);out.mkdir(exist_ok=True,parents=True)
    rows=[];choices=[];truth_arrays={};real_arrays={};inner=[]
    for key in NAMES:
        ds=independent_data(ROOT,key)
        for origin in ('semisynthetic','synthetic'):
            x=ds.x if origin=='semisynthetic' else synthetic(ds.y,1700+ord(key[0])+int(key[1]))
            truth_arrays[f'{key}_{origin}_target']=x
            truth_arrays[f'{key}_{origin}_cues']=ds.y
            truth_arrays[f'{key}_{origin}_ids']=ds.ids
            for held in range(5):
                train=ds.blocks!=held;test=~train
                fitted=fit(x[train],ds.y[train],ds.blocks[train],100+held)
                model=fitted['model'];candidate=fitted['candidate'];chosen=fitted['selected']
                choices.append(dict(dataset=key,origin=origin,held_block=held,train_ids=ds.ids[train].tolist(),test_ids=ds.ids[test].tolist(),
                                    candidate=candidate,threshold=fitted['threshold'],selected=chosen,
                                    inner_validation_block=fitted['inner_validation_block']))
                inner.extend(dict(dataset=key,origin=origin,held_block=held,**r) for r in fitted['inner_records'])
                for family in ('asymmetric_burst','direction_step'):
                    for seed in SEEDS:
                        for level in LEVELS:
                            noisy=unseen_contamination(x[test],ds.y[test],seed+held,level,family)
                            corrected=v7.apply_model(noisy,ds.y[test],model,candidate)
                            outputs={'uncorrected':noisy,
                                     'V7':v7.apply_v7(noisy,ds.y[test],model,candidate,fitted['mean_delta']),
                                     'shrink':shrink(noisy,ds.y[test],model,chosen['shrink']['alpha']),
                                     'gated':conservative_output(noisy,ds.y[test],model,candidate,chosen['gated']['alpha'],chosen['gated']['gate'],fitted['threshold'],corrected)}
                            for method,a in outputs.items():
                                rows.append(dict(dataset=key,origin=origin,held_block=held,family=family,seed=seed,level=level,method=method,n=int(test.sum()),**v7.known_errors(x[test],a,ds.y[test])))
                            if seed==SEEDS[0] and level==1 and held==0:
                                for method,a in outputs.items():truth_arrays[f'{key}_{origin}_{family}_{method}']=a
                                truth_arrays[f'{key}_{origin}_{family}_test_target']=x[test]
                if origin=='semisynthetic':
                    for method,a in {'before':x[test],'gated':conservative_output(x[test],ds.y[test],model,candidate,chosen['gated']['alpha'],chosen['gated']['gate'],fitted['threshold'])}.items():
                        real_arrays.setdefault(f'{key}_{method}',np.empty_like(x))[test]=a
            print('Q1 independent validation:',key,origin,flush=True)
        real_arrays[f'{key}_ids']=ds.ids;real_arrays[f'{key}_cues']=ds.y;real_arrays[f'{key}_blocks']=ds.blocks
    pd.DataFrame(rows).to_csv(out/'独立验证_逐折指标.csv',index=False)
    pd.DataFrame(inner).to_csv(out/'独立验证_内层选型.csv',index=False)
    (out/'独立验证_选择记录.json').write_text(json.dumps(choices,ensure_ascii=False,indent=2))
    np.savez_compressed(out/'独立验证_真值与示例.npz',times_ms=v7.T,**truth_arrays)
    np.savez_compressed(out/'独立验证_保守处理波形.npz',times_ms=v7.T,**real_arrays)
    frame=pd.DataFrame(rows);summary=frame.groupby(['origin','family','level','method'])[['normalized_RMSE','contrast_RMSE','positive_mean_error','latency_error_ms']].mean().reset_index()
    summary.to_csv(out/'独立验证_汇总.csv',index=False)
    render(out,summary)
    (out/'独立验证_清单.json').write_text(json.dumps({'status':'complete','cohort':275,'folds':'independent_blocks',
        'not_interchangeable_with_371_trial_V7':True,'semisynthetic_target':'preprocessed real EEG; may contain original artifacts',
        'synthetic_target':'generated waveforms; no physiological validation','guardrails':{'zero_NRMSE':.05,'zero_contrast_normalized':.05},
        'source_hashes':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'EEG_P300_artifact_correction_v7.py',ROOT/'q2/q2model/data.py']},
        'configuration':str(ROOT/'优化计划/本轮实验配置.json')},ensure_ascii=False,indent=2))


def render(out,summary):
    from q2.q2model.reporting import setup
    import matplotlib.pyplot as plt
    setup();fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for i,origin in enumerate(('semisynthetic','synthetic')):
        for j,family in enumerate(('asymmetric_burst','direction_step')):
            part=summary[(summary.origin==origin)&(summary.family==family)]
            method_names={'uncorrected':'未校正','V7':'基准方法','shrink':'模板收缩','gated':'保守门控'}
            for method,g in part.groupby('method'):axes[i,j].plot(g.level,g.normalized_RMSE,'o-',label=method_names.get(method,method))
            origin_name={'semisynthetic':'半合成','synthetic':'全合成'}.get(origin,origin)
            family_name={'asymmetric_burst':'非对称高频扰动','direction_step':'方向相关阶跃'}.get(family,family)
            axes[i,j].set(title=f'{origin_name} {family_name}',xlabel='伪影强度',ylabel='归一化恢复误差')
            axes[i,j].legend(fontsize=8)
    fig.suptitle('问题一降噪方法的独立污染恢复检验',fontsize=14)
    target=out/'独立验证_污染恢复.png';temporary=out/'.__plot_tmp.png'
    fig.savefig(temporary,dpi=180);target.unlink(missing_ok=True);temporary.replace(target);plt.close(fig)
    (out/'独立验证_报告.md').write_text('# 第一问独立恢复验证\n\n三通道原始数据在独立时间块内预处理；275次队列与历史371次分开。所有方法共用训练、测试、污染与目标。\n\n'
        '训练只使用原有污染生成器，留出测试使用非对称高频衰减和方向相关缓慢阶跃。合成目标有已知形状与潜伏期变化；半合成目标是可能仍含原伪影的实测背景。\n\n'
        '保守方案在训练内选择校正比例与门控，同时限制零注入背景和差分误差各不超过0.05归一化单位。简单模板收缩也在训练内选强度。没有根据测试组选择方法。\n\n'
        +__import__('q2.q2model.reporting',fromlist=['md_table']).md_table(summary,list(summary))+'\n\n表内为四记录、五块、三个注入种子的等权均值；种子重复不是新增受试者。合格潜伏期对数保存在逐折表。图见独立验证_污染恢复.png。\n',encoding='utf8')
