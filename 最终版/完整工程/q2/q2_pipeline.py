"""Full Q2 pipeline. Writes only Q2_* files in an already existing directory."""
from __future__ import annotations
import os
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'q2'))
sys.dont_write_bytecode=True
# Existing output directory doubles as a cache location; no new directories.
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'q2_result'))
for _name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[_name]='1'

import argparse
import hashlib
import json
import platform
import time
import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from q2model.data import (NAMES, TIMES, WEIGHTS, FS, independent_data, descriptive_data,
                          load_raw, means, save_json, save_csv)
from q2model.generative import (encode_shapes, build_banks, simulate, observation_sources,
                               common_delta, condition_waves, wave_metrics, delta_metrics, CHANNELS, select_contrast_shrinkage)
from q2model.validation import (FEATURE_SETS, PRIMARY, prepare_decoder, evaluate_all,
                               null_labels, bootstrap_metrics, transfer, metrics, feature_fold, standardize)

SEED=20260925


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def csv(out,name,rows):
    pd.DataFrame(rows).to_csv(out/f'Q2_{name}.csv',index=False,encoding='utf-8-sig')


def load_data(out):
    datasets={}; descriptives={}; events=[]; blockrows=[]; lineage=[]
    arrays={'times_s':TIMES}
    for key in NAMES:
        ds=independent_data(ROOT,key); datasets[key]=ds
        raw,on,y,duration=load_raw(ROOT,key)
        edges=[0]+[(int(on[j-1])+int(on[j]))//2 for j in (20,40,60,80)]+[raw.shape[1]]
        for i in range(100):
            b=i//20; retained=i+1 in ds.ids
            guard=(on[i]-64>=edges[b]+24*FS and on[i]+205<=edges[b+1]-24*FS)
            events.append(dict(dataset=key,trial_id=i+1,block=b,cue=y[i],onset_sample=on[i],
                               duration_ms=duration[i]*1000,filter_start=edges[b],filter_end=edges[b+1],
                               prestim_start=on[i]-int(24.25*FS),prestim_end=on[i],
                               retained=retained,reason='retained' if retained else ('quality' if guard else 'boundary')))
        for b in range(5):
            chosen=ds.blocks==b
            blockrows.append(dict(dataset=key,block=b,n=int(chosen.sum()),left=int(sum(ds.y[chosen]==-1)),
                                  right=int(sum(ds.y[chosen]==1)),start=edges[b],end=edges[b+1]))
        for stage in ('before','v7'):
            d=descriptive_data(ROOT,key,stage); descriptives[key,stage]=d
            if not np.array_equal(d.y,y[d.ids-1]):
                raise ValueError('Q1 lineage mismatch')
            arrays[f'{key}_{stage}_erp']=means(d.x,d.y)
            arrays[f'{key}_{stage}_ids']=d.ids
            lineage.append(dict(dataset=key,stage=stage,n=len(d.y),strict_n=len(ds.y),
                                overlap_n=len(np.intersect1d(d.ids,ds.ids)),role='descriptive_only'))
        for field in ('x','y','ids','blocks','onset','prestim_only'):
            arrays[f'{key}_{field}']=getattr(ds,field)
    csv(out,'事件审计',events);csv(out,'时间块审计',blockrows);csv(out,'Q1衔接',lineage)
    np.savez_compressed(out/'Q2_分析数据.npz',**arrays)
    return datasets,descriptives


def fixed_prediction(bank,index,coef,new_h):
    original=bank.meta[index].h
    ref=common_delta(original).reshape(-1,original.shape[-1])
    norm=np.sqrt((ref**2*bank.weight[:,None]).sum(0))
    h=common_delta(new_h).reshape(-1,new_h.shape[-1])/np.maximum(norm,1e-10)
    return condition_waves((h@coef[:h.shape[-1]]).reshape(2,len(TIMES),3).transpose(0,2,1))


def evaluate_mechanisms(out,datasets,descriptives,banks,shape,simulations,bootstrap):
    rows=[]; summaries=[]; paramrows=[]; choices=[]; sensitivity=[]; interv=[]; shrink_rows=[]; shrink_choices=[]
    arrays={'times_s':TIMES}; descriptive_rows=[]; full=banks['full']
    for key,ds in datasets.items():
        print(f'Mechanism nested evaluation {key}',flush=True)
        for model,bank in banks.items():
            projection=bank.prepare(ds.x); obs_f=[]; pred_f=[]
            for held in np.unique(ds.blocks):
                tr=ds.blocks!=held; te=~tr
                best,loss=bank.select_prepared(projection,ds.y,ds.blocks,tr)
                coef=bank.inverse[best]@bank.prepared_stats(projection,ds.y,tr)[best]
                pred=bank.predict(coef,best); obs=means(ds.x[te],ds.y[te])
                obs_f.append(obs);pred_f.append(pred)
                row=dict(dataset=key,model=model,held_block=int(held),n_train=int(tr.sum()),n_test=int(te.sum()),
                         scale=bank.meta[best].scale,recurrence=bank.meta[best].recurrence,ridge=bank.reg[best],
                         **delta_metrics(obs,pred),**wave_metrics(obs,pred))
                rows.append(row)
                choices.append(dict(dataset=key,model=model,held_block=int(held),selected=best,
                                    train_ids=ds.ids[tr].tolist(),test_ids=ds.ids[te].tolist(),
                                    candidate_losses=loss.tolist()))
                for condition,name in enumerate(('Left','Right')):
                    for ch,channel in enumerate(CHANNELS):
                        descriptive_rows.append(dict(dataset=key,stage='strict_oof',model=model,held_block=int(held),
                                                     condition=name,channel=channel,**wave_metrics(obs[condition,ch],pred[condition,ch])))
                if model=='full':
                    factor,shrink_loss,inner_records=select_contrast_shrinkage(bank,projection,ds,tr)
                    cd=common_delta(pred);cd[1]*=factor;shrunk=condition_waves(cd)
                    shrink_rows.append(dict(dataset=key,held_block=int(held),factor=factor,
                        original_delta_MSE=row['delta_MSE'],**delta_metrics(obs,shrunk),**wave_metrics(obs,shrunk)))
                    shrink_choices.append(dict(dataset=key,held_block=int(held),factor=factor,
                        train_ids=ds.ids[tr].tolist(),test_ids=ds.ids[te].tolist(),
                        losses=shrink_loss.tolist(),inner=inner_records))
                    arrays[f'{key}_shrink_{held}']=shrunk
                    arrays[f'{key}_coef_{held}']=coef
                    c=bank.meta[best]
                    same=np.tile(shape['inputs'][:2].mean(0),(2,1))
                    sim_no=simulate(same,c.scale,c.recurrence)
                    newh=observation_sources(sim_no)
                    no_shape=fixed_prediction(bank,best,coef,newh)
                    newh=observation_sources(simulations[f'{c.scale}_0.0'])
                    no_rec=fixed_prediction(bank,best,coef,newh)
                    sym=pred.copy();sym[:,1:]=sym[:,1:].mean(1,keepdims=True)
                    for ab,pp in [('no_shape',no_shape),('no_recurrence',no_rec),('symmetric_readout',sym)]:
                        interv.append(dict(dataset=key,held_block=int(held),intervention=ab,
                                           full_delta_MSE=row['delta_MSE'],**delta_metrics(obs,pp)))
                    arrays[f'{key}_intervention_{held}']=np.stack([no_shape,no_rec,sym])
                    for parameter in ('time_scale','recurrence','F4_readout'):
                        for factor in (.8,1.2):
                            eig=np.nan
                            if parameter=='F4_readout':
                                pp=pred.copy(); pp[:,2]*=factor
                            else:
                                scale=c.scale*factor if parameter=='time_scale' else c.scale
                                rec=c.recurrence*factor if parameter=='recurrence' else c.recurrence
                                cache=f'{scale}_{rec}'
                                if cache not in simulations:
                                    simulations[cache]=simulate(shape['inputs'][:2],scale,rec)
                                sim=simulations[cache];eig=sim['eigenmax']
                                pp=fixed_prediction(bank,best,coef,observation_sources(sim))
                            sensitivity.append(dict(dataset=key,held_block=int(held),parameter=parameter,factor=factor,
                                                    eigenmax=eig,**delta_metrics(obs,pp),
                                                    delta_change_RMS=float(np.sqrt(np.mean(((pp[1]-pp[0])-(pred[1]-pred[0]))**2)))))
            obs_f=np.stack(obs_f);pred_f=np.stack(pred_f)
            arrays[f'{key}_{model}_observed']=obs_f;arrays[f'{key}_{model}_predicted']=pred_f
            frame=pd.DataFrame([r for r in rows if r['dataset']==key and r['model']==model])
            rng=np.random.default_rng(SEED); ix=rng.integers(0,len(frame),(bootstrap,len(frame)))
            gains=1-frame.delta_MSE.to_numpy()[ix].mean(1)/frame.zero_MSE.to_numpy()[ix].mean(1)
            summaries.append(dict(dataset=key,model=model,S_delta=1-frame.delta_MSE.mean()/frame.zero_MSE.mean(),
                                  S_low=np.quantile(gains,.025),S_high=np.quantile(gains,.975),
                                  delta_RMSE=np.sqrt(frame.delta_MSE.mean()),
                                  positive_folds=int(sum(frame.S_delta>0)),**wave_metrics(obs_f,pred_f)))
        for stage,data in [('strict_all',ds),('q1_before',descriptives[key,'before']),('q1_v7',descriptives[key,'v7'])]:
            best,loss=full.select(data.x,data.y,data.blocks)
            obs=means(data.x,data.y);coef=full.fit(obs,best);pred=full.predict(coef,best)
            arrays[f'{key}_{stage}_observed']=obs;arrays[f'{key}_{stage}_predicted']=pred
            for condition,name in enumerate(('Left','Right')):
                for ch,channel in enumerate(CHANNELS):
                    descriptive_rows.append(dict(dataset=key,stage=stage,model='full',held_block=-1,
                                                 condition=name,channel=channel,**wave_metrics(obs[condition,ch],pred[condition,ch])))
            if stage=='strict_all':
                paramrows.append(dict(dataset=key,scale=full.meta[best].scale,recurrence=full.meta[best].recurrence,
                                      ridge=full.reg[best],role='descriptive_all_data'))
                arrays[f'{key}_full_coef']=coef
                sim=simulations[f'{full.meta[best].scale}_{full.meta[best].recurrence}']
                arrays[f'{key}_neural_t']=sim['t'];arrays[f'{key}_neural_state']=sim['state'];arrays[f'{key}_neural_q']=sim['q']
        # Equal-block, within-block direction-stratified bootstrap for descriptive
        # ERP uncertainty. No confidence statement about a population of subjects.
        rng=np.random.default_rng(SEED); boot=[]
        for _ in range(bootstrap):
            blocks=rng.choice(np.unique(ds.blocks),5)
            erps=[]
            for b in blocks:
                cond=[]
                for y in (-1,1):
                    ids=np.flatnonzero((ds.blocks==b)&(ds.y==y))
                    cond.append(ds.x[rng.choice(ids,len(ids))].mean(0))
                erps.append(cond)
            boot.append(np.mean(erps,axis=0))
        boot=np.array(boot)
        arrays[f'{key}_delta_ci']=np.quantile(boot[:,1]-boot[:,0],[.025,.975],axis=0)
        arrays[f'{key}_erp_ci']=np.quantile(boot,[.025,.975],axis=0)
        arrays[f'{key}_erp_equal_block']=np.mean([means(ds.x[ds.blocks==b],ds.y[ds.blocks==b]) for b in np.unique(ds.blocks)],axis=0)
    csv(out,'方向收缩对照',shrink_rows);save_json(shrink_choices,out/'Q2_方向收缩选型.json')
    csv(out,'机制逐折指标',rows);csv(out,'机制汇总',summaries);csv(out,'表2_ERP拟合与预测',descriptive_rows)
    csv(out,'标定参数',paramrows);csv(out,'固定参数消融',interv);csv(out,'参数敏感性',sensitivity)
    save_json(choices,out/'Q2_机制内层选择.json')
    np.savez_compressed(out/'Q2_机制波形.npz',**arrays)
    return arrays,pd.DataFrame(summaries)


def table_definitions(out):
    rows=[('tau_LGN','LGN 中继时间常数','20 × scale','ms','模型假设'),
          ('tau_E','三级兴奋时间常数','20,40,80 × scale','ms','模型假设'),
          ('tau_I','三级抑制时间常数','40,80,160 × scale','ms','模型假设'),
          ('tau_syn_E','三级兴奋双低通突触核','25,50,100 × scale','ms','模型假设'),
          ('tau_syn_I','三级抑制双低通突触核','75,150,300 × scale','ms','模型假设'),
          ('w_EE,w_EI,w_IE,w_II','局部 E/I 耦合','1,1,1,1','无量纲','模型假设'),
          ('feedforward,input','级间偏离静息驱动及输入增益','3,3','无量纲','模型假设'),
          ('bias,inhibitory_weight','sigmoid 偏置和抑制突触权重','-2,0.7','无量纲','模型假设'),
          ('scale','全局时间尺度','0.75,1,1.25','无量纲','训练内留块选取'),
          ('recurrence','同级对称交互强度','0,0.1,0.2','无量纲','训练内留块选取'),
          ('lambda','有效观测矩阵岭正则','0.1,0.3,1,3,10,30,100','归一化源单位','训练内留块选取'),
          ('L','条件共享的 3×9 有效观测矩阵','岭回归；不反演解剖坐标','记录幅值单位/归一化源','仅训练ERP估计'),
          ('duration','模拟视觉输入时长','52/256；实测另有53/256','s','事件记录代表值'),
          ('geometry','图像形状输入','题图像素、镜像；不等于实际视角','示意图像素','题图及模型构造'),
          ('fs,bandpass,notch','采样与观测滤波','256;0.1–30;60(Q=30)','Hz','采样率实测；滤波固定'),
          ('guard,baseline','时间块保护及基线','24 s；刺激前250 ms中位数','s,ms','固定方案'),
          ('LDA_shrinkage','协方差收缩强度','0.8','无量纲','固定，未按测试分数优化')]
    csv(out,'表1_模型参数',[dict(parameter=a,meaning=b,value=c,unit=d,source=e) for a,b,c,d,e in rows])
    names=['Fz','F3','F4']; defs=[]
    for feature,dim,formula,meaning in [
        ('amplitude',3,'mean(X_c(t)), t in [250,500)ms','关键窗有符号均值'),
        ('positive_peak',3,'max valid local peak; prominence >= 0.5*prestim_SD; value>0','有效正峰幅值；不自动称为P300'),
        ('latency',3,'time of selected valid peak (ms)','有效峰潜伏期；无峰记缺失'),
        ('missing_flag',3,'1(no valid peak)','缺失标记；填充值仅由训练折估计'),
        ('LI',1,'(A_F4-A_F3)/max(|A_F4|+|A_F3|,train_q10)','稳定化窗口侧化指数'),
        ('lateral_difference',1,'A_F4-A_F3','有符号空间差值'),
        ('mechanism_projection',6,'ridge projection onto train-fitted common and R-L templates, each channel','条件未知也能计算；不是源定位'),
        ('projection6',6,'b_Fz_common,b_Fz_delta,b_F3_common,b_F3_delta,b_F4_common,b_F4_delta','六维机制投影固定对照'),
        ('direction3',3,'b_Fz_delta,b_F3_delta,b_F4_delta','三维方向投影固定对照'),
        ('mechanism_residual',3,'RMS(X_c - reconstructed X_c), 50–750ms','前向模板不能解释的剩余波动'),
        ('covariance_baseline',24,'4 windows x (3 log variances + 3 correlations)','旧空间协方差基线'),
        ('past_only',18,'first 6 DCT coefficients x 3 scalp modes','仅刺激前信号'),
        ('previous_cue',1,'direction of immediately preceding original trial','序列对照；不使用当前方向'),
        ('post_given_past',23,'mechanism features minus train-ridge prediction from past features','扣除过去线性可预测信息')]:
        defs.append(dict(feature=feature,dimension=dim,formula=formula,meaning=meaning))
    csv(out,'表3_特征定义',defs)


def decoder_run(out,datasets,bank,args):
    prepared={}
    for key,ds in datasets.items():
        _,_,all_y,_=load_raw(ROOT,key)
        previous=np.r_[0,all_y[:-1]][ds.ids-1]
        prepared[key]=prepare_decoder(ds,bank,previous);prepared[key]['all_labels']=all_y
    frame,outputs,featuremap,choices=evaluate_all(prepared,bank,details=True)
    predrows=[]; featrows=[]
    for key,data in prepared.items():
        ds=data['ds']
        for name in FEATURE_SETS:
            for i in range(len(ds.y)):
                predrows.append(dict(dataset=key,trial_id=ds.ids[i],block=ds.blocks[i],cue=ds.y[i],
                                     features=name,score=outputs[key][name][i],prediction=1 if outputs[key][name][i]>=0 else -1))
                featrows.append(dict(dataset=key,trial_id=ds.ids[i],block=ds.blocks[i],cue=ds.y[i],features=name,
                                     **{f'f{j+1:02}':v for j,v in enumerate(featuremap[key][name][i])}))
    csv(out,'逐试次留出预测',predrows);csv(out,'逐试次折外特征',featrows)
    save_json(choices,out/'Q2_判别内层选择.json')
    print('Block-resampling uncertainty',flush=True)
    ci=bootstrap_metrics(prepared,outputs,args.bootstrap,SEED)
    frame=frame.merge(ci,on=['dataset','features'],how='left')
    csv(out,'判别增量区间',ci[ci.features.str.contains('minus')])
    nullrows=[]
    for kind,count in [('block_permutation',args.permutations),('circular_shift',args.circular)]:
        rng=np.random.default_rng(SEED+(kind=='circular_shift'))
        for i in range(count):
            labelmap=null_labels(prepared,rng,kind)
            # Circular shifts can produce single-class validation blocks; reject
            # before fitting, uniformly conditional on the same admissibility rule.
            attempts=0
            while any(any(len(np.unique(labelmap[k][d['ds'].blocks==b]))<2 for b in np.unique(d['ds'].blocks))
                      for k,d in prepared.items()):
                labelmap=null_labels(prepared,rng,kind);attempts+=1
                if attempts>1000: raise RuntimeError('No admissible null permutation')
            nf,_,_,_=evaluate_all(prepared,bank,labelmap)
            nullrows.extend(dict(kind=kind,iteration=i,dataset=r.dataset,features=r.features,BA=r.BA,AUC=r.AUC)
                            for r in nf.itertuples())
            if (i+1)%25==0 or i+1==count:
                print(f'{kind}: {i+1}/{count}',flush=True)
                csv(out,'置换分布',nullrows)
        null=pd.DataFrame(nullrows); subset=null[null.kind==kind]
        for j,row in frame.iterrows():
            sample=subset[(subset.dataset==row.dataset)&(subset.features==row.features)].BA.to_numpy()
            maximum=subset[(subset.dataset==row.dataset)&subset.features.isin(PRIMARY)].groupby('iteration').BA.max().to_numpy()
            frame.loc[j,f'{kind}_p']=(1+np.sum(sample>=row.BA))/(len(sample)+1)
            frame.loc[j,f'{kind}_maxT_p']=(1+np.sum(maximum>=row.BA))/(len(maximum)+1) if row.features in PRIMARY else np.nan
    csv(out,'表4_特征判别',frame)
    transferrows=[]; transferpred=[]
    for src,dst in [('A1','B1'),('B1','A1'),('A2','B2'),('B2','A2')]:
        scores,best=transfer(prepared[src],prepared[dst],bank);ds=datasets[dst]
        for name,ss in scores.items():
            transferrows.append(dict(train=src,test=dst,features=name,**metrics(ds.y,ss)))
            for i,s in enumerate(ss):
                transferpred.append(dict(train=src,test=dst,trial_id=ds.ids[i],cue=ds.y[i],features=name,score=s))
    csv(out,'跨受试者迁移',transferrows);csv(out,'迁移逐试次预测',transferpred)
    # One prespecified visualization split: blocks 0–3 train, block 4 test, per recording.
    from sklearn.decomposition import PCA
    pca=[]
    for key,data in prepared.items():
        ds=data['ds'];tr=ds.blocks!=4;te=~tr
        ff,meta=feature_fold(data,bank,ds.y,tr,te)
        a,b=standardize(*ff['mechanism']);pc=PCA(n_components=2).fit(a)
        for role,ix,z in [('train',np.flatnonzero(tr),pc.transform(a)),('test',np.flatnonzero(te),pc.transform(b))]:
            for i,point in zip(ix,z):
                pca.append(dict(dataset=key,trial_id=ds.ids[i],cue=ds.y[i],role=role,PC1=point[0],PC2=point[1]))
    csv(out,'训练参考PCA',pca)
    return frame


def verify_outputs(out):
    pred=pd.read_csv(out/'Q2_逐试次留出预测.csv')
    metric=pd.read_csv(out/'Q2_表4_特征判别.csv')
    for row in metric.itertuples():
        sub=pred[pred.features==row.features]
        if row.dataset!='pooled': sub=sub[sub.dataset==row.dataset]
        m=metrics(sub.cue.to_numpy(),sub.score.to_numpy())
        for col in ('Accuracy','BA','Precision','Recall','F1','AUC','TN','FP','FN','TP'):
            if not np.isclose(m[col],getattr(row,col),rtol=1e-10,atol=1e-10,equal_nan=True):
                raise AssertionError(f'Recalculation mismatch: {row.dataset}/{row.features}/{col}')
    folds=pd.read_csv(out/'Q2_机制逐折指标.csv'); summary=pd.read_csv(out/'Q2_机制汇总.csv')
    for row in summary.itertuples():
        f=folds[(folds.dataset==row.dataset)&(folds.model==row.model)]
        if not np.isclose(row.S_delta,1-f.delta_MSE.mean()/f.zero_MSE.mean(),atol=1e-10):
            raise AssertionError('S_delta mismatch')
    if pred.duplicated(['dataset','trial_id','features']).any():
        raise AssertionError('Duplicate prediction rows')
    save_json(dict(status='passed',decoder_metric_rows=len(metric),mechanism_summary_rows=len(summary),
                   unique_trials=len(pred[['dataset','trial_id']].drop_duplicates()),
                   checks=['independent metric recomputation','difference-score recomputation','unique trial lineage']),
              out/'Q2_结果复核.json')


def finalize_manifest(out,redraw=False):
    """Retain the statistical-run snapshot when presentation code is revised."""
    path=out/'Q2_运行清单.json'
    m=json.loads(path.read_text(encoding='utf8'))
    original=m.setdefault('statistical_run_source_hashes',dict(m['source_hashes']))
    numerical={'data.py','generative.py','validation.py','stimulus_shape.py'}
    for name,expected in original.items():
        p=ROOT/Path(name.replace('\\','/'))
        if p.name in numerical and digest(p)!=expected:
            raise RuntimeError('Numerical source changed; recompute statistics before final delivery')
    sources=list((ROOT/'q2').glob('*.py'))+list((ROOT/'q2/q2model').glob('*.py'))
    m['source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in sources}
    m['test_source_hashes']={str(p.relative_to(ROOT)):digest(p) for p in (ROOT/'q2/tests').glob('*.py')}
    m['requirements_sha256']=digest(ROOT/'q2/requirements.txt')
    m['presentation_refreshed']=bool(redraw)
    m['numerical_modules_unchanged']=True
    m['artifact_hashes']={p.name:digest(p) for p in sorted(out.glob('Q2_*')) if p.is_file() and p!=path}
    save_json(m,path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'q2_result')
    parser.add_argument('--quick',action='store_true',help='19 permutations / 9 shifts / 100 bootstrap; exploratory smoke run')
    parser.add_argument('--redraw',action='store_true',help='Regenerate figures and paper from saved numerical outputs')
    parser.add_argument('--permutations',type=int,default=1999)
    parser.add_argument('--circular',type=int,default=199)
    parser.add_argument('--bootstrap',type=int,default=1000)
    parser.add_argument('--no-plots',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not out.is_dir(): parser.error('Output directory must already exist; this pipeline never creates directories')
    if args.quick: args.permutations,args.circular,args.bootstrap=19,9,100
    if min(args.permutations,args.circular,args.bootstrap)<1: parser.error('Counts must be positive')
    os.environ['MPLCONFIGDIR']=str(out)
    from q2model.reporting import render_all,write_paper
    before_dirs={str(p.resolve()) for p in ROOT.rglob('*') if p.is_dir() and '.git' not in p.parts and '.venv' not in p.parts}
    if args.redraw:
        if not args.no_plots: render_all(out)
        write_paper(out);verify_outputs(out)
        finalize_manifest(out,redraw=True)
        print('Redraw and numerical verification complete',flush=True);return
    started=time.time()
    with threadpool_limits(limits=1):
        datasets,descriptives=load_data(out)
        shape=encode_shapes(ROOT);np.savez_compressed(out/'Q2_形状输入.npz',**shape)
        csv(out,'形状响应',[dict(stimulus=name,left_preference=f[0],right_preference=f[1],common=f[2])
                            for name,f in zip(shape['names'],shape['inputs'])])
        print('Building nine dynamical candidates and matched ablations',flush=True)
        banks,sims=build_banks(shape)
        candidate_meta=[dict(index=i,model=c.model,scale=c.scale,recurrence=c.recurrence,ridge=banks['full'].reg[i])
                        for i,c in enumerate(banks['full'].meta)]
        save_json(candidate_meta,out/'Q2_候选模型.json')
        evaluate_mechanisms(out,datasets,descriptives,banks,shape,sims,args.bootstrap)
        table_definitions(out)
        print('Nested feature decoding and null inference',flush=True)
        decoder_run(out,datasets,banks['full'],args)
    sources=list((ROOT/'q2').glob('*.py'))+list((ROOT/'q2/q2model').glob('*.py'))
    inputs=list((ROOT/'data').glob('*.mat'))+[ROOT/'服务于脑机接口与精神性疾病诊断的脑电图计算模型.docx']
    inputs+=list((ROOT/'eeg_v7_results').glob('*/可复核波形.npz'))
    manifest=dict(status='complete',seed=SEED,quick=args.quick,permutations=args.permutations,
                  circular_shifts=args.circular,bootstrap=args.bootstrap,seconds=time.time()-started,
                  python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,
                  input_hashes={str(p.relative_to(ROOT)):digest(p) for p in inputs},
                  source_hashes={str(p.relative_to(ROOT)):digest(p) for p in sources},
                  counts={k:len(d.y) for k,d in datasets.items()},
                  primary_features=list(PRIMARY), feature_sets=list(FEATURE_SETS),
                  baseline_archive='优化计划/优化前基线.zip',
                  amplitude_unit='原始记录幅值单位；未发现可验证的物理标定',
                  validation='repeatedly inspected development data; blocked internal and two-subject transfer only')
    save_json(manifest,out/'Q2_运行清单.json')
    verify_outputs(out)
    if not args.no_plots: render_all(out)
    write_paper(out)
    finalize_manifest(out)
    after_dirs={str(p.resolve()) for p in ROOT.rglob('*') if p.is_dir() and '.git' not in p.parts and '.venv' not in p.parts}
    if after_dirs-before_dirs:
        raise RuntimeError(f'Unexpected new directories: {after_dirs-before_dirs}')
    print(f'Complete: {out}, elapsed {time.time()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
