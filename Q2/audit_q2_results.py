#!/usr/bin/env python3
"""Independent post-run numerical audit and rank-conditional bootstrap summaries.

Does not refit or select models. Refuses incomplete/debug runs. All added output
and this audit source are registered in the existing manifest.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/eeg-q2-mpl")
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["OMP_NUM_THREADS"]="1"
from pathlib import Path
import argparse
import json
import sys
import numpy as np
import pandas as pd
from q2model.data import sha,save_csv,save_json,NAMES
from q2model.validation import metrics,holm


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'Q2_result')
    parser.add_argument('--refresh-report',action='store_true',help='Allow ONLY report.py to change; regenerate reporting, never reuse changed numerical code')
    args=parser.parse_args();out=args.output;root=Path(__file__).resolve().parents[1]
    manifest=json.loads((out/'manifest.json').read_text())
    if manifest['status']!='complete' or manifest['config']['quick']:
        raise RuntimeError('Requires a completed full run')
    report_change=None
    for p,digest in manifest['source'].items():
        current=sha(root/p)
        if current!=digest:
            if args.refresh_report and p=='Q2/q2model/report.py':
                report_change=dict(path=p,previous_sha=digest,new_sha=current,reason='report-only layout and narrative revision; numerical sources unchanged')
            else:raise RuntimeError(f'Changed numerical source: {p}')
    for p,digest in manifest['inputs'].items():
        if sha(root/p)!=digest:raise RuntimeError(f'Changed input: {p}')
    if args.refresh_report:
        if report_change:
            manifest.setdefault('report_revisions',[]).append(report_change)
            manifest['source'][report_change['path']]=report_change['new_sha']
            import hashlib
            manifest['signature']=hashlib.sha256(json.dumps(dict(source=manifest['source'],inputs=manifest['inputs'],config=manifest['config']),sort_keys=True).encode()).hexdigest()
            save_json(manifest,out/'manifest.json')
        from q2model.report import make_figures,write_report
        make_figures(out)
        write_report(out)
    checks=[];intervals=[];coef_intervals=[]
    summary=pd.read_csv(out/'validation/classification_summary.csv',float_precision='round_trip')
    pvals=pd.read_csv(out/'validation/permutation_tests.csv',float_precision='round_trip')
    raw_p=[]
    for key in NAMES:
        pred=pd.read_csv(out/'validation/blocked'/f'{key}_predictions.csv',float_precision='round_trip')
        for model,g in pred.groupby('model'):
            calc=metrics(g.truth.to_numpy(),g.prediction.to_numpy(),g.score.to_numpy())
            saved=summary[(summary.dataset==key)&(summary.model==model)].iloc[0]
            if len(g)!=g.trial_id.nunique():raise AssertionError('Duplicate OOF trials')
            for k in ['BA','AUC','TN','FP','FN','TP']:
                if not np.isclose(calc[k],saved[k],atol=1e-12):raise AssertionError((key,model,k))
        perm_files=sorted((out/'checkpoints/permutations'/key).glob('*.json'))
        if len(perm_files)!=999:raise AssertionError('Missing permutations')
        actual=pred[pred.model=='mechanism_selected']
        # Integer comparison is immune to floating-point ties in BA.
        n_left=int((actual.truth==-1).sum());n_right=int((actual.truth==1).sum())
        stat=int(((actual.truth==-1)&(actual.prediction==-1)).sum())*n_right+int(((actual.truth==1)&(actual.prediction==1)).sum())*n_left
        exceed=0
        for file in perm_files:
            r=json.loads(file.read_text());y=np.asarray(r['permuted_truth']);p=np.asarray(r['predictions'])
            if r['status']!='complete' or len(r['folds'])!=5:raise AssertionError('Incomplete supervised replay')
            if ((y==-1).sum(),(y==1).sum())!=(n_left,n_right):raise AssertionError('Permutation changed class totals')
            null_stat=int(((y==-1)&(p==-1)).sum())*n_right+int(((y==1)&(p==1)).sum())*n_left
            exceed+=int(null_stat>=stat)
            calc=metrics(y,p,np.asarray(r['scores']))
            if not np.isclose(calc['BA'],r['BA'],atol=1e-12):raise AssertionError('Permutation BA mismatch')
        exact_p=(exceed+1)/1000;reported=float(pvals.loc[pvals.dataset==key,'p_raw'].iloc[0]);raw_p.append(exact_p)
        checks.append(dict(dataset=key,p_integer_reference=exact_p,p_reported=reported,
                           p_matches=bool(np.isclose(exact_p,reported,atol=1e-12)),permutations_checked=999))
        boot=json.loads((out/'stability'/key/'full_refit_bootstrap.json').read_text())
        if len(boot)!=200:raise AssertionError('Missing full refits')
        for rank in (2,3):
            selected=[r for r in boot if r['rank']==rank]
            if not selected:continue
            theta=np.array([r['theta'] for r in selected]);lo,med,hi=np.quantile(theta,[.025,.5,.975],axis=0)
            for j,name in enumerate(['delay','tau_E','tau_I','tau_S'][:rank+1]):
                intervals.append(dict(dataset=key,rank=rank,n_refits=len(selected),parameter=name,
                    low_ms=lo[j],median_ms=med[j],high_ms=hi[j],
                    interpretation='conditional_on_selected_rank_and_original_time_blocks'))
            for kind in ('A','D'):
                v=np.array([r[kind] for r in selected]);lo,med,hi=np.quantile(v,[.025,.5,.975],axis=0)
                for mode in range(3):
                    for kernel in range(rank):
                        coef_intervals.append(dict(dataset=key,rank=rank,n_refits=len(selected),kind=kind,
                            spatial_mode=mode,kernel=kernel,low=lo[mode,kernel],median=med[mode,kernel],high=hi[mode,kernel]))
    save_csv(checks,out/'audit/independent_metric_checks.csv')
    save_csv(intervals,out/'stability/parameter_intervals_by_rank.csv')
    save_csv(coef_intervals,out/'stability/coefficient_intervals_by_rank.csv')
    # Do not conceal a material inference discrepancy: require correction upstream.
    if not all(r['p_matches'] for r in checks):
        raise AssertionError('Floating point BA ties changed p-values; see audit/independent_metric_checks.csv')
    np.testing.assert_allclose(holm(raw_p),pvals.p_holm.to_numpy(),atol=1e-12)
    import matplotlib
    matplotlib.use('Agg')
    from q2model.report import BLUE,ORANGE,GRAY
    import matplotlib.pyplot as plt
    data=pd.DataFrame(intervals);fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    names={'delay':'有效延迟','tau_E':'快核尺度','tau_I':'中核尺度','tau_S':'慢核尺度'}
    for ax,(key,label) in zip(axes.flat,NAMES.items()):
        g=data[data.dataset==key].reset_index(drop=True)
        for i,r in enumerate(g.itertuples()):
            color=BLUE if r.rank==2 else ORANGE
            ax.plot([r.low_ms,r.high_ms],[i,i],color=color,lw=2)
            ax.scatter(r.median_ms,i,color=color,marker='o' if r.rank==2 else 's')
        ax.set_yticks(range(len(g)),[f'R{r.rank} {names[r.parameter]} (n={r.n_refits})' for r in g.itertuples()])
        ax.invert_yaxis();ax.set_xlim(0,620);ax.set_xlabel('时间参数 / ms');ax.set_title(label)
    fig.suptitle('完整重估的参数波动\n按选中核数分组的2.5%–97.5%分位区间；固定原记录块，不是人群置信区间',fontsize=14)
    for ext in ('png','pdf'):fig.savefig(out/'figures'/f'11_参数重估区间.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    chart_path=out/'figures/chart_map.json'
    charts=json.loads(chart_path.read_text())
    charts=[r for r in charts if r['figure']!='11_参数重估区间']
    charts.append(dict(figure='11_参数重估区间',question='参数完整重估的波动与边界',
                      source='stability/parameter_intervals_by_rank.csv',renderer='matplotlib static',palette='blue orange and neutral'))
    save_json(charts,chart_path)
    from q2model.report import write_report
    write_report(out)
    source={'Q2/audit_q2_results.py':sha(Path(__file__))}
    verification=dict(status='passed',source=source,checks=checks,checked_permutations=3996,
                       checked_parameter_refits=800,notes='Integer-statistic permutation ties independently verified')
    save_json(verification,out/'audit/final_verification.json')
    manifest['verification']=verification
    manifest['output_hashes']={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*'))
        if p.is_file() and p.name!='manifest.json' and not p.name.endswith('.tmp')}
    save_json(manifest,out/'manifest.json')
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
