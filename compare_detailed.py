import pandas as pd

orig_res = pd.read_csv('eeg_v7_results/汇总与说明/四组核心指标重采样区间.csv')
unsup_res = pd.read_csv('eeg_unsupervised_results/汇总与说明/四组核心指标重采样区间.csv')

def get_mean(df, metric):
    return df[df['metric'] == metric]['point'].mean()

metrics = {
    'left_right_retention_ratio': '左右特征保留率',
    'proxy_MAE_reduction_pct': '波形误差降低比 (%)',
    'SNR_proxy_gain_dB': '信噪比提升 (dB)'
}

print("=== 整体对比 ===")
for m, name in metrics.items():
    orig_val = get_mean(orig_res, m)
    unsup_val = get_mean(unsup_res, m)
    diff = unsup_val - orig_val
    trend = "提升" if diff > 0 else "下降"
    print(f"{name}: 原始={orig_val:.3f}, 无监督={unsup_val:.3f} | {trend} {abs(diff):.3f}")

print("\n=== 分受试者/项目详细对比 (波形误差降低比 proxy_MAE_reduction_pct) ===")
# We don't have per-dataset proxy_MAE_reduction_pct easily in the summary, 
# let's look at the raw metric from 逐方向逐通道评价指标.csv
def compare_per_dataset(metric_col, higher_is_better=True):
    for subject in ['A', 'B']:
        for task in [1, 2]:
            dataset = f'受试者{subject}_项目{"一" if task==1 else "二"}'
            orig = pd.read_csv(f'eeg_v7_results/{dataset}/逐方向逐通道评价指标.csv')
            unsup = pd.read_csv(f'eeg_unsupervised_results/{dataset}/逐方向逐通道评价指标.csv')
            
            orig_val = orig[orig['stage'] == 'V7'][metric_col].mean()
            unsup_val = unsup[unsup['stage'] == 'V7'][metric_col].mean()
            
            diff = unsup_val - orig_val
            if higher_is_better:
                trend = "提升" if diff > 0 else "下降"
            else:
                trend = "优化" if diff < 0 else "退步"
            print(f"{dataset} {metric_col}: 原始={orig_val:.3f}, 无监督={unsup_val:.3f} | {trend} {abs(diff):.3f}")

print("\n--- 绝对误差 (MAE_after，越低越好) ---")
compare_per_dataset('MAE_after', False)

print("\n--- P300正面积绝对误差 (P300_AUC_error_after_unit_ms，越低越好) ---")
compare_per_dataset('P300_AUC_error_after_unit_ms', False)

