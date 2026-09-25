import pandas as pd
import numpy as np

orig_res = pd.read_csv('eeg_v7_results/汇总与说明/四组核心指标重采样区间.csv')
unsup_res = pd.read_csv('eeg_unsupervised_results/汇总与说明/四组核心指标重采样区间.csv')

def get_mean(df, metric):
    val = df[df['metric'] == metric]['point'].mean()
    return val

metrics = ['left_right_retention_ratio', 'proxy_MAE_reduction_pct', 'SNR_proxy_gain_dB']

print(f"{'Metric':<30} | {'Orig V7':<10} | {'Unsupervised':<10}")
print("-" * 55)
for m in metrics:
    orig_val = get_mean(orig_res, m)
    unsup_val = get_mean(unsup_res, m)
    print(f"{m:<30} | {orig_val:<10.3f} | {unsup_val:<10.3f}")

