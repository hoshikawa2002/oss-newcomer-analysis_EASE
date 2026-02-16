"""
変化点(2024-01)前後のリポジトリ別GFI変化を分析し、
低下の主要因となったリポジトリを特定する。
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path('../data')
gfi_dir = DATA_DIR / 'gfi_issues'
all_dir = DATA_DIR / 'all_issue_labels'

# 変化点: 2024-01 (Pettitt test)
# Before: 2021-07 ~ 2023-12 (30 months)
# After:  2024-01 ~ 2025-06 (18 months)
CHANGEPOINT = '2024-01'
START = '2021-07'
END = '2025-06'

print("=" * 70)
print("変化点前後のリポジトリ別GFI変化分析")
print(f"変化点: {CHANGEPOINT}")
print("=" * 70)

# Collect monthly data per repo
repo_monthly = {}

for gfi_file in gfi_dir.glob('*_gfi_issues.json'):
    repo = gfi_file.stem.replace('_gfi_issues', '')
    all_file = all_dir / f'{repo}_all_issues.json'
    if not all_file.exists():
        continue

    with open(gfi_file) as f:
        gfi_issues = json.load(f)
    with open(all_file) as f:
        data = json.load(f)
        all_issues = data.get('issues', data) if isinstance(data, dict) else data

    monthly = defaultdict(lambda: {'gfi': 0, 'total': 0})
    for issue in gfi_issues:
        m = issue['created_at'][:7]
        if START <= m <= END:
            monthly[m]['gfi'] += 1
    for issue in all_issues:
        m = issue['created_at'][:7]
        if START <= m <= END:
            monthly[m]['total'] += 1

    repo_monthly[repo] = monthly

# Calculate before/after for each repo
results = []
for repo, monthly in repo_monthly.items():
    before_gfi = sum(monthly[m]['gfi'] for m in monthly if m < CHANGEPOINT)
    before_total = sum(monthly[m]['total'] for m in monthly if m < CHANGEPOINT)
    after_gfi = sum(monthly[m]['gfi'] for m in monthly if m >= CHANGEPOINT)
    after_total = sum(monthly[m]['total'] for m in monthly if m >= CHANGEPOINT)

    before_months = len([m for m in monthly if m < CHANGEPOINT and monthly[m]['total'] > 0])
    after_months = len([m for m in monthly if m >= CHANGEPOINT and monthly[m]['total'] > 0])

    # Monthly average GFI count and ratio
    before_gfi_monthly = before_gfi / before_months if before_months > 0 else 0
    after_gfi_monthly = after_gfi / after_months if after_months > 0 else 0
    before_ratio = (before_gfi / before_total * 100) if before_total > 0 else 0
    after_ratio = (after_gfi / after_total * 100) if after_total > 0 else 0

    # Contribution to aggregate decline (absolute GFI count change per month)
    gfi_change_monthly = after_gfi_monthly - before_gfi_monthly

    results.append({
        'repository': repo,
        'before_gfi': before_gfi,
        'after_gfi': after_gfi,
        'before_gfi_monthly': before_gfi_monthly,
        'after_gfi_monthly': after_gfi_monthly,
        'gfi_change_monthly': gfi_change_monthly,
        'before_ratio': before_ratio,
        'after_ratio': after_ratio,
        'ratio_change': after_ratio - before_ratio,
        'before_total_monthly': before_total / before_months if before_months > 0 else 0,
        'after_total_monthly': after_total / after_months if after_months > 0 else 0,
    })

df = pd.DataFrame(results)

# Sort by absolute GFI count change (most negative = biggest contributor to decline)
df = df.sort_values('gfi_change_monthly')

print(f"\n--- 月平均GFI数の変化が大きい上位10リポジトリ (減少順) ---")
print(f"{'Repository':<35} {'Before':>8} {'After':>8} {'Change':>8} {'Before%':>8} {'After%':>8}")
print("-" * 80)
for _, row in df.head(10).iterrows():
    print(f"{row['repository']:<35} {row['before_gfi_monthly']:>8.2f} {row['after_gfi_monthly']:>8.2f} "
          f"{row['gfi_change_monthly']:>8.2f} {row['before_ratio']:>7.3f}% {row['after_ratio']:>7.3f}%")

print(f"\n--- 月平均GFI数が増加した上位5リポジトリ ---")
df_inc = df.sort_values('gfi_change_monthly', ascending=False)
print(f"{'Repository':<35} {'Before':>8} {'After':>8} {'Change':>8} {'Before%':>8} {'After%':>8}")
print("-" * 80)
for _, row in df_inc.head(5).iterrows():
    print(f"{row['repository']:<35} {row['before_gfi_monthly']:>8.2f} {row['after_gfi_monthly']:>8.2f} "
          f"{row['gfi_change_monthly']:>8.2f} {row['before_ratio']:>7.3f}% {row['after_ratio']:>7.3f}%")

# Aggregate: how much of the total decline is explained by top repos?
total_before = df['before_gfi_monthly'].sum()
total_after = df['after_gfi_monthly'].sum()
total_change = total_after - total_before

print(f"\n--- 集計 ---")
print(f"全体: 月平均GFI {total_before:.1f} → {total_after:.1f} (変化: {total_change:.1f})")

# Top 5 decliners
top5_change = df.head(5)['gfi_change_monthly'].sum()
print(f"上位5リポジトリの寄与: {top5_change:.1f} ({top5_change/total_change*100:.1f}% of total decline)")

top3_change = df.head(3)['gfi_change_monthly'].sum()
print(f"上位3リポジトリの寄与: {top3_change:.1f} ({top3_change/total_change*100:.1f}% of total decline)")

# Deep dive into top decliners: show monthly pattern around changepoint
print(f"\n{'='*70}")
print("主要リポジトリの月別GFI数推移 (変化点前後)")
print(f"{'='*70}")

for _, row in df.head(5).iterrows():
    repo = row['repository']
    monthly = repo_monthly[repo]

    print(f"\n--- {repo} ---")
    print(f"  変化点前平均: {row['before_gfi_monthly']:.2f}/月, 変化点後平均: {row['after_gfi_monthly']:.2f}/月")

    # Show last 6 months before and first 6 months after changepoint
    all_months = sorted(monthly.keys())
    before_6 = [m for m in all_months if '2023-07' <= m < CHANGEPOINT]
    after_6 = [m for m in all_months if CHANGEPOINT <= m <= '2024-06']

    print(f"  Y3後半 (2023-07~2023-12): ", end="")
    for m in before_6:
        print(f"{m[-2:]}:{monthly[m]['gfi']} ", end="")
    print()

    print(f"  Y3/Y4境界 (2024-01~2024-06): ", end="")
    for m in after_6:
        print(f"{m[-2:]}:{monthly[m]['gfi']} ", end="")
    print()

    after_12 = [m for m in all_months if '2024-07' <= m <= '2025-06']
    print(f"  Y4 (2024-07~2025-06): ", end="")
    for m in after_12:
        print(f"{m[-2:]}:{monthly[m]['gfi']} ", end="")
    print()

    # Check if total issues also declined (to distinguish GFI-specific vs overall decline)
    total_change_pct = (row['after_total_monthly'] - row['before_total_monthly']) / row['before_total_monthly'] * 100 if row['before_total_monthly'] > 0 else 0
    print(f"  全issue数の変化: {row['before_total_monthly']:.0f}/月 → {row['after_total_monthly']:.0f}/月 ({total_change_pct:+.1f}%)")
    if abs(total_change_pct) < 20 and row['gfi_change_monthly'] < -1:
        print(f"  → Issue数は安定、GFIラベリングが選択的に減少")
    elif total_change_pct < -20:
        print(f"  → Issue数自体も減少")

# Save
df.to_csv('../results/rq1/changepoint_repo_analysis.csv', index=False)
print(f"\n結果保存: ../results/rq1/changepoint_repo_analysis.csv")
