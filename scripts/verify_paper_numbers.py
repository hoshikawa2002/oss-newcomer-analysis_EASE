"""
論文内の数値をデータから再計算して検証するスクリプト。
Usage: python verify_paper_numbers.py
"""
import json, os, re, csv
import numpy as np
from collections import Counter, defaultdict
from pathlib import Path
from statsmodels.stats.multitest import multipletests

DATA_DIR = Path('../data')

# ===== Helper functions =====
def get_year(date_str):
    y, m = int(date_str[:4]), int(date_str[5:7])
    if (y == 2021 and m >= 7) or (y == 2022 and m <= 6): return 'Y1'
    elif (y == 2022 and m >= 7) or (y == 2023 and m <= 6): return 'Y2'
    elif (y == 2023 and m >= 7) or (y == 2024 and m <= 6): return 'Y3'
    elif (y == 2024 and m >= 7) or (y == 2025 and m <= 6): return 'Y4'
    return None

def mann_kendall(data):
    from scipy import stats as sp
    n = len(data)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = data[j] - data[i]
            if diff > 0: s += 1
            elif diff < 0: s -= 1
    unique = np.unique(data)
    if len(unique) == n:
        var_s = n * (n - 1) * (2 * n + 5) / 18
    else:
        tp = np.array([np.sum(data == u) for u in unique])
        var_s = (n * (n - 1) * (2 * n + 5) - np.sum(tp * (tp - 1) * (2 * tp + 5))) / 18
    if s > 0: z = (s - 1) / np.sqrt(var_s)
    elif s < 0: z = (s + 1) / np.sqrt(var_s)
    else: z = 0
    p = 2 * sp.norm.sf(abs(z))
    tau = s / (n * (n - 1) / 2)
    return tau, p

# ===== Load data =====
print("=" * 70)
print("論文データ検証スクリプト")
print("=" * 70)

# --- Total counts ---
gfi_dir = DATA_DIR / 'gfi_issues'
all_dir = DATA_DIR / 'all_issue_labels'
pr_dir = DATA_DIR / 'gfi_prs'

total_gfi = 0
total_issues = 0
gfi_by_year = Counter()
total_by_year = Counter()
repo_count_gfi = 0
repo_count_all = 0

for f in gfi_dir.glob('*_gfi_issues.json'):
    with open(f) as fp:
        issues = json.load(fp)
    repo_count_gfi += 1
    for i in issues:
        yr = get_year(i['created_at'])
        if yr:
            gfi_by_year[yr] += 1
            total_gfi += 1

for f in all_dir.glob('*_all_issues.json'):
    with open(f) as fp:
        data = json.load(fp)
    issues = data.get('issues', data) if isinstance(data, dict) else data
    repo_count_all += 1
    for i in issues:
        yr = get_year(i['created_at'])
        if yr:
            total_by_year[yr] += 1
            total_issues += 1

total_prs = 0
total_merged = 0
pr_by_year = Counter()
merged_by_year = Counter()

for f in pr_dir.glob('*_gfi_prs.json'):
    with open(f) as fp:
        prs = json.load(fp)
    for pr in prs:
        d = pr['pr_data']
        yr = get_year(d['created_at'])
        if yr:
            total_prs += 1
            pr_by_year[yr] += 1
            if d.get('state') == 'MERGED':
                total_merged += 1
                merged_by_year[yr] += 1

# --- GFI Engagement ---
gfi_map = {}
for f in gfi_dir.glob('*_gfi_issues.json'):
    repo = f.stem.replace('_gfi_issues', '')
    with open(f) as fp:
        issues = json.load(fp)
    gfi_map[repo] = {}
    for i in issues:
        yr = get_year(i['created_at'])
        if yr:
            gfi_map[repo][i['number']] = yr

engaged = set()
engaged_by_year = Counter()
for f in pr_dir.glob('*_gfi_prs.json'):
    repo = f.stem.replace('_gfi_prs', '')
    with open(f) as fp:
        prs = json.load(fp)
    for pr in prs:
        for ref in pr.get('referenced_gfi_issues', []):
            num = ref['issue_number']
            if repo in gfi_map and num in gfi_map[repo]:
                key = (repo, num)
                if key not in engaged:
                    engaged.add(key)
                    engaged_by_year[gfi_map[repo][num]] += 1

# --- Monthly data for MK test ---
monthly_gfi = defaultdict(lambda: {'gfi': 0, 'total': 0})
for f in gfi_dir.glob('*_gfi_issues.json'):
    with open(f) as fp:
        issues = json.load(fp)
    for i in issues:
        m = i['created_at'][:7]
        if '2021-07' <= m <= '2025-06':
            monthly_gfi[m]['gfi'] += 1

for f in all_dir.glob('*_all_issues.json'):
    with open(f) as fp:
        data = json.load(fp)
    issues = data.get('issues', data) if isinstance(data, dict) else data
    for i in issues:
        m = i['created_at'][:7]
        if '2021-07' <= m <= '2025-06':
            monthly_gfi[m]['total'] += 1

monthly_pr = defaultdict(lambda: {'merged': 0, 'total': 0, 'body_lengths': []})
for f in pr_dir.glob('*_gfi_prs.json'):
    with open(f) as fp:
        prs = json.load(fp)
    for pr in prs:
        d = pr['pr_data']
        m = d['created_at'][:7]
        if '2021-07' <= m <= '2025-06':
            monthly_pr[m]['total'] += 1
            if d.get('state') == 'MERGED':
                monthly_pr[m]['merged'] += 1
            monthly_pr[m]['body_lengths'].append(d.get('body_length', 0))

# Compute monthly arrays
months = sorted(set(list(monthly_gfi.keys()) + list(monthly_pr.keys())))
gfi_ratios_monthly = []
merge_rates_monthly = []
body_lengths_monthly = []
for m in months:
    if monthly_gfi[m]['total'] > 0:
        gfi_ratios_monthly.append(monthly_gfi[m]['gfi'] / monthly_gfi[m]['total'] * 100)
    if monthly_pr[m]['total'] > 0:
        merge_rates_monthly.append(monthly_pr[m]['merged'] / monthly_pr[m]['total'] * 100)
        body_lengths_monthly.append(np.median(monthly_pr[m]['body_lengths']))

# Yearly averages of monthly values
yearly_avg = defaultdict(lambda: defaultdict(list))
for m in months:
    yr = get_year(m + '-01')
    if yr and monthly_gfi[m]['total'] > 0:
        yearly_avg[yr]['gfi_ratio'].append(monthly_gfi[m]['gfi'] / monthly_gfi[m]['total'] * 100)
    if yr and monthly_pr[m]['total'] > 0:
        yearly_avg[yr]['merge_rate'].append(monthly_pr[m]['merged'] / monthly_pr[m]['total'] * 100)
        yearly_avg[yr]['body_length'].append(np.median(monthly_pr[m]['body_lengths']))

# ===== Print verification =====
print("\n--- 基本カウント ---")
checks = [
    ("Total issues", total_issues, 406826),
    ("GFI issues", total_gfi, 3300),
    ("GFI PRs", total_prs, 1117),
    ("Repos with GFI", repo_count_gfi, 30),
    ("Total repos", repo_count_all, 37),
    ("Merged PRs", total_merged, 592),
    ("Merge rate (%)", total_merged / total_prs * 100, 53.0),
    ("Engaged GFI issues", len(engaged), 891),
    ("Engagement rate (%)", len(engaged) / total_gfi * 100, 27.0),
]
for name, actual, expected in checks:
    match = "✓" if abs(actual - expected) < 0.15 else "✗"
    print(f"  {match} {name:25s}: data={actual:>10.1f}  paper={expected:>10.1f}")

print("\n--- Table 1: GFI Ratio (yearly avg of monthly) ---")
for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
    avg = np.mean(yearly_avg[yr]['gfi_ratio'])
    print(f"  {yr}: {avg:.2f}%")

print("\n--- Table 1: Merge Rate (yearly avg of monthly) ---")
for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
    avg = np.mean(yearly_avg[yr]['merge_rate'])
    print(f"  {yr}: {avg:.1f}%")

print("\n--- Table 1: Description Length (yearly avg of monthly medians) ---")
for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
    avg = np.mean(yearly_avg[yr]['body_length'])
    print(f"  {yr}: {avg:.0f}")

print("\n--- Table 1: Newcomer Engagement (aggregate by year) ---")
for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
    rate = engaged_by_year[yr] / gfi_by_year[yr] * 100
    print(f"  {yr}: {engaged_by_year[yr]}/{gfi_by_year[yr]} = {rate:.1f}%")

print("\n--- Mann-Kendall tests ---")
gfi_arr = np.array(gfi_ratios_monthly)
mr_arr = np.array(merge_rates_monthly)

tau_gfi, p_gfi = mann_kendall(gfi_arr)
tau_mr, p_mr = mann_kendall(mr_arr)

# Description Length: use CSV (HTML comments removed per methodology)
csv_path_pr = Path('../results/rq2/monthly_pr_metrics.csv')
if csv_path_pr.exists():
    with open(csv_path_pr) as f:
        pr_rows = list(csv.DictReader(f))
    bl_csv = np.array([float(r['body_length']) for r in pr_rows])
    tau_bl, p_bl = mann_kendall(bl_csv)
    bl_note = "(from CSV, HTML comments removed)"
else:
    bl_arr = np.array(body_lengths_monthly)
    tau_bl, p_bl = mann_kendall(bl_arr)
    bl_note = "(from raw data, includes HTML comments)"

mk_checks = [
    ("GFI Ratio", tau_gfi, p_gfi, -0.44),
    ("Merge Rate", tau_mr, p_mr, -0.35),
    ("Desc Length", tau_bl, p_bl, 0.35),
]
for name, tau, p, expected_tau in mk_checks:
    match = "✓" if abs(round(tau, 2) - expected_tau) < 0.015 else "✗"
    sig = "***" if p < 0.001 else ("**" if p < 0.01 else ("*" if p < 0.05 else "n.s."))
    extra = f"  {bl_note}" if name == "Desc Length" else ""
    print(f"  {match} {name:15s}: tau={tau:.4f} ({sig}), p={p:.6f}  (paper: {expected_tau}){extra}")

# Also show Description Length yearly averages from CSV
if csv_path_pr.exists():
    print("\n--- Table 1: Description Length (from CSV, HTML removed) ---")
    csv_yearly = defaultdict(list)
    for r in pr_rows:
        yr = get_year(r['month'] + '-01')
        if yr:
            csv_yearly[yr].append(float(r['body_length']))
    for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
        avg = np.mean(csv_yearly[yr])
        print(f"  {yr}: {avg:.0f}")

print("\n--- Table 2: Merge rate by task type ---")
# Load from script output
csv_path = Path('../results/rq2/monthly_metrics_by_type.csv')
if csv_path.exists():
    type_year = defaultdict(lambda: {'pr': 0, 'merged': 0})
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            yr = get_year(row['month'] + '-01')
            t = row['task_type']
            n = int(row['pr_count'])
            m = round(n * float(row['merge_rate']) / 100)
            type_year[(t, yr)]['pr'] += n
            type_year[(t, yr)]['merged'] += m

    for task in ['bug', 'feature', 'documentation', 'other']:
        parts = []
        for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
            d = type_year[(task, yr)]
            rate = d['merged'] / d['pr'] * 100 if d['pr'] > 0 else 0
            parts.append(f"{yr}:{rate:.1f}%")
        total_m = sum(type_year[(task, yr)]['merged'] for yr in ['Y1', 'Y2', 'Y3', 'Y4'])
        total_p = sum(type_year[(task, yr)]['pr'] for yr in ['Y1', 'Y2', 'Y3', 'Y4'])
        overall = total_m / total_p * 100 if total_p > 0 else 0
        print(f"  {task:15s}  {' '.join(parts)}  Total:{overall:.1f}%")

print("\n--- Multiple testing corrections verification ---")

# Family A (Table 1): Holm-Bonferroni, n=4
# GFI Ratio, Merge Rate, Desc Length, Engagement
# Need Newcomer Engagement MK p-value
eng_csv = Path('../results/rq1/monthly_newcomer_engagement.csv')
if eng_csv.exists():
    with open(eng_csv) as f:
        eng_rows = list(csv.DictReader(f))
    eng_values = np.array([float(r['engagement_ratio']) for r in eng_rows])
    tau_eng, p_eng = mann_kendall(eng_values)
else:
    tau_eng, p_eng = 0.06, 0.52  # fallback from paper

family_a_raw = [p_gfi, p_mr, p_bl, p_eng]
reject_a, padj_a, _, _ = multipletests(family_a_raw, alpha=0.05, method='holm')
print("\nFamily A (Table 1, Holm, n=4):")
labels_a = ['GFI Ratio', 'Merge Rate', 'Desc Length', 'Engagement']
for name, raw, adj in zip(labels_a, family_a_raw, padj_a):
    sig_raw = "***" if raw < 0.001 else ("**" if raw < 0.01 else ("*" if raw < 0.05 else "n.s."))
    sig_adj = "***" if adj < 0.001 else ("**" if adj < 0.01 else ("*" if adj < 0.05 else "n.s."))
    print(f"  {name:15s}: raw p={raw:.6f} ({sig_raw}) -> adj p={adj:.6f} ({sig_adj})")

# Family B (Table 2): Holm-Bonferroni, n=4
type_trend_csv = Path('../results/rq2/type_trend_analysis.csv')
if type_trend_csv.exists():
    with open(type_trend_csv) as f:
        type_rows = list(csv.DictReader(f))
    family_b_raw = [float(r['p_value']) for r in type_rows]
    family_b_names = [r['task_type'] for r in type_rows]
    reject_b, padj_b, _, _ = multipletests(family_b_raw, alpha=0.05, method='holm')
    print("\nFamily B (Table 2, Holm, n=4):")
    for name, raw, adj in zip(family_b_names, family_b_raw, padj_b):
        sig_raw = "***" if raw < 0.001 else ("**" if raw < 0.01 else ("*" if raw < 0.05 else "n.s."))
        sig_adj = "***" if adj < 0.001 else ("**" if adj < 0.01 else ("*" if adj < 0.05 else "n.s."))
        change = f" CHANGED" if sig_raw != sig_adj else ""
        print(f"  {name:15s}: raw p={raw:.6f} ({sig_raw}) -> adj p={adj:.6f} ({sig_adj}){change}")

# Family C (Table 3): Holm-Bonferroni, n=6
merge_factors_csv = Path('../results/rq2/merge_factors.csv')
if merge_factors_csv.exists():
    with open(merge_factors_csv) as f:
        mf_rows = list(csv.DictReader(f))
    family_c_raw = [float(r['p_value']) for r in mf_rows]
    family_c_names = [r['metric'] for r in mf_rows]
    reject_c, padj_c, _, _ = multipletests(family_c_raw, alpha=0.05, method='holm')
    print("\nFamily C (Table 3, Holm, n=6):")
    for name, raw, adj in zip(family_c_names, family_c_raw, padj_c):
        sig_raw = "***" if raw < 0.001 else ("**" if raw < 0.01 else ("*" if raw < 0.05 else "n.s."))
        sig_adj = "***" if adj < 0.001 else ("**" if adj < 0.01 else ("*" if adj < 0.05 else "n.s."))
        change = f" CHANGED" if sig_raw != sig_adj else ""
        print(f"  {name:20s}: raw p={raw:.6f} ({sig_raw}) -> adj p={adj:.6f} ({sig_adj}){change}")

# Family D (30 repos): Benjamini-Hochberg
repo_trends_csv = Path('../results/rq1/repository_gfi_trends.csv')
if repo_trends_csv.exists():
    with open(repo_trends_csv) as f:
        repo_rows = list(csv.DictReader(f))
    if 'p_adjusted' in repo_rows[0]:
        dec = sum(1 for r in repo_rows if r.get('trend_corrected') == 'decreasing')
        inc = sum(1 for r in repo_rows if r.get('trend_corrected') == 'increasing')
        no_trend = sum(1 for r in repo_rows if r.get('trend_corrected') == 'no trend')
        total = len(repo_rows)
        print(f"\nFamily D (30 repos, BH corrected):")
        print(f"  Decreasing: {dec} ({dec/total*100:.1f}%)  [paper: 7 (23.3%)]")
        print(f"  No trend:   {no_trend} ({no_trend/total*100:.1f}%)  [paper: 21 (70.0%)]")
        print(f"  Increasing: {inc} ({inc/total*100:.1f}%)  [paper: 2 (6.7%)]")

        match_d = "✓" if dec == 7 and inc == 2 and no_trend == 21 else "✗"
        print(f"  {match_d} Matches paper values")
    else:
        print("\nFamily D: p_adjusted column not found. Run rq1_analysis.py first.")

print("\n===== 検証完了 =====")
