"""
変化点検出分析: GFI比率の時系列データに対して構造的変化点を特定する。

手法:
1. Pettitt検定 (ノンパラメトリック、単一変化点)
2. PELT (Pruned Exact Linear Time) - ruptures
3. BinSeg (Binary Segmentation) - ruptures
"""
import numpy as np
import pandas as pd
from scipy import stats as sp

# ===== Pettitt test implementation =====
def pettitt_test(data):
    """
    Pettitt's test for a single change-point in a time series.
    H0: No change point exists.
    Returns: (change_point_index, test_statistic K, p_value)
    """
    n = len(data)
    # Compute U statistics
    U = np.zeros(n, dtype=np.float64)
    for t in range(n):
        for j in range(n):
            U[t] += np.sign(data[t] - data[j])

    # Cumulative sum approach (more efficient)
    # U_t = 2 * sum of ranks up to t - t*(n+1)
    # But let's use the direct Mann-Whitney-like formulation
    U_cum = np.zeros(n)
    for t in range(1, n):
        s = 0
        for i in range(t):
            for j in range(t, n):
                s += np.sign(data[i] - data[j])
        U_cum[t] = abs(s)

    K = np.max(U_cum)
    t_star = np.argmax(U_cum)

    # Approximate p-value
    p_value = 2.0 * np.exp(-6.0 * K**2 / (n**3 + n**2))
    p_value = min(p_value, 1.0)

    return t_star, K, p_value


def pettitt_test_fast(data):
    """Faster Pettitt test using rank-based approach."""
    n = len(data)
    ranks = sp.rankdata(data)

    # U_t = 2 * sum(ranks[0:t]) - t*(n+1)
    cum_ranks = np.cumsum(ranks)
    t_indices = np.arange(1, n + 1)
    U = 2 * cum_ranks - t_indices * (n + 1)

    K = np.max(np.abs(U))
    t_star = np.argmax(np.abs(U))

    # Approximate p-value
    p_value = 2.0 * np.exp(-6.0 * K**2 / (n**3 + n**2))
    p_value = min(p_value, 1.0)

    return t_star, K, p_value


# ===== Load data =====
df = pd.read_csv('../results/rq1/monthly_gfi_trend.csv')
values = df['gfi_ratio'].values
months = df['month'].values

print("=" * 65)
print("変化点検出分析: Monthly GFI Ratio")
print("=" * 65)
print(f"データ: {len(values)} months ({months[0]} to {months[-1]})")
print(f"平均: {values.mean():.4f}%, 標準偏差: {values.std():.4f}%")
print()

# ===== 1. Pettitt test =====
print("--- 1. Pettitt検定 (単一変化点, ノンパラメトリック) ---")
t_star, K, p_value = pettitt_test_fast(values)
print(f"  変化点: index {t_star} = {months[t_star]}")
print(f"  検定統計量 K = {K:.2f}")
print(f"  p-value = {p_value:.6f}")
print(f"  有意性: {'有意 (p < 0.05)' if p_value < 0.05 else '非有意'}")

# Before/after statistics
before = values[:t_star + 1]
after = values[t_star + 1:]
print(f"  変化点前 ({months[0]}--{months[t_star]}): 平均={before.mean():.4f}%, n={len(before)}")
print(f"  変化点後 ({months[t_star+1]}--{months[-1]}): 平均={after.mean():.4f}%, n={len(after)}")
print()

# ===== 2. ruptures: PELT =====
print("--- 2. PELT (Pruned Exact Linear Time) ---")
import ruptures as rpt

# Try different penalty values
for pen in [1.0, 2.0, 3.0, 5.0]:
    algo = rpt.Pelt(model="rbf", min_size=6).fit(values)
    result = algo.predict(pen=pen)
    # result includes the final index (n), remove it
    breakpoints = [r for r in result if r < len(values)]
    bp_months = [months[bp] for bp in breakpoints]
    print(f"  penalty={pen}: breakpoints at indices {breakpoints} = {bp_months}")

print()

# ===== 3. ruptures: BinSeg with 1 changepoint =====
print("--- 3. Binary Segmentation (1 changepoint) ---")
algo = rpt.Binseg(model="l2", min_size=6).fit(values)
result = algo.predict(n_bkps=1)
bp = result[0] if result[0] < len(values) else result[0] - 1
print(f"  変化点: index {bp} = {months[min(bp, len(months)-1)]}")

before_bs = values[:bp]
after_bs = values[bp:]
print(f"  変化点前: 平均={before_bs.mean():.4f}%, n={len(before_bs)}")
print(f"  変化点後: 平均={after_bs.mean():.4f}%, n={len(after_bs)}")
print()

# ===== 3b. BinSeg with 2 changepoints =====
print("--- 3b. Binary Segmentation (2 changepoints) ---")
result2 = algo.predict(n_bkps=2)
breakpoints2 = [r for r in result2 if r < len(values)]
bp_months2 = [months[bp] for bp in breakpoints2]
print(f"  変化点: indices {breakpoints2} = {bp_months2}")
for i, bp in enumerate(breakpoints2):
    print(f"  BP{i+1}: {months[bp]} (index {bp})")
print()

# ===== 4. Year-boundary analysis =====
print("--- 4. 年度境界での平均比較 ---")
year_boundaries = {
    'Y1/Y2 (2022-07)': 12,
    'Y2/Y3 (2023-07)': 24,
    'Y3/Y4 (2024-07)': 36,
}

for label, split in year_boundaries.items():
    before_y = values[:split]
    after_y = values[split:]
    # Mann-Whitney U test for before vs after
    stat, pval = sp.mannwhitneyu(before_y, after_y, alternative='two-sided')
    print(f"  {label}: before={before_y.mean():.4f}% vs after={after_y.mean():.4f}% "
          f"(Mann-Whitney p={pval:.4f})")

print()

# ===== Summary =====
print("=" * 65)
print("まとめ")
print("=" * 65)
print(f"Pettitt検定: 変化点 = {months[t_star]} (p={p_value:.6f})")
print(f"  → Y3/Y4境界({months[36]})との差: {abs(t_star - 36)} months")

# Determine analysis year of changepoint
cp_month = months[t_star]
if cp_month < '2022-07':
    cp_year = 'Y1'
elif cp_month < '2023-07':
    cp_year = 'Y2'
elif cp_month < '2024-07':
    cp_year = 'Y3'
else:
    cp_year = 'Y4'
print(f"  → 分析年度: {cp_year}")
print(f"  → 論文の主張「Y4で急落」との整合性を確認")
