"""
Sensitivity Analysis for RQ2: Overlapping Task-Type Label Handling

Compares three classification strategies for the 47 PRs whose issues carry
labels matching more than one task type (Bug+Doc, Feature+Doc, Bug+Feature):

  Strategy A (baseline): Manual classification by first author
                         (distinguishing type labels from area labels)
  Strategy B (exclusion): Exclude all 47 overlapping PRs from the analysis
  Strategy C (file-based): Classify by actual files changed in the PR
                            (doc-only PR → Documentation; code-present PR →
                             Bug or Feature based on issue labels alone)

Output: results/rq2/sensitivity_analysis/
  - classification_comparison.csv   side-by-side A/B/C for each PR
  - merge_rate_by_type_*.csv        merge rate by task type for each strategy
  - yearly_merge_rate_*.csv         year-by-year merge rate for each strategy
  - summary_statistics.csv          top-level numbers for all three strategies
"""

import json
import re
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import kendalltau

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GFI_PRS_DIR   = Path('../data/gfi_prs')
OVERLAP_CSV   = Path('../results/rq2/overlapping_task_type_labels.csv')
OUTPUT_DIR    = Path('../results/rq2/sensitivity_analysis')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

START_MONTH = '2021-07'
END_MONTH   = '2025-06'

# Year boundaries used in the paper
YEAR_RANGES = {
    'Y1': ('2021-07', '2022-06'),
    'Y2': ('2022-07', '2023-06'),
    'Y3': ('2023-07', '2024-06'),
    'Y4': ('2024-07', '2025-06'),
}

# File extensions considered documentation
DOC_EXTS = {
    'md', 'rst', 'txt', 'adoc', 'asciidoc',
    'ipynb', 'html', 'htm', 'tex', 'rdoc', 'pod', 'wiki',
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def month_to_year(month: str):
    for label, (start, end) in YEAR_RANGES.items():
        if start <= month <= end:
            return label
    return None


def classify_automatic(labels):
    """Label-keyword classification (no disambiguation for overlaps)."""
    ll = [l.lower() for l in labels]
    has_bug     = any(re.search(r'\bbug\b', l) for l in ll)
    has_feature = any('feature' in l or 'enhancement' in l for l in ll)
    has_doc     = any('doc' in l for l in ll)

    if has_bug:
        return 'bug'
    elif has_feature:
        return 'feature'
    elif has_doc:
        return 'documentation'
    return 'other'


def classify_by_files(pr_data, issue_labels):
    """
    Classify a PR by the types of files it changes.

    Rules
    -----
    1. If ALL changed files have doc-type extensions → 'documentation'
    2. Otherwise (at least one code file) → use issue labels to pick
       'bug' or 'feature', ignoring any doc-area labels.
    """
    file_types = pr_data.get('file_types', {})
    if not file_types:
        return classify_automatic(issue_labels)   # fallback

    total = sum(file_types.values())
    n_doc = sum(v for k, v in file_types.items() if k in DOC_EXTS)
    doc_ratio = n_doc / total if total else 0.0

    if doc_ratio == 1.0:
        return 'documentation'

    # Code is present → distinguish bug vs feature via labels
    ll = [l.lower() for l in issue_labels]
    has_bug     = any(re.search(r'\bbug\b', l) for l in ll)
    has_feature = any('feature' in l or 'enhancement' in l for l in ll)

    if has_bug:
        return 'bug'
    elif has_feature:
        return 'feature'
    return 'other'

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_manual_classifications():
    classifications = {}
    if not OVERLAP_CSV.exists():
        return classifications
    with open(OVERLAP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            repo    = row['repository']
            pr_num  = int(row['pr_number'])
            manual  = row.get('classification', '').strip().lower()
            if manual:
                classifications[(repo, pr_num)] = manual
    return classifications


def load_overlapping_set():
    """Return (repo, pr_number) pairs for all 47 overlapping PRs."""
    result = set()
    if not OVERLAP_CSV.exists():
        return result
    with open(OVERLAP_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            result.add((row['repository'], int(row['pr_number'])))
    return result


def load_gfi_prs():
    """Load every newcomer GFI PR within the analysis window."""
    all_prs = []
    for file in GFI_PRS_DIR.glob('*_gfi_prs.json'):
        if 'experienced' in file.name:
            continue
        repo = file.stem.replace('_gfi_prs', '')
        with open(file, encoding='utf-8') as f:
            records = json.load(f)

        for rec in records:
            pr = rec.get('pr_data', {})
            if not pr:
                continue

            created_at = pr.get('created_at', '')
            if len(created_at) < 7:
                continue
            month = created_at[:7]
            if not (START_MONTH <= month <= END_MONTH):
                continue

            issue_labels = []
            for ir in rec.get('referenced_gfi_issues', []):
                issue_labels.extend(ir.get('labels', []))

            all_prs.append({
                'pr_data':     pr,
                'issue_labels': issue_labels,
                'repository':  repo,
                'month':       month,
            })
    return all_prs

# ---------------------------------------------------------------------------
# Build analysis DataFrame under each strategy
# ---------------------------------------------------------------------------

def build_df(prs, manual_cls, overlapping, strategy):
    """
    strategy: 'A' | 'B' | 'C'
    """
    rows = []
    for rec in prs:
        pr     = rec['pr_data']
        repo   = rec['repository']
        pr_num = pr.get('number')
        month  = rec['month']
        labels = rec['issue_labels']

        ll = [l.lower() for l in labels]
        has_bug     = any(re.search(r'\bbug\b', l) for l in ll)
        has_feature = any('feature' in l or 'enhancement' in l for l in ll)
        has_doc     = any('doc' in l for l in ll)
        is_overlap  = (repo, pr_num) in overlapping

        # ── Strategy B: skip overlapping PRs ─────────────────────────────
        if strategy == 'B' and is_overlap:
            continue

        # ── Classify ─────────────────────────────────────────────────────
        matched = sum([has_bug, has_feature, has_doc])

        if strategy == 'A':
            if matched >= 2 and is_overlap:
                key = (repo, pr_num)
                task_type = manual_cls.get(key, classify_automatic(labels))
            else:
                task_type = classify_automatic(labels)

        elif strategy == 'B':
            task_type = classify_automatic(labels)   # no overlaps present

        elif strategy == 'C':
            if matched >= 2 and is_overlap:
                task_type = classify_by_files(pr, labels)
            else:
                task_type = classify_automatic(labels)

        else:
            raise ValueError(f'Unknown strategy: {strategy}')

        additions = pr.get('additions', 0) or 0
        deletions = pr.get('deletions', 0) or 0

        rows.append({
            'month':       month,
            'year':        month_to_year(month),
            'repository':  repo,
            'pr_number':   pr_num,
            'merged':      1 if pr.get('merged', False) else 0,
            'task_type':   task_type,
            'pr_size':     additions + deletions,
            'body_length': pr.get('body_cleaned_length', pr.get('body_length', 0)) or 0,
            'review_count': pr.get('review_count', 0) or 0,
            'is_overlap':  is_overlap,
        })

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def merge_rate_by_type(df):
    g = df.groupby('task_type').agg(
        pr_count  = ('merged', 'count'),
        merged    = ('merged', 'sum'),
        merge_rate = ('merged', 'mean'),
    ).reset_index()
    g['merge_rate'] = (g['merge_rate'] * 100).round(1)
    return g.sort_values('task_type')


def yearly_merge_rate(df):
    rows = []
    for year, (s, e) in YEAR_RANGES.items():
        sub = df[(df['month'] >= s) & (df['month'] <= e)]
        if len(sub) == 0:
            continue
        rows.append({
            'year':       year,
            'pr_count':   len(sub),
            'merged':     sub['merged'].sum(),
            'merge_rate': round(sub['merged'].mean() * 100, 1),
        })
    return pd.DataFrame(rows)


def yearly_merge_rate_by_type(df):
    rows = []
    for year, (s, e) in YEAR_RANGES.items():
        sub = df[(df['month'] >= s) & (df['month'] <= e)]
        for tt, grp in sub.groupby('task_type'):
            rows.append({
                'year':       year,
                'task_type':  tt,
                'pr_count':   len(grp),
                'merged':     grp['merged'].sum(),
                'merge_rate': round(grp['merged'].mean() * 100, 1),
            })
    return pd.DataFrame(rows)


def mann_kendall(values):
    n = len(values)
    if n < 3:
        return np.nan, np.nan, 'insufficient data'
    tau, p = kendalltau(np.arange(n), values)
    if p < 0.05:
        trend = 'increasing' if tau > 0 else 'decreasing'
    else:
        trend = 'no trend'
    return tau, p, trend

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print('Loading data …')
    prs         = load_gfi_prs()
    manual_cls  = load_manual_classifications()
    overlapping = load_overlapping_set()

    print(f'  Total PRs in window: {len(prs)}')
    print(f'  Overlapping PRs:     {len(overlapping)}')
    print(f'  Manual classifications loaded: {len(manual_cls)}\n')

    strategy_labels = {
        'A': 'Manual classification (baseline)',
        'B': 'Exclude overlapping PRs',
        'C': 'File-change-based classification',
    }

    dfs = {}
    for s in ('A', 'B', 'C'):
        dfs[s] = build_df(prs, manual_cls, overlapping, s)
        print(f'Strategy {s} ({strategy_labels[s]}): {len(dfs[s])} PRs')

    # ── 1. Side-by-side classification comparison ─────────────────────────
    # For each overlapping PR, show how A, B, C classify it
    cmp_rows = []
    for (repo, pr_num) in sorted(overlapping):
        row_A = dfs['A'][(dfs['A']['repository'] == repo) &
                          (dfs['A']['pr_number']  == pr_num)]
        row_C = dfs['C'][(dfs['C']['repository'] == repo) &
                          (dfs['C']['pr_number']  == pr_num)]

        cls_A = row_A['task_type'].values[0] if len(row_A) else 'n/a'
        cls_C = row_C['task_type'].values[0] if len(row_C) else 'n/a'

        cmp_rows.append({
            'repository': repo,
            'pr_number':  pr_num,
            'strategy_A_manual':    cls_A,
            'strategy_B':           'excluded',
            'strategy_C_filebased': cls_C,
            'A_C_agree':            cls_A == cls_C,
        })

    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUTPUT_DIR / 'classification_comparison.csv', index=False)

    n_agree = cmp_df['A_C_agree'].sum()
    print(f'\nA vs C agreement on overlapping PRs: {n_agree}/{len(cmp_df)} '
          f'({n_agree/len(cmp_df)*100:.1f}%)')

    # ── 2. Merge rate by task type ────────────────────────────────────────
    all_type_stats = []
    for s, df in dfs.items():
        t = merge_rate_by_type(df)
        t.insert(0, 'strategy', s)
        all_type_stats.append(t)
        t.to_csv(OUTPUT_DIR / f'merge_rate_by_type_{s}.csv', index=False)

    pd.concat(all_type_stats).to_csv(
        OUTPUT_DIR / 'merge_rate_by_type_ALL.csv', index=False)

    # ── 3. Yearly merge rate (overall) ───────────────────────────────────
    all_yearly = []
    for s, df in dfs.items():
        y = yearly_merge_rate(df)
        y.insert(0, 'strategy', s)
        all_yearly.append(y)
        y.to_csv(OUTPUT_DIR / f'yearly_merge_rate_{s}.csv', index=False)

    pd.concat(all_yearly).to_csv(
        OUTPUT_DIR / 'yearly_merge_rate_ALL.csv', index=False)

    # ── 4. Yearly merge rate by task type ─────────────────────────────────
    for s, df in dfs.items():
        yearly_merge_rate_by_type(df).to_csv(
            OUTPUT_DIR / f'yearly_by_type_{s}.csv', index=False)

    # ── 5. Mann-Kendall on monthly overall merge rate ─────────────────────
    mk_rows = []
    for s, df in dfs.items():
        monthly = (df.groupby('month')['merged']
                     .mean()
                     .reset_index()
                     .sort_values('month'))
        tau, p, trend = mann_kendall(monthly['merged'].values)
        mk_rows.append({
            'strategy': s,
            'description': strategy_labels[s],
            'n_months': len(monthly),
            'kendall_tau': round(tau, 4) if not np.isnan(tau) else np.nan,
            'p_value': round(p, 4) if not np.isnan(p) else np.nan,
            'trend': trend,
        })

    pd.DataFrame(mk_rows).to_csv(OUTPUT_DIR / 'mann_kendall_overall.csv', index=False)

    # ── 6. Summary statistics table ───────────────────────────────────────
    summary_rows = []
    for s, df in dfs.items():
        yearly = yearly_merge_rate(df)
        y1 = yearly[yearly['year'] == 'Y1']['merge_rate'].values
        y4 = yearly[yearly['year'] == 'Y4']['merge_rate'].values
        summary_rows.append({
            'strategy':    s,
            'description': strategy_labels[s],
            'n_prs':       len(df),
            'n_overlapping_included': (df['is_overlap'].sum()
                                       if 'is_overlap' in df.columns else 0),
            'overall_merge_rate': round(df['merged'].mean() * 100, 1),
            'Y1_merge_rate': float(y1[0]) if len(y1) else np.nan,
            'Y4_merge_rate': float(y4[0]) if len(y4) else np.nan,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(OUTPUT_DIR / 'summary_statistics.csv', index=False)

    # ── Print results ─────────────────────────────────────────────────────
    print('\n' + '=' * 70)
    print('SUMMARY STATISTICS')
    print('=' * 70)
    print(summary_df.to_string(index=False))

    print('\n' + '=' * 70)
    print('MERGE RATE BY TASK TYPE')
    print('=' * 70)
    for s in ('A', 'B', 'C'):
        print(f'\n  Strategy {s}: {strategy_labels[s]}')
        t = merge_rate_by_type(dfs[s])
        print(t.to_string(index=False))

    print('\n' + '=' * 70)
    print('YEARLY MERGE RATE (ALL TASK TYPES COMBINED)')
    print('=' * 70)
    combined = pd.concat(all_yearly)
    pivot = combined.pivot_table(index='year',
                                 columns='strategy',
                                 values='merge_rate')
    pivot.columns = [f'Strategy_{c}' for c in pivot.columns]
    print(pivot.to_string())

    print('\n' + '=' * 70)
    print('MANN-KENDALL (monthly overall merge rate)')
    print('=' * 70)
    print(pd.DataFrame(mk_rows).to_string(index=False))

    print(f'\nAll results saved to: {OUTPUT_DIR.resolve()}')


if __name__ == '__main__':
    main()
