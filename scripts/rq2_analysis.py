import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt


class RQ2Analyzer:
    def __init__(self):
        self.gfi_prs_dir = Path('../data/gfi_prs')
        self.output_dir = Path('../results/rq2')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Analysis period
        self.start_month = '2021-07'
        self.end_month = '2025-06'

        # Load manual classifications for overlapping PRs
        self.manual_classifications = self._load_manual_classifications()

    def _load_manual_classifications(self):
        """Load manual classifications for PRs with overlapping task type labels."""
        import csv
        csv_path = self.output_dir / 'overlapping_task_type_labels.csv'
        classifications = {}
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    repo = row['repository']
                    pr_num = int(row['pr_number'])
                    manual = row.get('manual_classification', '').strip().lower()
                    if manual:
                        classifications[(repo, pr_num)] = manual
        return classifications

    def _classify_task_type(self, labels, repo=None, pr_number=None):
        """Classify task type from issue labels.

        For PRs matching a single type, automatic classification is used.
        For PRs matching multiple types (47 cases, 4.2%), the first author
        manually classified each case by distinguishing type labels
        (e.g., 'bug', 'type: bug', 'C-bug') from area labels
        (e.g., 'addon: docs', 'A-docs', 'module: docs').

        Uses word-boundary matching for 'bug' to avoid false positives
        (e.g., 'debug' should not match 'bug')."""
        labels_lower = [l.lower() for l in labels]
        has_bug = any(re.search(r'\bbug\b', l) for l in labels_lower)
        has_feature = any('feature' in l or 'enhancement' in l for l in labels_lower)
        has_doc = any('doc' in l for l in labels_lower)

        matched = sum([has_bug, has_feature, has_doc])

        if matched >= 2 and repo and pr_number:
            # Use manual classification for overlapping cases
            key = (repo, pr_number)
            if key in self.manual_classifications:
                return self.manual_classifications[key]

        # Single-type or fallback: automatic classification
        if has_bug:
            return 'bug'
        elif has_feature:
            return 'feature'
        elif has_doc:
            return 'documentation'
        else:
            return 'other'

    def _add_moving_averages(self, df, value_col, windows=[3, 6]):
        """Add moving averages to a DataFrame."""
        df = df.sort_values('month')
        for window in windows:
            col_name = f'{value_col}_ma{window}'
            df[col_name] = df[value_col].rolling(window=window, min_periods=1).mean()
        return df

    def _mann_kendall_test(self, values):
        """Perform Mann-Kendall trend test."""
        n = len(values)
        if n < 3:
            return None, None, 'insufficient data'

        time_index = np.arange(n)
        tau, p_value = kendalltau(time_index, values)

        if p_value < 0.05:
            trend = 'increasing' if tau > 0 else 'decreasing'
        else:
            trend = 'no trend'

        return tau, p_value, trend

    def load_gfi_prs(self):
        """Load all GFI PRs with issue labels."""
        all_prs = []
        for file in self.gfi_prs_dir.glob('*_gfi_prs.json'):
            if 'experienced' in file.name:
                continue

            repo = file.stem.replace('_gfi_prs', '')
            with open(file, 'r', encoding='utf-8') as f:
                prs = json.load(f)

            for pr_record in prs:
                pr_data = pr_record.get('pr_data', {})
                if not pr_data:
                    continue

                # Get labels from referenced GFI issues
                issue_labels = []
                for issue_ref in pr_record.get('referenced_gfi_issues', []):
                    issue_labels.extend(issue_ref.get('labels', []))

                all_prs.append({
                    'pr_data': pr_data,
                    'issue_labels': issue_labels,
                    'repository': repo
                })

        return all_prs

    def analyze_characteristics(self):
        """Analyze GFI PR characteristics with monthly trends."""
        print('RQ2: GFI PR Characteristics Analysis\n')

        prs = self.load_gfi_prs()
        print(f'Total GFI PRs loaded: {len(prs)}')

        data = []
        for pr_record in prs:
            pr = pr_record['pr_data']
            created_at = pr.get('created_at', '')

            if len(created_at) < 7:
                continue

            month = created_at[:7]
            if not (self.start_month <= month <= self.end_month):
                continue

            additions = pr.get('additions', 0) or 0
            deletions = pr.get('deletions', 0) or 0
            insertions_log = np.log(additions + 1) if additions >= 0 else 0
            deletions_log = np.log(deletions + 1) if deletions >= 0 else 0

            time_to_merge_hours = pr.get('time_to_merge_hours')
            time_to_merge_days = time_to_merge_hours / 24 if time_to_merge_hours else None

            # Classify task type
            pr_number = pr.get('number')
            task_type = self._classify_task_type(
                pr_record['issue_labels'],
                repo=pr_record['repository'],
                pr_number=pr_number
            )

            data.append({
                'month': month,
                'repository': pr_record['repository'],
                'merged': 1 if pr.get('merged', False) else 0,
                'pr_size': additions + deletions,
                'insertions_log': insertions_log,
                'deletions_log': deletions_log,
                'files_changed': pr.get('files_changed', 0) or 0,
                'commits_count': pr.get('commits_count', 0) or 0,
                'review_count': pr.get('review_count', 0) or 0,
                'body_length': pr.get('body_cleaned_length', pr.get('body_length', 0)) or 0,
                'time_to_merge_days': time_to_merge_days,
                'task_type': task_type
            })

        df = pd.DataFrame(data)
        print(f'PRs in analysis period: {len(df)}')

        # Save raw data
        df.to_csv(self.output_dir / 'gfi_pr_characteristics.csv', index=False)

        # Overall statistics
        overall_merge_rate = df['merged'].mean() * 100
        print(f'\nOverall merge rate: {overall_merge_rate:.1f}%')

        return df

    def analyze_monthly_trends(self, df):
        """Analyze monthly trends in PR metrics."""
        print('\n' + '=' * 60)
        print('Monthly PR Metrics Trends')
        print('=' * 60)

        # Calculate monthly aggregates
        monthly = df.groupby('month').agg({
            'merged': 'mean',
            'body_length': 'median',
            'review_count': 'median',
            'time_to_merge_days': 'median',
            'pr_size': 'median',
            'repository': 'count'  # Use as PR count
        }).rename(columns={'repository': 'pr_count', 'merged': 'merge_rate'})

        monthly['merge_rate'] = monthly['merge_rate'] * 100
        monthly = monthly.reset_index()

        # Add moving averages
        for col in ['merge_rate', 'body_length', 'review_count']:
            monthly = self._add_moving_averages(monthly, col, [3, 6])

        # Trend tests for key metrics
        trend_results = []
        for metric in ['merge_rate', 'body_length', 'review_count']:
            values = monthly[metric].dropna().values
            tau, p_value, trend = self._mann_kendall_test(values)

            print(f'\n{metric}:')
            print(f'  Start: {values[0]:.2f}, End: {values[-1]:.2f}')
            print(f'  Kendall tau: {tau:.4f}, p-value: {p_value:.4f}')
            print(f'  Trend: {trend}')

            trend_results.append({
                'metric': metric,
                'start_value': values[0],
                'end_value': values[-1],
                'kendall_tau': tau,
                'p_value': p_value,
                'trend': trend
            })

        # Save results
        monthly.to_csv(self.output_dir / 'monthly_pr_metrics.csv', index=False)
        pd.DataFrame(trend_results).to_csv(self.output_dir / 'trend_statistics.csv', index=False)

        return monthly, trend_results

    def analyze_by_task_type(self, df):
        """Analyze merge rates by task type (from issue labels)."""
        print('\n' + '=' * 60)
        print('Task Type Analysis (from Issue Labels)')
        print('=' * 60)

        # Overall statistics by task type
        type_stats = df.groupby('task_type').agg({
            'merged': ['count', 'sum', 'mean'],
            'body_length': 'median',
            'review_count': 'median'
        }).round(3)

        type_stats.columns = ['pr_count', 'merged_count', 'merge_rate', 'body_length_median', 'review_count_median']
        type_stats['merge_rate'] = type_stats['merge_rate'] * 100
        type_stats = type_stats.reset_index()

        print('\nOverall by Task Type:')
        print(type_stats.to_string(index=False))

        # Save overall stats
        type_stats.to_csv(self.output_dir / 'merge_rate_by_type.csv', index=False)

        # Monthly trends by task type
        monthly_by_type = df.groupby(['month', 'task_type']).agg({
            'merged': ['count', 'mean']
        }).reset_index()
        monthly_by_type.columns = ['month', 'task_type', 'pr_count', 'merge_rate']
        monthly_by_type['merge_rate'] = monthly_by_type['merge_rate'] * 100

        # Trend test for each task type
        print('\nTrends by Task Type:')
        type_trends = []
        for task_type in df['task_type'].unique():
            type_data = monthly_by_type[monthly_by_type['task_type'] == task_type].sort_values('month')
            if len(type_data) >= 6:  # Need at least 6 months
                values = type_data['merge_rate'].values
                tau, p_value, trend = self._mann_kendall_test(values)

                print(f'\n  {task_type}:')
                print(f'    Start: {values[0]:.1f}%, End: {values[-1]:.1f}%')
                print(f'    Kendall tau: {tau:.4f}, p-value: {p_value:.4f}, Trend: {trend}')

                type_trends.append({
                    'task_type': task_type,
                    'start_merge_rate': values[0],
                    'end_merge_rate': values[-1],
                    'kendall_tau': tau,
                    'p_value': p_value,
                    'trend': trend
                })

        # Apply Holm-Bonferroni correction (Family B: n=4 task type trend tests)
        if type_trends:
            raw_pvalues = [t['p_value'] for t in type_trends]
            reject, p_adjusted, _, _ = multipletests(raw_pvalues, alpha=0.05, method='holm')
            for i, t in enumerate(type_trends):
                t['p_adjusted'] = p_adjusted[i]
                if p_adjusted[i] < 0.05:
                    t['trend_corrected'] = 'increasing' if t['kendall_tau'] > 0 else 'decreasing'
                else:
                    t['trend_corrected'] = 'no trend'

            print(f'\n  Holm-Bonferroni corrected (n={len(type_trends)}):')
            for t in type_trends:
                sig_raw = '***' if t['p_value'] < 0.001 else ('**' if t['p_value'] < 0.01 else ('*' if t['p_value'] < 0.05 else 'n.s.'))
                sig_adj = '***' if t['p_adjusted'] < 0.001 else ('**' if t['p_adjusted'] < 0.01 else ('*' if t['p_adjusted'] < 0.05 else 'n.s.'))
                print(f'    {t["task_type"]}: raw p={t["p_value"]:.4f} ({sig_raw}) -> adj p={t["p_adjusted"]:.4f} ({sig_adj})')

        # Save type trends
        pd.DataFrame(type_trends).to_csv(self.output_dir / 'type_trend_analysis.csv', index=False)
        monthly_by_type.to_csv(self.output_dir / 'monthly_metrics_by_type.csv', index=False)

        return type_stats, monthly_by_type

    def analyze_merge_factors(self, df):
        """Analyze factors associated with merge success."""
        print('\n' + '=' * 60)
        print('Merge Success Factors Analysis')
        print('=' * 60)

        df_merged = df[df['merged'] == 1]
        df_not_merged = df[df['merged'] == 0]

        print(f'\nMerged PRs: {len(df_merged)}')
        print(f'Not Merged PRs: {len(df_not_merged)}')

        metrics = ['insertions_log', 'deletions_log', 'files_changed', 'commits_count', 'review_count', 'body_length']
        results = []

        print('\nMetric Comparison (Merged vs Not Merged):')
        print('-' * 70)

        for metric in metrics:
            merged_values = df_merged[metric].dropna()
            not_merged_values = df_not_merged[metric].dropna()

            if len(merged_values) < 2 or len(not_merged_values) < 2:
                continue

            merged_median = merged_values.median()
            not_merged_median = not_merged_values.median()
            stat, pval = stats.mannwhitneyu(merged_values, not_merged_values, alternative='two-sided')

            sig = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
            print(f'{metric:20s} | Merged: {merged_median:8.1f} | Not Merged: {not_merged_median:8.1f} | p={pval:.4f} {sig}')

            results.append({
                'metric': metric,
                'merged_median': merged_median,
                'not_merged_median': not_merged_median,
                'mann_whitney_u': stat,
                'p_value': pval,
                'significant': 'Yes' if pval < 0.05 else 'No'
            })

        results_df = pd.DataFrame(results)

        # Apply Holm-Bonferroni correction (Family C: n=6 merge factor tests)
        if len(results_df) > 0:
            raw_pvalues = results_df['p_value'].values
            reject, p_adjusted, _, _ = multipletests(raw_pvalues, alpha=0.05, method='holm')
            results_df['p_adjusted'] = p_adjusted
            results_df['significant_corrected'] = ['Yes' if r else 'No' for r in reject]

            print(f'\n  Holm-Bonferroni corrected (n={len(results_df)}):')
            for _, row in results_df.iterrows():
                sig_raw = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else 'n.s.'))
                sig_adj = '***' if row['p_adjusted'] < 0.001 else ('**' if row['p_adjusted'] < 0.01 else ('*' if row['p_adjusted'] < 0.05 else 'n.s.'))
                print(f'    {row["metric"]:20s}: raw p={row["p_value"]:.6f} ({sig_raw}) -> adj p={row["p_adjusted"]:.6f} ({sig_adj})')

        results_df.to_csv(self.output_dir / 'merge_factors.csv', index=False)

        # Task type effect on merge
        print('\nMerge Rate by Task Type:')
        type_merge = df.groupby('task_type')['merged'].agg(['count', 'mean']).round(3)
        type_merge.columns = ['count', 'merge_rate']
        type_merge['merge_rate'] = type_merge['merge_rate'] * 100
        print(type_merge)

        return results_df

    def run(self):
        print('=' * 60)
        print('RQ2 Analysis: GFI PR Characteristics and Task Types')
        print('=' * 60)

        df = self.analyze_characteristics()
        self.analyze_monthly_trends(df)
        self.analyze_by_task_type(df)
        self.analyze_merge_factors(df)

        print(f'\n{"=" * 60}')
        print(f'Results saved to: {self.output_dir.resolve()}')


if __name__ == '__main__':
    analyzer = RQ2Analyzer()
    analyzer.run()
