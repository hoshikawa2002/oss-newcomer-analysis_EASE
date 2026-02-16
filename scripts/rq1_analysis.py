import json
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from scipy import stats as scipy_stats
from scipy.stats import kendalltau
from statsmodels.stats.multitest import multipletests


class RQ1Analyzer:
    def __init__(self):
        self.gfi_issues_dir = Path('../data/gfi_issues')
        self.gfi_prs_dir = Path('../data/gfi_prs')
        self.newcomer_prs_dir = Path('../data/newcomer_prs')
        self.all_issues_dir = Path('../data/all_issue_labels')
        self.output_dir = Path('../results/rq1')
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Analysis period
        self.start_month = '2021-07'
        self.end_month = '2025-06'

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

    def analyze_monthly_gfi_trend(self):
        """Analyze monthly GFI ratio trends across all repositories."""
        print('\nRQ1-1: Monthly GFI Ratio Trend Analysis')

        monthly_stats = defaultdict(lambda: {'gfi_count': 0, 'total_count': 0})

        # Collect GFI issues by month
        for file in self.gfi_issues_dir.glob('*_gfi_issues.json'):
            with open(file, 'r', encoding='utf-8') as f:
                issues = json.load(f)

            for issue in issues:
                month = issue.get('created_at', '')[:7]
                if self.start_month <= month <= self.end_month:
                    monthly_stats[month]['gfi_count'] += 1

        # Collect all issues by month
        for file in self.all_issues_dir.glob('*_all_issues.json'):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                issues = data.get('issues', data) if isinstance(data, dict) else data

            for issue in issues:
                month = issue.get('created_at', '')[:7]
                if self.start_month <= month <= self.end_month:
                    monthly_stats[month]['total_count'] += 1

        # Create DataFrame
        df = pd.DataFrame([
            {
                'month': month,
                'gfi_count': stats['gfi_count'],
                'total_count': stats['total_count'],
                'gfi_ratio': (stats['gfi_count'] / stats['total_count'] * 100)
                             if stats['total_count'] > 0 else 0
            }
            for month, stats in sorted(monthly_stats.items())
        ])

        # Add moving averages
        df = self._add_moving_averages(df, 'gfi_ratio', [3, 6])

        # Perform Mann-Kendall trend test
        tau, p_value, trend = self._mann_kendall_test(df['gfi_ratio'].values)

        print(f'\nTotal months analyzed: {len(df)}')
        print(f'GFI ratio range: {df["gfi_ratio"].min():.2f}% - {df["gfi_ratio"].max():.2f}%')
        print(f'Start ({df["month"].iloc[0]}): {df["gfi_ratio"].iloc[0]:.2f}%')
        print(f'End ({df["month"].iloc[-1]}): {df["gfi_ratio"].iloc[-1]:.2f}%')
        print(f'\nMann-Kendall Trend Test:')
        print(f'  Kendall tau: {tau:.4f}')
        print(f'  p-value: {p_value:.4f}')
        print(f'  Trend: {trend}')

        # Save results
        df.to_csv(self.output_dir / 'monthly_gfi_trend.csv', index=False)

        # Save trend statistics
        trend_stats = pd.DataFrame([{
            'metric': 'GFI Ratio',
            'start_value': df['gfi_ratio'].iloc[0],
            'end_value': df['gfi_ratio'].iloc[-1],
            'kendall_tau': tau,
            'p_value': p_value,
            'trend': trend
        }])
        trend_stats.to_csv(self.output_dir / 'trend_statistics.csv', index=False)

        return df

    def analyze_newcomer_gfi_engagement(self):
        """Analyze newcomer engagement with GFI issues (Issue-centric approach).

        For each GFI issue created in a given month, check if any newcomer PR
        addressed it (regardless of when the PR was created).
        """
        print('\nRQ1-2: Newcomer GFI Engagement (Issue-centric)')

        # Step 1: Collect all GFI issues
        all_gfi_issues = {}  # {issue_key: {'month': ..., 'repo': ..., 'number': ...}}

        for file in self.gfi_issues_dir.glob('*_gfi_issues.json'):
            repo = file.stem.replace('_gfi_issues', '')
            with open(file, 'r', encoding='utf-8') as f:
                issues = json.load(f)

            for issue in issues:
                month = issue.get('created_at', '')[:7]
                if self.start_month <= month <= self.end_month:
                    issue_key = f"{repo}_{issue.get('number')}"
                    all_gfi_issues[issue_key] = {
                        'month': month,
                        'repo': repo,
                        'number': issue.get('number')
                    }

        print(f'Total GFI issues: {len(all_gfi_issues)}')

        # Step 2: Collect GFI issues that have newcomer PRs (within analysis period)
        addressed_issues = set()

        for file in self.gfi_prs_dir.glob('*_gfi_prs.json'):
            if 'experienced' in file.name:
                continue

            repo = file.stem.replace('_gfi_prs', '')
            with open(file, 'r', encoding='utf-8') as f:
                prs = json.load(f)

            for pr in prs:
                pr_month = pr.get('pr_data', {}).get('created_at', '')[:7]
                if not (self.start_month <= pr_month <= self.end_month):
                    continue

                for issue_ref in pr.get('referenced_gfi_issues', []):
                    issue_num = issue_ref.get('issue_number')
                    issue_key = f"{repo}_{issue_num}"
                    if issue_key in all_gfi_issues:
                        addressed_issues.add(issue_key)

        print(f'GFI issues with newcomer PRs: {len(addressed_issues)}')

        # Step 3: Calculate engagement by GFI creation month
        monthly_stats = defaultdict(lambda: {'total': 0, 'addressed': 0})

        for issue_key, info in all_gfi_issues.items():
            month = info['month']
            monthly_stats[month]['total'] += 1
            if issue_key in addressed_issues:
                monthly_stats[month]['addressed'] += 1

        # Create DataFrame
        df = pd.DataFrame([
            {
                'month': month,
                'gfi_created': stats['total'],
                'gfi_with_pr': stats['addressed'],
                'engagement_ratio': (stats['addressed'] / stats['total'] * 100)
                                    if stats['total'] > 0 else 0
            }
            for month, stats in sorted(monthly_stats.items())
        ])

        # Add moving averages
        df = self._add_moving_averages(df, 'engagement_ratio', [3, 6])

        # Perform Mann-Kendall trend test
        tau, p_value, trend = self._mann_kendall_test(df['engagement_ratio'].values)

        # Overall statistics
        overall_rate = len(addressed_issues) / len(all_gfi_issues) * 100

        print(f'\nOverall engagement rate: {overall_rate:.1f}%')
        print(f'Monthly range: {df["engagement_ratio"].min():.1f}% - {df["engagement_ratio"].max():.1f}%')
        print(f'\nMann-Kendall Trend Test:')
        print(f'  Kendall tau: {tau:.4f}')
        print(f'  p-value: {p_value:.4f}')
        print(f'  Trend: {trend}')

        # Yearly summary
        df['year'] = df['month'].str[:4]
        yearly = df.groupby('year').agg({
            'gfi_created': 'sum',
            'gfi_with_pr': 'sum'
        }).reset_index()
        yearly['engagement_rate'] = yearly['gfi_with_pr'] / yearly['gfi_created'] * 100

        print('\nYearly engagement rates:')
        for _, row in yearly.iterrows():
            print(f"  {row['year']}: {row['engagement_rate']:.1f}%")

        # Save results
        df.drop(columns=['year'], inplace=True)
        df.to_csv(self.output_dir / 'monthly_newcomer_engagement.csv', index=False)
        yearly.to_csv(self.output_dir / 'yearly_newcomer_engagement.csv', index=False)

        # Append to trend statistics
        trend_stats = pd.read_csv(self.output_dir / 'trend_statistics.csv')
        new_row = pd.DataFrame([{
            'metric': 'Newcomer Engagement',
            'start_value': overall_rate,  # Use overall rate instead of start/end
            'end_value': overall_rate,
            'kendall_tau': tau,
            'p_value': p_value,
            'trend': trend
        }])
        trend_stats = pd.concat([trend_stats, new_row], ignore_index=True)
        trend_stats.to_csv(self.output_dir / 'trend_statistics.csv', index=False)

        return df

    def analyze_repository_heterogeneity(self):
        """Analyze repository-level heterogeneity in GFI usage (simplified)."""
        print('\nRQ1-3: Repository Heterogeneity in GFI Trends')

        repo_trends = []

        for gfi_file in self.gfi_issues_dir.glob('*_gfi_issues.json'):
            repo = gfi_file.stem.replace('_gfi_issues', '')

            all_issues_file = self.all_issues_dir / f'{repo}_all_issues.json'
            if not all_issues_file.exists():
                continue

            with open(gfi_file, 'r', encoding='utf-8') as f:
                gfi_issues = json.load(f)

            with open(all_issues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                all_issues = data.get('issues', data) if isinstance(data, dict) else data

            # Calculate monthly GFI ratios for this repository
            monthly_data = defaultdict(lambda: {'gfi': 0, 'total': 0})

            for issue in gfi_issues:
                month = issue.get('created_at', '')[:7]
                if self.start_month <= month <= self.end_month:
                    monthly_data[month]['gfi'] += 1

            for issue in all_issues:
                month = issue.get('created_at', '')[:7]
                if self.start_month <= month <= self.end_month:
                    monthly_data[month]['total'] += 1

            if len(monthly_data) < 6:  # Need at least 6 months of data
                continue

            # Calculate trend
            sorted_months = sorted(monthly_data.keys())
            ratios = [
                (monthly_data[m]['gfi'] / monthly_data[m]['total'] * 100)
                if monthly_data[m]['total'] > 0 else 0
                for m in sorted_months
            ]

            tau, p_value, trend = self._mann_kendall_test(ratios)

            total_gfi = sum(monthly_data[m]['gfi'] for m in sorted_months)
            total_issues = sum(monthly_data[m]['total'] for m in sorted_months)

            repo_trends.append({
                'repository': repo,
                'total_gfi': total_gfi,
                'total_issues': total_issues,
                'overall_ratio': (total_gfi / total_issues * 100) if total_issues > 0 else 0,
                'kendall_tau': tau,
                'p_value': p_value,
                'trend': trend
            })

        df = pd.DataFrame(repo_trends)

        # Classify trends (uncorrected)
        increasing = len(df[(df['trend'] == 'increasing')])
        decreasing = len(df[(df['trend'] == 'decreasing')])
        stable = len(df[(df['trend'] == 'no trend')])

        print(f'\nRepositories analyzed: {len(df)}')
        print(f'\n--- Uncorrected ---')
        print(f'Increasing trend: {increasing} ({increasing/len(df)*100:.1f}%)')
        print(f'Decreasing trend: {decreasing} ({decreasing/len(df)*100:.1f}%)')
        print(f'No significant trend: {stable} ({stable/len(df)*100:.1f}%)')

        # Apply Benjamini-Hochberg correction for multiple testing (Family D)
        raw_pvalues = df['p_value'].values
        reject, p_adjusted, _, _ = multipletests(raw_pvalues, alpha=0.05, method='fdr_bh')
        df['p_adjusted'] = p_adjusted
        df['trend_corrected'] = df.apply(
            lambda row: ('increasing' if row['kendall_tau'] > 0 else 'decreasing')
            if row['p_adjusted'] < 0.05 else 'no trend', axis=1
        )

        increasing_corr = len(df[df['trend_corrected'] == 'increasing'])
        decreasing_corr = len(df[df['trend_corrected'] == 'decreasing'])
        stable_corr = len(df[df['trend_corrected'] == 'no trend'])

        print(f'\n--- Benjamini-Hochberg corrected (n={len(df)}) ---')
        print(f'Increasing trend: {increasing_corr} ({increasing_corr/len(df)*100:.1f}%)')
        print(f'Decreasing trend: {decreasing_corr} ({decreasing_corr/len(df)*100:.1f}%)')
        print(f'No significant trend: {stable_corr} ({stable_corr/len(df)*100:.1f}%)')

        # Show repos that changed significance
        changed = df[df['trend'] != df['trend_corrected']]
        if len(changed) > 0:
            print(f'\nRepos with changed significance after correction:')
            for _, row in changed.iterrows():
                print(f'  {row["repository"]}: {row["trend"]} -> {row["trend_corrected"]} '
                      f'(raw p={row["p_value"]:.4f}, adj p={row["p_adjusted"]:.4f})')

        df.to_csv(self.output_dir / 'repository_gfi_trends.csv', index=False)

        # Save summary (using corrected values)
        summary = pd.DataFrame([{
            'total_repos': len(df),
            'increasing': increasing_corr,
            'decreasing': decreasing_corr,
            'stable': stable_corr,
            'increasing_pct': increasing_corr/len(df)*100,
            'decreasing_pct': decreasing_corr/len(df)*100,
            'stable_pct': stable_corr/len(df)*100,
            'correction_method': 'Benjamini-Hochberg',
            'increasing_uncorrected': increasing,
            'decreasing_uncorrected': decreasing,
            'stable_uncorrected': stable
        }])
        summary.to_csv(self.output_dir / 'repository_heterogeneity.csv', index=False)

        return df

    def run(self):
        print('=' * 60)
        print('RQ1 Analysis: GFI Trends and Newcomer Engagement')
        print('=' * 60)

        self.analyze_monthly_gfi_trend()
        self.analyze_newcomer_gfi_engagement()
        self.analyze_repository_heterogeneity()

        print(f'\n{"=" * 60}')
        print(f'Results saved to: {self.output_dir.resolve()}')


if __name__ == '__main__':
    analyzer = RQ1Analyzer()
    analyzer.run()
