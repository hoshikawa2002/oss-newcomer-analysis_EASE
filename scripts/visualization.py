"""
GFI Trends Visualization

Creates figures for the paper:
- Figure 1: Monthly GFI ratio trend with moving average
- Figure 2: Repository heterogeneity (existing)
- Figure 3: PR metrics trends (merge rate, description length)
- Figure 4: Task type merge rate comparison
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 10


class GFIVisualizer:
    def __init__(self):
        self.data_dir = Path('../data')
        self.rq1_dir = Path('../results/rq1')
        self.rq2_dir = Path('../results/rq2')
        self.figures_dir = Path('../results/figures')
        self.figures_dir.mkdir(parents=True, exist_ok=True)

    def create_figure1_gfi_trend(self):
        """Figure 1: Monthly GFI ratio trend with moving average."""
        print('\nCreating Figure 1: GFI Ratio Trend...')

        df = pd.read_csv(self.rq1_dir / 'monthly_gfi_trend.csv')
        df['date'] = pd.to_datetime(df['month'] + '-01')

        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot raw data
        ax.plot(df['date'], df['gfi_ratio'], 'o-', color='lightgray',
                markersize=4, linewidth=1, alpha=0.7, label='Monthly')

        # Plot 6-month moving average
        ax.plot(df['date'], df['gfi_ratio_ma6'], '-', color='#1f77b4',
                linewidth=2.5, label='6-month MA')

        # Labels
        ax.set_xlabel('Date')
        ax.set_ylabel('GFI Ratio (%)')

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.xaxis.set_minor_locator(mdates.MonthLocator(interval=3))

        # Legend
        ax.legend(loc='upper right')

        # Y-axis limit
        ax.set_ylim(0, max(df['gfi_ratio']) * 1.1)

        plt.tight_layout()

        # Save
        plt.savefig(self.figures_dir / 'figure1_gfi_trend.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure1_gfi_trend.png', dpi=300, bbox_inches='tight')
        print(f'Saved: {self.figures_dir / "figure1_gfi_trend.pdf"}')
        plt.close()

    def create_figure2_newcomer_engagement(self):
        """Figure 2: Newcomer engagement with GFI issues trend."""
        print('\nCreating Figure 2: Newcomer Engagement Trend...')

        df = pd.read_csv(self.rq1_dir / 'monthly_newcomer_engagement.csv')
        df['date'] = pd.to_datetime(df['month'] + '-01')

        fig, ax = plt.subplots(figsize=(8, 4))

        # Plot raw data
        ax.plot(df['date'], df['engagement_ratio'], 'o-', color='lightgray',
                markersize=4, linewidth=1, alpha=0.7, label='Monthly')

        # Plot 6-month moving average
        ax.plot(df['date'], df['engagement_ratio_ma6'], '-', color='#2ca02c',
                linewidth=2.5, label='6-month MA')

        # Labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Newcomer Engagement Rate (%)')

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        # Legend
        ax.legend(loc='upper left')

        plt.tight_layout()

        # Save
        plt.savefig(self.figures_dir / 'figure2_newcomer_engagement.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure2_newcomer_engagement.png', dpi=300, bbox_inches='tight')
        print(f'Saved: {self.figures_dir / "figure2_newcomer_engagement.pdf"}')
        plt.close()

    def create_figure3_pr_metrics(self):
        """Figure 3: PR metrics trends (merge rate, description length)."""
        print('\nCreating Figure 3: PR Metrics Trends...')

        df = pd.read_csv(self.rq2_dir / 'monthly_pr_metrics.csv')
        df['date'] = pd.to_datetime(df['month'] + '-01')

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Merge Rate
        ax = axes[0]
        ax.plot(df['date'], df['merge_rate'], 'o-', color='lightgray',
                markersize=4, linewidth=1, alpha=0.7, label='Monthly')
        ax.plot(df['date'], df['merge_rate_ma6'], '-', color='#d62728',
                linewidth=2.5, label='6-month MA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Merge Rate (%)')
        ax.legend(loc='upper right')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title('(a) Merge Rate', fontweight='bold')

        # Description Length
        ax = axes[1]
        ax.plot(df['date'], df['body_length'], 'o-', color='lightgray',
                markersize=4, linewidth=1, alpha=0.7, label='Monthly')
        ax.plot(df['date'], df['body_length_ma6'], '-', color='#9467bd',
                linewidth=2.5, label='6-month MA')
        ax.set_xlabel('Date')
        ax.set_ylabel('Description Length (characters)')
        ax.legend(loc='upper left')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_title('(b) Description Length', fontweight='bold')

        plt.tight_layout()

        # Save
        plt.savefig(self.figures_dir / 'figure3_pr_metrics.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure3_pr_metrics.png', dpi=300, bbox_inches='tight')
        print(f'Saved: {self.figures_dir / "figure3_pr_metrics.pdf"}')
        plt.close()

    def create_figure4_task_type_comparison(self):
        """Figure 4: Merge rate comparison by task type."""
        print('\nCreating Figure 4: Task Type Comparison...')

        df = pd.read_csv(self.rq2_dir / 'merge_rate_by_type.csv')
        df = df.sort_values('merge_rate', ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4))

        colors = ['#d62728' if r < 50 else '#2ca02c' for r in df['merge_rate']]

        bars = ax.barh(df['task_type'], df['merge_rate'], color=colors, edgecolor='black')

        # Add value labels
        for bar, count in zip(bars, df['pr_count']):
            width = bar.get_width()
            ax.text(width + 1, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}% (n={count})',
                   va='center', fontsize=10)

        ax.set_xlabel('Merge Rate (%)')
        ax.set_ylabel('Task Type')
        ax.set_xlim(0, 100)
        ax.axvline(x=50, color='gray', linestyle='--', linewidth=1, alpha=0.5)

        plt.tight_layout()

        # Save
        plt.savefig(self.figures_dir / 'figure4_task_type.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure4_task_type.png', dpi=300, bbox_inches='tight')
        print(f'Saved: {self.figures_dir / "figure4_task_type.pdf"}')
        plt.close()

    def create_figure5_task_type_trends(self):
        """Figure 5: Task type merge rate trends over time."""
        print('\nCreating Figure 5: Task Type Trends...')

        df = pd.read_csv(self.rq2_dir / 'monthly_metrics_by_type.csv')
        df['date'] = pd.to_datetime(df['month'] + '-01')

        fig, ax = plt.subplots(figsize=(10, 5))

        colors = {
            'bug': '#1f77b4',
            'documentation': '#ff7f0e',
            'feature': '#2ca02c',
            'cleanup': '#d62728',
            'other': '#9467bd'
        }

        for task_type in ['bug', 'documentation', 'feature', 'other']:
            type_df = df[df['task_type'] == task_type].copy()
            if len(type_df) >= 6:
                type_df = type_df.sort_values('date')
                # Calculate moving average
                type_df['ma6'] = type_df['merge_rate'].rolling(window=6, min_periods=1).mean()
                ax.plot(type_df['date'], type_df['ma6'], '-',
                       color=colors.get(task_type, 'gray'),
                       linewidth=2, label=task_type.capitalize())

        ax.set_xlabel('Date')
        ax.set_ylabel('Merge Rate (%, 6-month MA)')
        ax.legend(loc='upper right', ncol=2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_ylim(0, 100)

        plt.tight_layout()

        # Save
        plt.savefig(self.figures_dir / 'figure5_task_type_trends.pdf', bbox_inches='tight')
        plt.savefig(self.figures_dir / 'figure5_task_type_trends.png', dpi=300, bbox_inches='tight')
        print(f'Saved: {self.figures_dir / "figure5_task_type_trends.pdf"}')
        plt.close()

    def create_repository_heterogeneity(self):
        """Repository heterogeneity visualization (existing figure)."""
        print('\nCreating Repository Heterogeneity Figure...')

        try:
            # Load data
            metadata_df = pd.read_csv(self.data_dir / 'repository_metadata.csv')
            changes_df = pd.read_csv(self.rq1_dir / 'repository_gfi_trends.csv')

            # Merge
            df = changes_df.merge(metadata_df, on='repository', how='left')

            if len(df) == 0:
                print('No data to plot for repository heterogeneity')
                return

            # Categorize by trend
            colors = {
                'increasing': '#2ca02c',
                'decreasing': '#d62728',
                'no trend': '#bcbd22'
            }

            fig, ax = plt.subplots(figsize=(8, 5))

            for trend_type in ['decreasing', 'no trend', 'increasing']:
                data = df[df['trend'] == trend_type]
                if len(data) > 0:
                    ax.scatter(data['age_years'], data['overall_ratio'],
                              s=data['stars'] / 500 if 'stars' in data.columns else 100,
                              c=colors[trend_type], alpha=0.6,
                              edgecolors='black', linewidth=0.5,
                              label=f'{trend_type.capitalize()} (n={len(data)})')

            ax.set_xlabel('Repository Age (years)')
            ax.set_ylabel('Overall GFI Ratio (%)')
            ax.legend(loc='upper right')
            ax.set_ylim(0, None)

            plt.tight_layout()

            plt.savefig(self.figures_dir / 'repository_heterogeneity.pdf', bbox_inches='tight')
            plt.savefig(self.figures_dir / 'repository_heterogeneity.png', dpi=300, bbox_inches='tight')
            print(f'Saved: {self.figures_dir / "repository_heterogeneity.pdf"}')
            plt.close()

        except Exception as e:
            print(f'Could not create repository heterogeneity figure: {e}')

    def run(self):
        """Run complete visualization pipeline."""
        print('=' * 60)
        print('GFI Trends Visualization')
        print('=' * 60)

        self.create_figure1_gfi_trend()
        self.create_figure2_newcomer_engagement()
        self.create_figure3_pr_metrics()
        self.create_figure4_task_type_comparison()
        self.create_figure5_task_type_trends()
        self.create_repository_heterogeneity()

        print('\n' + '=' * 60)
        print('All figures saved!')
        print(f'Output directory: {self.figures_dir.resolve()}')
        print('=' * 60)


if __name__ == '__main__':
    visualizer = GFIVisualizer()
    visualizer.run()
