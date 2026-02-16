"""
Data Science Report: Predicting GFI PR Merge Outcomes & Repository Clustering
"""
import json, os, re, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix, roc_curve)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist

warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# ===== Data Loading =====
DATA_DIR = Path('../data')
OUT_DIR = Path('../results/ds_report')
OUT_DIR.mkdir(parents=True, exist_ok=True)

def get_year(date_str):
    y, m = int(date_str[:4]), int(date_str[5:7])
    if (y == 2021 and m >= 7) or (y == 2022 and m <= 6): return 'Y1'
    elif (y == 2022 and m >= 7) or (y == 2023 and m <= 6): return 'Y2'
    elif (y == 2023 and m >= 7) or (y == 2024 and m <= 6): return 'Y3'
    elif (y == 2024 and m >= 7) or (y == 2025 and m <= 6): return 'Y4'
    return None

def classify_task(labels):
    labels_lower = [l.lower() for l in labels]
    types = set()
    for l in labels_lower:
        if re.search(r'\bbug\b', l) or 'type: bug' in l or 'c-bug' in l:
            types.add('Bug')
        if 'feature' in l or 'enhancement' in l:
            types.add('Feature')
        if 'doc' in l:
            types.add('Docs')
    if len(types) == 0: return 'Other'
    if len(types) == 1: return types.pop()
    return list(types)[0]

# Load PR data
pr_dir = DATA_DIR / 'gfi_prs'
rows = []
for fname in sorted(pr_dir.glob('*_gfi_prs.json')):
    repo = fname.stem.replace('_gfi_prs', '')
    with open(fname) as f:
        prs = json.load(f)
    for pr in prs:
        d = pr['pr_data']
        all_labels = []
        for ref in pr.get('referenced_gfi_issues', []):
            all_labels.extend(ref.get('labels', []))

        year = get_year(d['created_at'])
        if not year:
            continue

        rows.append({
            'repo': repo,
            'pr_number': d['number'],
            'year': year,
            'merged': 1 if d.get('state') == 'MERGED' else 0,
            'insertions': d.get('additions', 0),
            'deletions': d.get('deletions', 0),
            'changed_files': d.get('changed_files', 0),
            'body_length': d.get('body_length', 0),
            'commits': d.get('commits', 1),
            'reviews': d.get('review_comments', 0),
            'task_type': classify_task(all_labels),
            'created_at': d['created_at'],
        })

df = pd.DataFrame(rows)
print(f"Total GFI PRs: {len(df)}")
print(f"Merged: {df['merged'].sum()} ({df['merged'].mean()*100:.1f}%)")
print(f"Repos: {df['repo'].nunique()}")
print()

# ===== Part 1: Merge Prediction with ML =====
print("=" * 60)
print("PART 1: Merge Prediction")
print("=" * 60)

# Feature engineering
df['log_insertions'] = np.log1p(df['insertions'])
df['log_deletions'] = np.log1p(df['deletions'])
df['total_changes'] = df['insertions'] + df['deletions']
df['log_total_changes'] = np.log1p(df['total_changes'])
df['change_ratio'] = df['insertions'] / (df['total_changes'] + 1)
df['year_num'] = df['year'].map({'Y1': 1, 'Y2': 2, 'Y3': 3, 'Y4': 4})

# Encode task type
le_task = LabelEncoder()
df['task_type_enc'] = le_task.fit_transform(df['task_type'])

# Features for prediction
feature_cols = ['log_insertions', 'log_deletions', 'changed_files',
                'body_length', 'commits', 'reviews',
                'log_total_changes', 'change_ratio', 'year_num', 'task_type_enc']

X = df[feature_cols].values
y = df['merged'].values

# Handle any NaN/Inf
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for name, model in models.items():
    y_pred = cross_val_predict(model, X_scaled, y, cv=cv)
    y_proba = cross_val_predict(model, X_scaled, y, cv=cv, method='predict_proba')[:, 1]

    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)

    results[name] = {'accuracy': acc, 'f1': f1, 'auc': auc, 'y_pred': y_pred, 'y_proba': y_proba}
    print(f"{name}: Accuracy={acc:.3f}, F1={f1:.3f}, AUC-ROC={auc:.3f}")

# Save results table
res_df = pd.DataFrame({k: {m: f"{v:.3f}" for m, v in vals.items() if m not in ['y_pred', 'y_proba']}
                        for k, vals in results.items()}).T
res_df.to_csv(OUT_DIR / 'model_comparison.csv')
print(f"\nModel comparison saved.")

# Best model: Gradient Boosting - feature importance
best_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
best_model.fit(X_scaled, y)
importances = best_model.feature_importances_

feat_imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': importances
}).sort_values('importance', ascending=True)

# Figure 1: Feature Importance
fig, ax = plt.subplots(figsize=(6, 4))
feature_labels = {
    'log_insertions': 'Insertions (log)',
    'log_deletions': 'Deletions (log)',
    'changed_files': 'Changed Files',
    'body_length': 'Description Length',
    'commits': 'Commit Count',
    'reviews': 'Review Count',
    'log_total_changes': 'Total Changes (log)',
    'change_ratio': 'Addition Ratio',
    'year_num': 'Analysis Year',
    'task_type_enc': 'Task Type',
}
labels = [feature_labels.get(f, f) for f in feat_imp['feature']]
ax.barh(range(len(labels)), feat_imp['importance'], color='steelblue')
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
ax.set_xlabel('Feature Importance (Gradient Boosting)')
ax.set_title('Feature Importance for Merge Prediction')
plt.tight_layout()
plt.savefig(OUT_DIR / 'feature_importance.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'feature_importance.png', bbox_inches='tight')
plt.close()
print("Feature importance figure saved.")

# Figure 2: ROC Curves
fig, ax = plt.subplots(figsize=(5, 4.5))
for name, vals in results.items():
    fpr, tpr, _ = roc_curve(y, vals['y_proba'])
    ax.plot(fpr, tpr, label=f"{name} (AUC={vals['auc']:.3f})")
ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves for Merge Prediction')
ax.legend(loc='lower right', fontsize=8)
plt.tight_layout()
plt.savefig(OUT_DIR / 'roc_curves.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'roc_curves.png', bbox_inches='tight')
plt.close()
print("ROC curves figure saved.")

# Confusion matrix for best model
best_pred = results['Gradient Boosting']['y_pred']
cm = confusion_matrix(y, best_pred)
print(f"\nConfusion Matrix (Gradient Boosting):")
print(f"  TN={cm[0,0]}, FP={cm[0,1]}")
print(f"  FN={cm[1,0]}, TP={cm[1,1]}")
print(classification_report(y, best_pred, target_names=['Not Merged', 'Merged']))

# ===== Part 2: Repository Clustering =====
print("=" * 60)
print("PART 2: Repository Clustering")
print("=" * 60)

# Load GFI issue data per repo
gfi_dir = DATA_DIR / 'gfi_issues'
all_dir = DATA_DIR / 'all_issue_labels'

repo_features = []
for fname in sorted(gfi_dir.glob('*_gfi_issues.json')):
    repo = fname.stem.replace('_gfi_issues', '')
    with open(fname) as f:
        gfi_issues = json.load(f)

    # GFI counts by year
    gfi_by_year = Counter()
    for issue in gfi_issues:
        yr = get_year(issue['created_at'])
        if yr:
            gfi_by_year[yr] += 1

    # Total issues
    all_fname = all_dir / f"{repo}_all_issues.json"
    if not all_fname.exists():
        continue
    with open(all_fname) as f:
        data = json.load(f)
    all_issues = data.get('issues', data) if isinstance(data, dict) else data
    total_by_year = Counter()
    for issue in all_issues:
        yr = get_year(issue['created_at'])
        if yr:
            total_by_year[yr] += 1

    # PR data for this repo
    repo_prs = df[df['repo'] == repo]

    total_gfi = sum(gfi_by_year.values())
    if total_gfi < 5:
        continue

    # Compute features
    gfi_ratios = {}
    for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
        if total_by_year[yr] > 0:
            gfi_ratios[yr] = gfi_by_year[yr] / total_by_year[yr] * 100
        else:
            gfi_ratios[yr] = 0

    early_ratio = np.mean([gfi_ratios['Y1'], gfi_ratios['Y2']])
    late_ratio = np.mean([gfi_ratios['Y3'], gfi_ratios['Y4']])

    merge_rate = repo_prs['merged'].mean() * 100 if len(repo_prs) > 0 else 0
    avg_reviews = repo_prs['reviews'].mean() if len(repo_prs) > 0 else 0
    avg_body_len = repo_prs['body_length'].mean() if len(repo_prs) > 0 else 0

    repo_features.append({
        'repo': repo,
        'total_gfi': total_gfi,
        'gfi_ratio_y1': gfi_ratios['Y1'],
        'gfi_ratio_y4': gfi_ratios['Y4'],
        'ratio_change': late_ratio - early_ratio,
        'merge_rate': merge_rate,
        'avg_reviews': avg_reviews,
        'avg_body_len': avg_body_len,
        'pr_count': len(repo_prs),
        'early_ratio': early_ratio,
        'late_ratio': late_ratio,
    })

repo_df = pd.DataFrame(repo_features)
print(f"Repos for clustering: {len(repo_df)}")

# Clustering features
cluster_cols = ['total_gfi', 'ratio_change', 'merge_rate', 'avg_reviews', 'pr_count']
X_cluster = repo_df[cluster_cols].values
X_cluster_scaled = StandardScaler().fit_transform(X_cluster)

# Hierarchical clustering
linkage_matrix = linkage(X_cluster_scaled, method='ward')

# Choose k=3 clusters
k = 3
clusters = fcluster(linkage_matrix, k, criterion='maxclust')
repo_df['cluster'] = clusters

# Figure 3: PCA visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_cluster_scaled)
repo_df['pc1'] = X_pca[:, 0]
repo_df['pc2'] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(7, 5))
colors = ['#e74c3c', '#3498db', '#2ecc71']
cluster_names = {1: 'Cluster 1', 2: 'Cluster 2', 3: 'Cluster 3'}
for c in sorted(repo_df['cluster'].unique()):
    mask = repo_df['cluster'] == c
    ax.scatter(repo_df.loc[mask, 'pc1'], repo_df.loc[mask, 'pc2'],
               c=colors[c-1], label=cluster_names[c], s=60, alpha=0.8, edgecolors='white')
    for _, row in repo_df[mask].iterrows():
        short_name = row['repo'].split('_')[-1][:12]
        ax.annotate(short_name, (row['pc1'], row['pc2']), fontsize=6, alpha=0.7,
                    xytext=(3, 3), textcoords='offset points')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
ax.set_title('Repository Clustering by GFI Characteristics (PCA)')
ax.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / 'repo_clustering.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'repo_clustering.png', bbox_inches='tight')
plt.close()
print("Clustering figure saved.")

# Cluster statistics
print("\nCluster Statistics:")
for c in sorted(repo_df['cluster'].unique()):
    subset = repo_df[repo_df['cluster'] == c]
    print(f"\nCluster {c} ({len(subset)} repos):")
    print(f"  Repos: {', '.join(subset['repo'].values[:5])}")
    print(f"  Avg total GFI: {subset['total_gfi'].mean():.0f}")
    print(f"  Avg ratio change: {subset['ratio_change'].mean():.3f}")
    print(f"  Avg merge rate: {subset['merge_rate'].mean():.1f}%")
    print(f"  Avg PR count: {subset['pr_count'].mean():.0f}")

# Figure 4: Cluster comparison
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
for idx, (col, title) in enumerate([
    ('total_gfi', 'Total GFI Issues'),
    ('ratio_change', 'GFI Ratio Change\n(Late - Early)'),
    ('merge_rate', 'Merge Rate (%)')
]):
    data_by_cluster = [repo_df[repo_df['cluster'] == c][col].values for c in sorted(repo_df['cluster'].unique())]
    bp = axes[idx].boxplot(data_by_cluster, labels=[f'C{c}' for c in sorted(repo_df['cluster'].unique())],
                           patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[idx].set_title(title, fontsize=9)
    axes[idx].tick_params(labelsize=8)

plt.suptitle('Cluster Characteristics Comparison', fontsize=11)
plt.tight_layout()
plt.savefig(OUT_DIR / 'cluster_comparison.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'cluster_comparison.png', bbox_inches='tight')
plt.close()
print("Cluster comparison figure saved.")

# Save cluster assignments
repo_df.to_csv(OUT_DIR / 'repo_clusters.csv', index=False)

# ===== Part 3: Temporal Feature Analysis =====
print("\n" + "=" * 60)
print("PART 3: Temporal Analysis of Merge Factors")
print("=" * 60)

# Check if review count's importance changes over time
for yr in ['Y1', 'Y2', 'Y3', 'Y4']:
    subset = df[df['year'] == yr]
    if len(subset) < 30:
        continue
    merged_reviews = subset[subset['merged'] == 1]['reviews'].median()
    unmerged_reviews = subset[subset['merged'] == 0]['reviews'].median()
    mr = subset['merged'].mean() * 100
    print(f"{yr}: n={len(subset)}, merge_rate={mr:.1f}%, "
          f"median_reviews(merged)={merged_reviews:.1f}, median_reviews(unmerged)={unmerged_reviews:.1f}")

# Figure 5: Merge rate by task type over time (heatmap)
pivot = df.groupby(['year', 'task_type'])['merged'].mean().unstack() * 100
fig, ax = plt.subplots(figsize=(6, 3))
sns.heatmap(pivot[['Bug', 'Feature', 'Docs', 'Other']], annot=True, fmt='.1f',
            cmap='RdYlGn', ax=ax, vmin=0, vmax=100, linewidths=0.5)
ax.set_title('Merge Rate (%) by Task Type and Year')
ax.set_ylabel('Analysis Year')
plt.tight_layout()
plt.savefig(OUT_DIR / 'merge_heatmap.pdf', bbox_inches='tight')
plt.savefig(OUT_DIR / 'merge_heatmap.png', bbox_inches='tight')
plt.close()
print("Merge heatmap saved.")

print("\n===== ANALYSIS COMPLETE =====")
print(f"All outputs saved to {OUT_DIR}")
