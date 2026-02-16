# GFI (Good First Issue) 時系列トレンド分析

## 研究概要

**研究タイトル**: Examining Changes in Good First Issue Practices and Newcomer Pull Request Characteristics in Popular OSS Projects

**分析期間**: 2021年7月 〜 2025年6月（4年間）

**対象**: GitHub上の人気OSSプロジェクト37件（うちGFIラベル使用30件）

---

## データセット

| 項目 | 件数 |
|------|------|
| 総Issue数 | 406,826 |
| GFIラベル付きIssue | 3,300 |
| 新規貢献者PR（全体） | 43,906 |
| GFI対応PR | 1,117 |

---

## リサーチクエスチョン

### RQ1: GFI慣行と新規貢献者エンゲージメントは4年間でどのように変化したか？

### RQ2: GFI PRの特性はどのように変化し、マージ成功と関連する要因は何か？

---

## 分析年度の定義

| 年度 | 期間 |
|------|------|
| Y1 | 2021年7月 〜 2022年6月 |
| Y2 | 2022年7月 〜 2023年6月 |
| Y3 | 2023年7月 〜 2024年6月 |
| Y4 | 2024年7月 〜 2025年6月 |

---

## 主要な発見

### RQ1: GFI慣行の変化

#### GFI比率の減少傾向
- **Kendall's τ = -0.44, p < 0.001**（統計的に有意な減少）
- GFI比率: 0.53%（開始時）→ 0.48%（終了時）

#### リポジトリ間の異質性
| トレンド | リポジトリ数 | 割合 |
|----------|-------------|------|
| 減少傾向 | 10 | 33.3% |
| 傾向なし | 17 | 56.7% |
| 増加傾向 | 3 | 10.0% |

- リポジトリ年齢、主要プログラミング言語との相関なし
- **プロジェクト固有の戦略的意思決定に依存**

#### 新規貢献者エンゲージメント
- **約27%で4年間安定**（τ = 0.06, p = 0.52、有意な傾向なし）
- GFIラベルは引き続き有効なオンボーディングシグナルとして機能

---

### RQ2: GFI PRの特性変化

#### 全体のマージ率
- **53.0%**（1,117件中592件がマージ）
- 減少傾向あり（τ = -0.31, p < 0.001）

#### タスクタイプ別マージ率（年度別）

| タスクタイプ | Y1 | Y2 | Y3 | Y4 | 総合 | トレンド |
|-------------|------|------|------|------|------|---------|
| Bug | 64.5% | 83.5% | 71.9% | 45.9% | 68.7% | 減少** |
| Feature | 53.7% | 54.8% | 53.3% | 55.6% | 54.4% | なし |
| Documentation | 68.4% | 65.2% | 42.4% | 47.7% | 52.9% | 減少* |
| Other | 57.1% | 46.4% | 33.3% | 28.6% | 40.7% | 減少* |

**注**: *p<0.05, **p<0.01, ***p<0.001 (Mann-Kendall検定)

#### タスクタイプ分類ロジック
```python
def classify_task_type(labels):
    labels_lower = [l.lower() for l in labels]
    has_bug = any('bug' in l for l in labels_lower)
    has_feature = any('feature' in l or 'enhancement' in l for l in labels_lower)
    has_doc = any('doc' in l for l in labels_lower)

    if has_bug:
        return 'Bug'
    elif has_feature:
        return 'Feature'
    elif has_doc:
        return 'Documentation'
    else:
        return 'Other'
```

---

## PyTorchの特殊性

### 問題
- PyTorchは**188件のGFI PRで0%マージ率**
- モジュールベースのラベリングシステムを使用（`module: dynamo`, `module: inductor`等）
- タスクタイプラベル（bug, feature, doc）をほとんど使用しない

### Otherカテゴリへの影響

| 年度 | PyTorchの占有率 | Other全体マージ率 | PyTorch除外時 |
|------|----------------|------------------|--------------|
| Y1 | 11.2% (11/98) | 57.1% | 64.4% |
| Y2 | 29.7% (41/138) | 46.4% | 66.0% |
| Y3 | 37.5% (45/120) | 33.3% | 53.3% |
| Y4 | 40.5% (51/126) | 28.6% | 48.0% |

**結論**: Otherカテゴリのマージ率低下はPyTorchの影響が大きいが、除外しても依然として減少傾向は見られる

---

## マージ成功要因

| メトリクス | マージされたPR | マージされなかったPR | p値 |
|-----------|--------------|-------------------|-----|
| Insertions (log) | 3.02 | 2.89 | 0.885 |
| Deletions (log) | 1.10 | 1.39 | 0.994 |
| Changed Files | 2.0 | 2.0 | 0.155 |
| Commits Count | 3.0 | 2.0 | <0.001*** |
| **Review Count** | **2.0** | **1.0** | **<0.001***|
| Description Length | 382.5 | 435.0 | 0.608 |

**主要発見**: レビューカウントのみがマージ成功と有意に関連。コードの量や説明文長は関連なし。

---

## 実践的示唆

### メンテナー向け
- GFIラベルは依然として有効（エンゲージメント率27%で安定）
- Bug-fixタスクのマージ率が最も高い（68.7%）→ 明確なスコープのバグ修正を優先的にGFIとしてラベル付け
- レビュー対話が成功の鍵

### 新規貢献者向け
- GFI比率は減少傾向 → 複数プロジェクトを横断的に探索
- Bug-fixタスクを選択すると成功確率が高い
- レビュアーとの対話に積極的に参加

### 研究者向け
- プロジェクト異質性を考慮することが重要
- ラベリング慣行はプロジェクトによって大きく異なる（タスクタイプ vs モジュールベース）
- 集約統計だけでは多様な戦略を見落とす可能性

---

## ファイル構成

```
oss-newcomer-analysis/
├── data/
│   ├── gfi_prs/           # GFI PR データ (JSON)
│   ├── newcomer_prs/      # 新規貢献者PR データ (JSON)
│   └── issues/            # Issue データ (JSON)
├── results/
│   ├── rq1/
│   │   ├── monthly_gfi_trend.csv
│   │   ├── repository_gfi_trends.csv
│   │   └── monthly_newcomer_engagement.csv
│   └── rq2/
│       ├── merge_rate_by_type.csv
│       ├── type_trend_analysis.csv
│       ├── monthly_metrics_by_type.csv
│       └── merge_factors.csv
├── paper/
│   └── conference_101719.tex  # 論文本体
└── scripts/
    ├── rq1_analysis.py
    └── rq2_analysis.py
```

---

## 統計手法

### Mann-Kendall トレンド検定
- 時系列データの単調増加/減少傾向を検定
- ノンパラメトリック手法（正規性の仮定不要）

```python
from scipy.stats import kendalltau
tau, p_value = kendalltau(np.arange(len(values)), values)
```

### Mann-Whitney U検定
- 2群間の中央値の差を検定
- マージ/非マージPRの比較に使用

---

## 今後の課題

1. **質的調査**: なぜ一部のプロジェクトがGFI使用を増やし、他は減らしているのか
2. **マージ率低下の根本原因**: Issue品質、メンテナンスリソース、品質基準の変化
3. **タスクタイプ別支援戦略**: Bug vs Documentation vs Feature それぞれに最適なオンボーディング方法

---

## 参考文献（主要）

- Tan et al. (2020): GFIラベルの利用実態分析
- Cao et al. (2023): メンタリングの重要性
- Turzo et al. (2025): オンボーディング推奨事項の有効性評価
- Steinmacher et al. (2015): 新規貢献者の障壁分類
