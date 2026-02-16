# Examining Changes in Good First Issue Practice and Newcomer Pull Requests since the Release of ChatGPT

This package contains data and analysis scripts to reproduce the results reported in the paper.

## Package Structure

```
replication_package/
├── data/                          # Data files
│   ├── gfi_issues/               # GFI-labeled issues (30 repositories)
│   ├── gfi_prs/                  # PRs addressing GFI issues
│   ├── all_issue_labels/         # All issues (for ratio calculation)
│   ├── repositories.csv          # List of 37 repositories
│   └── repository_metadata.csv   # Repository metadata
├── scripts/                       # Analysis scripts
│   ├── rq1_analysis.py           # RQ1: GFI trends analysis
│   ├── rq2_analysis.py           # RQ2: PR characteristics (placeholder)
│   └── visualization.py          # Generate Figure 2
└── results/                       # Output directory
    ├── rq1/                      # RQ1 results
    └── rq2/                      # RQ2 results
```

## Requirements

- Python 3.8+
- Required packages: `pandas`, `numpy`, `matplotlib`, `scipy`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Analysis

### RQ1 Analysis
```bash
cd scripts
python rq1_analysis.py
```

### RQ2 Analysis
```bash
cd scripts
python rq2_analysis.py
```

# oss-newcomer-analysis_EASE
