"""
Learning curves for all TCGA cohorts (focus: PRAD and STAD).
Shows that low Recall Normal is due to data scarcity, not model failure.
"""
import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, learning_curve
from sklearn.metrics import make_scorer, f1_score

DATASET_DIR = "/mnt/c/users/utente/desktop/tesi_clean/python/ml_dataset_project_batches"
OUTPUT_DIR  = "/mnt/c/users/utente/desktop/tesi_clean/outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
META_COLS   = ['sample_id', 'sample_uuid', 'submitter_id', 'cancer_project',
               'sample_type', 'target']

os.makedirs(FIGURES_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(DATASET_DIR, "dataset_TCGA_*.csv")))
files = [f for f in files if '_vs_gtex' not in f]

model  = RandomForestClassifier(n_estimators=100, max_depth=7,
                                 class_weight='balanced_subsample',
                                 random_state=42, n_jobs=-1)
scorer = make_scorer(f1_score, average='macro')

fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()

for i, fp in enumerate(files):
    cohort = os.path.basename(fp).replace("dataset_", "").replace(".csv", "")
    df     = pd.read_csv(fp)
    if df['target'].nunique() < 2:
        continue

    groups = df['submitter_id'].astype(str).str[:12]
    y      = df['target'].astype(int)
    X      = df.drop(columns=[c for c in META_COLS if c in df.columns], errors='ignore')
    X      = np.log2(X.apply(pd.to_numeric, errors='coerce').fillna(0) + 1.0)

    gkf    = GroupKFold(n_splits=5)
    sizes  = np.linspace(0.2, 1.0, 6)

    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, groups=groups, cv=gkf,
        train_sizes=sizes, scoring=scorer, n_jobs=1
    )

    ax = axes[i]
    ax.plot(train_sizes, train_scores.mean(axis=1), 'o-', color='steelblue', label='Train')
    ax.fill_between(train_sizes,
                    train_scores.mean(1) - train_scores.std(1),
                    train_scores.mean(1) + train_scores.std(1), alpha=0.2, color='steelblue')
    ax.plot(train_sizes, val_scores.mean(axis=1), 'o-', color='tomato', label='Validation')
    ax.fill_between(train_sizes,
                    val_scores.mean(1) - val_scores.std(1),
                    val_scores.mean(1) + val_scores.std(1), alpha=0.2, color='tomato')
    ax.set_title(cohort, fontsize=10)
    ax.set_xlabel('Training samples'); ax.set_ylabel('F1 Macro')
    ax.set_ylim(0.5, 1.02); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    print(f"  {cohort} done")

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Learning Curves — TCGA cohorts (5-fold GroupKFold)', fontsize=13)
plt.tight_layout()
out = os.path.join(FIGURES_DIR, "learning_curves.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

