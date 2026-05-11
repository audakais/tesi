"""
Computes and visualises the overlap of top-50 RF biomarkers across TCGA cohorts.
Produces a binary presence/absence heatmap and a co-occurrence matrix.
"""
import os
import glob
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR  = "/mnt/c/users/utente/desktop/tesi_clean/outputs"
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
TOP_N       = 50

os.makedirs(FIGURES_DIR, exist_ok=True)

files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "BIOMARCATORI_RF_TCGA_*.csv")))
cohort_genes = {}
for fp in files:
    cohort = os.path.basename(fp).replace("BIOMARCATORI_RF_", "").replace(".csv", "")
    genes  = pd.read_csv(fp).head(TOP_N)['Gene'].tolist()
    cohort_genes[cohort] = set(genes)

all_genes = sorted(set.union(*cohort_genes.values()))
cohorts   = sorted(cohort_genes.keys())

# Binary presence/absence matrix
mat = pd.DataFrame(
    {c: [1 if g in cohort_genes[c] else 0 for g in all_genes] for c in cohorts},
    index=all_genes
)
# Keep only genes present in >= 2 cohorts
mat = mat[mat.sum(axis=1) >= 2]

fig, axes = plt.subplots(1, 2, figsize=(18, max(6, len(mat) * 0.25 + 2)))

sns.heatmap(mat, ax=axes[0], cmap='Blues', cbar=False,
            linewidths=0.3, linecolor='grey')
axes[0].set_title('Biomarker presence across cohorts (top-50 RF, shared by ≥2)', fontsize=11)
axes[0].set_xlabel('Cohort'); axes[0].set_ylabel('Gene')
axes[0].tick_params(axis='y', labelsize=7)

# Co-occurrence matrix: how many genes two cohorts share
cooc = pd.DataFrame(index=cohorts, columns=cohorts, dtype=float)
for c1 in cohorts:
    for c2 in cohorts:
        cooc.loc[c1, c2] = len(cohort_genes[c1] & cohort_genes[c2])

sns.heatmap(cooc.astype(float), ax=axes[1], annot=True, fmt='.0f',
            cmap='YlOrRd', linewidths=0.5)
axes[1].set_title('Shared biomarkers between cohorts (top-50 RF)', fontsize=11)

plt.tight_layout()
out = os.path.join(FIGURES_DIR, "biomarker_overlap.png")
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

shared = mat[mat.sum(axis=1) == len(cohorts)]
if not shared.empty:
    print(f"\nGenes in ALL cohorts: {shared.index.tolist()}")
else:
    print("\nNo gene shared by all cohorts.")

csv_out = os.path.join(OUTPUT_DIR, "Biomarker_Overlap.csv")
mat.reset_index().rename(columns={'index': 'Gene'}).to_csv(csv_out, index=False)
print(f"Matrix saved: {csv_out}")

