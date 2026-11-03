#!/usr/bin/env python3
"""Generate zoomed-in cluster heatmap demonstrating permission inheritance."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster

sys.path.insert(0, str(Path(__file__).parents[4]))  # process-simulation-engine/
from resources.resource_permissions.data_preparation import ResourceDataPreparation

OUTPUT = Path(__file__).parent / "cluster_zoom_inheritance.png"
N_CLUSTERS = 15

# Load and prepare data
prep = ResourceDataPreparation(log_path="eventlog/eventlog.xes.gz")
df = prep.prepare(filter_completed=True, exclude_resources=['User_1'], drop_na=True)

# Build Resource-Activity frequency matrix
matrix = df.pivot_table(index='org:resource', columns='concept:name', aggfunc='size', fill_value=0)

# Cluster resources (Ward's method, matching OrdinoR)
labels = fcluster(linkage(matrix.values, method='ward'), t=N_CLUSTERS, criterion='maxclust')
clusters = {}
for res, cid in zip(matrix.index, labels):
    clusters.setdefault(cid, []).append(res)

# Find best cluster for inheritance demo (4-8 resources, high-freq inheritance pattern)
best = max(
    ((cid, res) for cid, res in clusters.items() if 4 <= len(res) <= 8),
    key=lambda x: max(
        (matrix.loc[x[1], act].max() for act in matrix.columns
         if (matrix.loc[x[1], act] > 0).any() and (matrix.loc[x[1], act] == 0).any()),
        default=0
    )
)
cluster_id, resources = best

# Subset to selected cluster and top activities
subset = matrix.loc[resources]
top_acts = subset.sum().nlargest(8).index.tolist()
subset = subset[top_acts]

# Generate heatmap
fig, ax = plt.subplots(figsize=(14, 9))
sns.heatmap(
    np.log1p(subset), annot=subset, fmt='d', cmap='Blues',
    linewidths=1.0, linecolor='white', ax=ax,
    cbar_kws={'label': 'log(freq + 1)', 'shrink': 0.8},
    annot_kws={'size': 12, 'weight': 'bold'}
)
ax.set_xlabel('Activities', fontsize=14, fontweight='bold', labelpad=15)
ax.set_ylabel('Resources', fontsize=14, fontweight='bold', labelpad=15)
plt.setp(ax.get_xticklabels(), rotation=35, ha='right', fontsize=11, fontweight='bold')
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=13, fontweight='bold')
ax.set_title(f'Cluster {cluster_id}: Role-Based Permission Inheritance\n'
             f'Resources with similar behavioral profiles share permissions',
             fontsize=15, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT}")
print(f"Cluster {cluster_id}: {', '.join(resources)}")
