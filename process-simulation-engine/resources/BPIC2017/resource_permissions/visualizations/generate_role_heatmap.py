#!/usr/bin/env python3
"""Generate full Resource-Activity clustered heatmap for Role Discovery visualization."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parents[4]))  # process-simulation-engine/
from resources.resource_permissions.data_preparation import ResourceDataPreparation

OUTPUT = Path(__file__).parent / "resource_role_heatmap.png"

# Load and prepare data
prep = ResourceDataPreparation(log_path="eventlog/eventlog.xes.gz")
df = prep.prepare(filter_completed=True, exclude_resources=['User_1'], drop_na=True)

# Build Resource-Activity frequency matrix
matrix = df.pivot_table(index='org:resource', columns='concept:name', aggfunc='size', fill_value=0)

# Generate clustered heatmap (Ward's method, matching OrdinoR)
g = sns.clustermap(
    np.log1p(matrix), method='ward', metric='euclidean', cmap='Blues',
    figsize=(16, 12), dendrogram_ratio=(0.12, 0.12),
    cbar_pos=(0.02, 0.75, 0.03, 0.18), linewidths=0.1, linecolor='lightgray'
)
g.ax_heatmap.set_xlabel('Activities', fontsize=12, fontweight='bold')
g.ax_heatmap.set_ylabel('Resources', fontsize=12, fontweight='bold')
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9)
plt.setp(g.ax_heatmap.get_yticklabels(), fontsize=7)
g.fig.suptitle('Resource-Activity Matrix with Hierarchical Clustering\n(Role Discovery for Permission Inheritance)',
               fontsize=14, fontweight='bold', y=1.02)
g.ax_cbar.set_ylabel('log(frequency + 1)', fontsize=10)

plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved: {OUTPUT}")
