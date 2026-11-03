#!/usr/bin/env python3
"""
Generate a dendrogram visualization of the Agglomerative Hierarchical Clustering (AHC)
used for resource role discovery in the OrdinoR model.

This script:
1. Loads and prepares the event log data
2. Builds a Resource-Activity frequency matrix
3. Computes hierarchical clustering using Ward's method
4. Generates a publication-ready dendrogram

Usage:
    python resources/resource_permissions/visualizations/generate_dendrogram.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[4]))  # process-simulation-engine/
from resources.resource_permissions.data_preparation import ResourceDataPreparation

# Configuration
OUTPUT = Path(__file__).parent / "ahc_dendrogram.png"
N_CLUSTERS = 15  # Number of clusters to highlight (matches OrdinoR model)


def generate_dendrogram():
    """Generate and save the dendrogram visualization."""
    
    print("Loading and preparing event log data...")
    prep = ResourceDataPreparation(log_path="eventlog/eventlog.xes.gz")
    df = prep.prepare(filter_completed=True, exclude_resources=['User_1'], drop_na=True)
    
    print("Building Resource-Activity frequency matrix...")
    # Pivot to create Resource × Activity matrix
    matrix = df.pivot_table(
        index='org:resource', 
        columns='concept:name', 
        aggfunc='size', 
        fill_value=0
    )
    
    print(f"Matrix shape: {matrix.shape[0]} resources × {matrix.shape[1]} activities")
    
    # Compute hierarchical clustering using Ward's method (same as OrdinoR)
    print("Computing hierarchical clustering (Ward's method)...")
    Z = linkage(matrix.values, method='ward')
    
    # Compute cluster assignments for coloring
    cluster_labels = fcluster(Z, t=N_CLUSTERS, criterion='maxclust')
    
    # Create a color palette for clusters
    from matplotlib import cm
    colors = cm.tab20(np.linspace(0, 1, N_CLUSTERS))
    label_colors = {res: colors[cluster_labels[i] - 1] for i, res in enumerate(matrix.index)}
    
    # Create the figure
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Generate dendrogram
    dendro = dendrogram(
        Z,
        labels=matrix.index.tolist(),
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax,
        color_threshold=Z[-(N_CLUSTERS-1), 2] if N_CLUSTERS > 1 else 0,  # Color clusters
        above_threshold_color='#888888'
    )
    
    # Styling
    ax.set_xlabel('Resources', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_ylabel('Ward Distance', fontsize=14, fontweight='bold', labelpad=15)
    ax.set_title(
        f'Agglomerative Hierarchical Clustering (AHC) Dendrogram\n'
        f'{matrix.shape[0]} Resources clustered into {N_CLUSTERS} role groups using Ward\'s method',
        fontsize=16, fontweight='bold', pad=20
    )
    
    # Add horizontal line showing cut threshold
    if N_CLUSTERS > 1:
        cut_height = Z[-(N_CLUSTERS-1), 2]
        ax.axhline(y=cut_height, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.annotate(
            f'Cut for {N_CLUSTERS} clusters',
            xy=(0.02, cut_height),
            xycoords=('axes fraction', 'data'),
            fontsize=11,
            color='red',
            fontweight='bold',
            va='bottom'
        )
    
    # Add cluster count annotation
    ax.annotate(
        f'Based on {len(df):,} events\nacross {matrix.shape[1]} activities',
        xy=(0.98, 0.98),
        xycoords='axes fraction',
        fontsize=10,
        ha='right',
        va='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='gray', alpha=0.9)
    )
    
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Saved dendrogram: {OUTPUT}")
    
    # Print cluster summary
    print(f"\nCluster Summary ({N_CLUSTERS} clusters):")
    print("-" * 50)
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    for cid, size in cluster_sizes.items():
        members = [res for res, lbl in zip(matrix.index, cluster_labels) if lbl == cid]
        print(f"  Cluster {cid:2d}: {size:3d} resources - {', '.join(members[:3])}{'...' if len(members) > 3 else ''}")


if __name__ == "__main__":
    generate_dendrogram()
