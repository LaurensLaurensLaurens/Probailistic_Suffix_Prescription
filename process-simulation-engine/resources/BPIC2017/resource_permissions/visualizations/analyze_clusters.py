"""
Cluster Analysis Script for OrdinoR Model

Analyzes the discovered clusters to:
1. Identify top activities per cluster
2. Find unique/distinctive activities
3. Propose business role names
4. Compute cluster statistics

Usage:
    python resources/resource_permissions/analyze_clusters.py
"""
import pickle
import os
import sys
from collections import Counter
from typing import Dict, Set, List, Tuple, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))  # process-simulation-engine/


def load_model(model_path: str) -> Dict[str, Any]:
    """Load the OrdinoR model from pickle file."""
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_clusters(model_data: Dict[str, Any], event_log_path: str = None) -> List[Dict]:
    """
    Analyze each cluster to extract:
    - Size (number of resources)
    - Top activities performed
    - Unique activities
    - Proposed role name
    """
    import pandas as pd
    import pm4py
    
    groups: List[Set[str]] = model_data['groups']
    activity_to_groups: Dict[str, Set[int]] = model_data['activity_to_groups']
    resource_to_group: Dict[str, int] = model_data['resource_to_group']
    
    # Invert activity_to_groups to get group_to_activities
    group_to_activities: Dict[int, Set[str]] = {i: set() for i in range(len(groups))}
    for activity, group_indices in activity_to_groups.items():
        for idx in group_indices:
            group_to_activities[idx].add(activity)
    
    # Load event log to get activity frequencies per resource
    if event_log_path and os.path.exists(event_log_path):
        print(f"Loading event log from {event_log_path}...")
        df = pm4py.read_xes(event_log_path)
        # Filter to completed events only
        if 'lifecycle:transition' in df.columns:
            df = df[df['lifecycle:transition'].str.lower() == 'complete']
    else:
        df = None
        print("Warning: No event log found, using only model data")
    
    results = []
    
    for cluster_id, resources in enumerate(groups):
        cluster_info = {
            'cluster_id': cluster_id,
            'size': len(resources),
            'resources': sorted(resources),
            'activities': sorted(group_to_activities.get(cluster_id, set())),
            'top_activities': [],
            'unique_activities': [],
            'proposed_role': 'Unknown'
        }
        
        # If we have the event log, compute activity frequencies
        if df is not None:
            # Get events for resources in this cluster
            cluster_events = df[df['org:resource'].isin(resources)]
            
            if not cluster_events.empty:
                # Compute activity frequency within this cluster
                activity_counts = cluster_events['concept:name'].value_counts()
                cluster_info['top_activities'] = list(activity_counts.head(5).items())
                
                # Compute total events for size metric
                cluster_info['total_events'] = len(cluster_events)
        else:
            # Fall back to just listing activities
            cluster_info['top_activities'] = [(a, 0) for a in list(cluster_info['activities'])[:5]]
        
        # Find unique activities (only performed by this cluster)
        unique = set()
        for activity in group_to_activities.get(cluster_id, set()):
            # Check if this activity is only in this cluster
            if activity in activity_to_groups:
                performing_groups = activity_to_groups[activity]
                if len(performing_groups) == 1 and cluster_id in performing_groups:
                    unique.add(activity)
        cluster_info['unique_activities'] = sorted(unique)
        
        # Compute homogeneity - how similar are members in their activity patterns
        if df is not None and len(resources) > 1:
            cluster_events = df[df['org:resource'].isin(resources)]
            if not cluster_events.empty:
                # Compute activity set per resource
                resource_activities = cluster_events.groupby('org:resource')['concept:name'].apply(set).to_dict()
                
                # Compute Jaccard similarity between pairs
                from itertools import combinations
                similarities = []
                res_list = list(resource_activities.keys())
                for r1, r2 in combinations(res_list, 2):
                    a1, a2 = resource_activities.get(r1, set()), resource_activities.get(r2, set())
                    if a1 or a2:
                        jaccard = len(a1 & a2) / len(a1 | a2) if (a1 | a2) else 0
                        similarities.append(jaccard)
                
                if similarities:
                    cluster_info['homogeneity'] = sum(similarities) / len(similarities)
                else:
                    cluster_info['homogeneity'] = 1.0  # Single resource or no data
            else:
                cluster_info['homogeneity'] = 0.0
        else:
            cluster_info['homogeneity'] = 1.0
        
        # Propose role name based on activity patterns
        cluster_info['proposed_role'] = propose_role_name(cluster_info)
        
        results.append(cluster_info)
    
    return results


def propose_role_name(cluster_info: Dict) -> str:
    """
    Propose a business role name based on activities performed.
    Uses pattern matching on activity names.
    """
    activities = set(cluster_info['activities'])
    top_acts = [a[0] for a in cluster_info.get('top_activities', [])]
    
    # Pattern matching rules
    role_patterns = [
        # Approval patterns
        (lambda a: any('Approve' in x or 'Accept' in x for x in a), 'Approvers'),
        (lambda a: 'W_Assess potential fraud' in a, 'Fraud Analysts'),
        (lambda a: any('Complete' in x for x in a) and any('Validate' in x for x in a), 'Application Processors'),
        (lambda a: any('O_Create Offer' in x for x in a), 'Offer Creators'),
        (lambda a: any('O_Sent' in x for x in a), 'Offer Senders'),
        (lambda a: any('O_Returned' in x for x in a), 'Offer Return Handlers'),
        (lambda a: any('Call' in x for x in a), 'Customer Service / Callers'),
        (lambda a: any('Handle leads' in x for x in a), 'Lead Handlers'),
        (lambda a: any('Cancelled' in x for x in a) or any('Denied' in x for x in a), 'Denial/Cancellation Handlers'),
        (lambda a: any('collection' in x.lower() for x in a), 'Collections Agents'),
        (lambda a: any('A_Create Application' in x for x in a) or any('A_Submitted' in x for x in a), 'Application Intake'),
        (lambda a: any('Assess' in x for x in a), 'Assessors'),
    ]
    
    for pattern_fn, role_name in role_patterns:
        if pattern_fn(activities) or pattern_fn(top_acts):
            # Refine based on top activities
            top_3_acts = top_acts[:3] if top_acts else list(activities)[:3]
            
            # Check for senior/specialist patterns
            if any('Level_2' in a for a in activities) or any('senior' in a.lower() for a in activities):
                role_name = f"Senior {role_name}"
            
            return role_name
    
    # Fallback: use most common activity type
    if top_acts:
        first_act = top_acts[0]
        if first_act.startswith('W_'):
            return f"Workers ({first_act[2:].split(' ')[0]})"
        elif first_act.startswith('O_'):
            return f"Offer Handlers"
        elif first_act.startswith('A_'):
            return f"Application Handlers"
    
    return f"General Workers (Cluster {cluster_info['cluster_id']})"


def print_markdown_table(results: List[Dict]):
    """Print results as a markdown table."""
    print("\n## Cluster Profiles Analysis\n")
    print("| Cluster ID | Size | Top 3 Activities | Unique Activities | Homogeneity | Proposed Role Name |")
    print("|------------|------|------------------|-------------------|-------------|-------------------|")
    
    for r in sorted(results, key=lambda x: x['size'], reverse=True):
        top_3 = r['top_activities'][:3]
        top_str = ', '.join([f"{a[0]} ({a[1]})" if a[1] else a[0] for a in top_3])
        unique_str = ', '.join(r['unique_activities'][:3]) if r['unique_activities'] else '(none)'
        homogeneity = f"{r.get('homogeneity', 0):.2f}" if 'homogeneity' in r else 'N/A'
        
        print(f"| {r['cluster_id']} | {r['size']} | {top_str} | {unique_str} | {homogeneity} | {r['proposed_role']} |")
    
    print("\n")


def print_detailed_report(results: List[Dict]):
    """Print a detailed report for each cluster."""
    print("\n## Detailed Cluster Breakdown\n")
    
    for r in sorted(results, key=lambda x: x['size'], reverse=True):
        print(f"### Cluster {r['cluster_id']}: {r['proposed_role']}")
        print(f"- **Size**: {r['size']} resources")
        print(f"- **Homogeneity**: {r.get('homogeneity', 'N/A'):.2f}" if isinstance(r.get('homogeneity'), float) else f"- **Homogeneity**: N/A")
        if r.get('total_events'):
            print(f"- **Total Events**: {r['total_events']:,}")
        
        print("\n**Top 5 Activities**:")
        for act, count in r['top_activities'][:5]:
            print(f"  1. {act}" + (f" ({count:,} events)" if count else ""))
        
        if r['unique_activities']:
            print(f"\n**Unique/Near-Unique Activities**: {', '.join(r['unique_activities'])}")
        
        print(f"\n**Members**: {', '.join(sorted(r['resources'])[:10])}" + 
              (f"... (+{len(r['resources'])-10} more)" if len(r['resources']) > 10 else ""))
        print("\n---\n")


if __name__ == "__main__":
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '../../../..'))
    
    # Use the full_recall model by default
    model_path = os.path.join(script_dir, "..", "ordinor_fullrecall.pkl")
    event_log_path = os.path.join(project_root, "eventlog", "eventlog.xes.gz")
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Available models:")
        for f in os.listdir(script_dir):
            if f.endswith('.pkl'):
                print(f"  - {f}")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model_data = load_model(model_path)
    
    print(f"Model type: {model_data.get('profiling_mode', 'unknown')}")
    print(f"Number of groups: {len(model_data.get('groups', []))}")
    
    results = analyze_clusters(model_data, event_log_path)
    
    print_markdown_table(results)
    print_detailed_report(results)
