import sys
import os
import pandas as pd
import pm4py
from collections import defaultdict

# Add project root to path to ensure imports work
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))  # process-simulation-engine/

from resources.resource_permissions.resource_permissions import BasicResourcePermissions
from resources.BPIC2017.resource_permissions.ordinor import OrdinoRResourcePermissions

def run_comparison():
    # 1. Setup Paths
    # Try to find the event log
    possible_paths = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../eventlog/eventlog_plain.xes')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../eventlog/eventlog.xes')),
        os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../eventlog/eventlog.xes.gz'))
    ]
    
    log_path = None
    for p in possible_paths:
        if os.path.exists(p):
            log_path = p
            break
            
    if not log_path:
        print("Error: Could not find event log.")
        print("Checked locations:", possible_paths)
        return

    print(f"Loading event log from: {log_path}")

    # 2. Instantiate Both Models
    print("Instantiating Basic Model...")
    basic_model = BasicResourcePermissions(log_path=log_path)
    
    print("Instantiating Advanced (OrdinoR) Model...")
    advanced_model = OrdinoRResourcePermissions(log_path=log_path, profiling_mode='full_recall')
    print("Discovering Advanced Model (this may take a moment)...")
    advanced_model.discover_model()

    # 3. Extract Permission Sets
    # We need the set of all unique resources and unique activities
    df = basic_model.df
    all_resources = df['org:resource'].unique()
    all_activities = df['concept:name'].unique()
    
    print(f"Analyzing {len(all_resources)} resources and {len(all_activities)} activities...")

    basic_permissions_count = 0
    advanced_permissions_count = 0
    unlock_count = 0
    
    # Store unlocks per resource to find the "Star Witness"
    # resource -> list of newly unlocked activities
    unlocks_per_resource = defaultdict(list)
    # Store basic permissions per resource for the "previously only could do..." part
    basic_per_resource = defaultdict(list)
    
    total_pairs = len(all_resources) * len(all_activities)
    processed = 0

    print("Computing metrics...")
    
    # Pre-fetch basic permissions (it's a dict lookup, fast)
    # Basic model exposes get_eligible_resources(activity) -> list[resource]
    # Inverting this for faster lookup: (resource, activity) -> bool
    basic_map = set()
    for act in all_activities:
        resources = basic_model.get_eligible_resources(act)
        for res in resources:
            basic_map.add((res, act))
            basic_per_resource[res].append(act)
            
    basic_permissions_count = len(basic_map)

    # Advanced model also has lookup. 
    # For full_recall, get_eligible_resources is efficient enough, or we can invert.
    advanced_map = set()
    for act in all_activities:
        resources = advanced_model.get_eligible_resources(act)
        for res in resources:
            advanced_map.add((res, act))
            
    advanced_permissions_count = len(advanced_map)

    # 4. Compute Delta and specific unlocks
    # Iterate to find the "Unlock" (False in Basic, True in Advanced)
    # Since we have sets of (res, act), we can do set difference
    
    unlocked_pairs = advanced_map - basic_map
    unlock_count = len(unlocked_pairs)
    
    for res, act in unlocked_pairs:
        unlocks_per_resource[res].append(act)

    # Calculate Percentage Increase
    if basic_permissions_count > 0:
        pct_increase = (unlock_count / basic_permissions_count) * 100
    else:
        pct_increase = 0

    # 5. Find "Star Witness"
    # Resource with most new permissions
    star_resource = None
    max_unlocks = -1
    
    for res, unlocked_acts in unlocks_per_resource.items():
        if len(unlocked_acts) > max_unlocks:
            max_unlocks = len(unlocked_acts)
            star_resource = res
            
    # 6. Final Output
    print("\n" + "="*40)
    print("COMPARISON METRICS: BASIC vs ADVANCED")
    print("="*40)
    print(f"Total Permissions (Basic):    {basic_permissions_count}")
    print(f"Total Permissions (Advanced): {advanced_permissions_count}")
    print(f"The 'Unlock' Count:           {unlock_count}")
    print(f"Percentage Increase:          {pct_increase:.1f}%")
    
    print("\n" + "="*40)
    print("STAR WITNESS EXAMPLE")
    print("="*40)
    
    if star_resource:
        prev_acts = sorted(basic_per_resource[star_resource])
        new_acts = sorted(unlocks_per_resource[star_resource])
        
        # Format the output string requested
        # "Resource [ID] could previously only do [X, Y]. With Role Discovery, they are now also permitted to do [Z], effectively doubling their utility."
        
        # Truncate lists if too long for display
        def fmt_acts(act_list):
            if not act_list: return "Nothing"
            if len(act_list) <= 3: return ", ".join(act_list)
            return ", ".join(act_list[:3]) + f" (+{len(act_list)-3} more)"

        prev_str = fmt_acts(prev_acts)
        new_str = fmt_acts(new_acts)
        
        utility_multiplier = (len(prev_acts) + len(new_acts)) / len(prev_acts) if len(prev_acts) > 0 else float('inf')
        utility_str = f"{utility_multiplier:.1f}x"
        
        print(f"Resource {star_resource} could previously only do [{prev_str}].")
        print(f"With Role Discovery, they are now also permitted to do [{new_str}],")
        print(f"effectively increasing their utility by {utility_str}.")
    else:
        print("No resources gained new permissions.")

if __name__ == "__main__":
    run_comparison()
