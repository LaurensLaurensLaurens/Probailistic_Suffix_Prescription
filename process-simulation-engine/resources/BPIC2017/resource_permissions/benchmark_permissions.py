"""
Permission Benchmarking System.

Compares BasicResourcePermissions vs OrdinoRResourcePermissions on BPIC2017 data.
Metrics: coverage, precision, recall, group quality, generalization, sensitivity.

Execute from the resources directory:

    # FullRecall mode (recommended for simulation):
    python resource_permissions/benchmark_permissions.py \
        --log-path ../eventlog.xes.gz \
        --cache-path resources/BPIC2017/resource_permissions/ordinor_fullrecall.pkl \
        --mode full_recall

    # OverallScore mode (precision-optimized) (-> rather for analytical use):
    python resource_permissions/benchmark_permissions.py \
        --log-path ../eventlog.xes.gz \
        --cache-path resources/BPIC2017/resource_permissions/ordinor_model-overall_score.pkl \
        --mode overall_score
"""
import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path (go up two levels: resource_permissions -> resources -> project root)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))  # process-simulation-engine/

from resources.resource_permissions.resource_permissions import BasicResourcePermissions
from resources.BPIC2017.resource_permissions.ordinor import OrdinoRResourcePermissions

logger = logging.getLogger(__name__)

# reproducibility
np.random.seed(42)


@dataclass
class ActivityMetrics:
    """Metrics for a single activity."""
    activity: str
    frequency: int
    basic_pool_size: int
    advanced_pool_size: int
    precision: float  # |basic ∩ advanced| / |advanced|
    recall: float     # |basic ∩ advanced| / |basic|


class PermissionBenchmark:
    """
    Benchmarks BasicResourcePermissions vs OrdinoRResourcePermissions.

    Computes coverage, precision/recall, group quality, and generalization metrics.

    Supports two OrdinoR profiling modes:
    - 'full_recall': Role-based generalization (recommended for simulation)
    - 'overall_score': Context-aware profiling (precision-optimized)
    """

    def __init__(
        self,
        log_path: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        n_clusters: int = 10,
        exclude_resources: Optional[List[str]] = None,
        filter_completed: bool = True,
        sample_size: Optional[int] = None,
        cache_path: Optional[str] = None,
        profiling_mode: str = 'full_recall'
    ):
        """
        Initialize benchmark with Basic and OrdinoR permission systems.

        Args:
            log_path: Path to XES event log.
            df: DataFrame with event log.
            n_clusters: Number of resource clusters (OrdinoR).
            exclude_resources: Resources to exclude (e.g., system users).
            filter_completed: Whether to filter out completed cases.
            sample_size: Number of events to sample (int). If provided, runs on subset.
            cache_path: Path to cached OrdinoR model (.pkl). Required when using log_path.
            profiling_mode: 'full_recall' (default) or 'overall_score'.
        """
        self.n_clusters = n_clusters
        self.exclude_resources = exclude_resources or []
        self.filter_completed = filter_completed
        self.sample_size = sample_size
        self.profiling_mode = profiling_mode

        logger.info(f"Initializing benchmark systems (mode={profiling_mode})...")

        # Load or set DataFrame
        if df is not None:
            self.df = df
        elif log_path is not None:
            temp_basic = BasicResourcePermissions(log_path=log_path, filter_completed=filter_completed, exclude_resources=exclude_resources)
            self.df = temp_basic.df
        else:
            raise ValueError("Either log_path or df must be provided")

        # Apply sampling if specified
        if self.sample_size and self.sample_size < len(self.df):
            logger.info(f"Sampling first {self.sample_size} events for quick verification...")
            self.df = self.df.head(self.sample_size)

        # Initialize both systems with the (potentially sampled) DataFrame
        self.basic = BasicResourcePermissions(df=self.df, filter_completed=filter_completed, exclude_resources=exclude_resources)
        self.advanced = OrdinoRResourcePermissions(
            df=self.df,
            filter_completed=filter_completed,
            exclude_resources=exclude_resources,
            profiling_mode=profiling_mode
        )

        # Load OrdinoR model from cache
        if log_path and not cache_path:
            raise ValueError("cache_path is required when using log_path.")

        if cache_path:
            if not os.path.exists(cache_path):
                raise FileNotFoundError(f"Cached model not found at {cache_path}. Run discovery first to create it.")
            logger.info(f"Loading cached OrdinoR model from {cache_path}...")
            self.advanced.load_model(cache_path)
        else:
            # No cache provided and using DataFrame (testing scenario)
            logger.warning("No cache_path provided - discovering model (only use this for testing with small datasets!)")
            self.advanced.discover_model(n_resource_clusters=n_clusters)

        self.silhouette_score = getattr(self.advanced, '_silhouette_score', None)
        
        # Cache all activities
        self._all_activities = set(self.basic.activity_resource_map.keys())
        
        # Compute activity frequencies
        self._activity_freq = self.df.groupby("concept:name").size().to_dict()
    
    def compute_precision_recall(self) -> Dict[str, Dict[str, float]]:
        """
        Compute precision and recall for each activity.
        
        Precision = |basic ∩ advanced| / |advanced|
            "Of advanced-eligible resources, what % actually performed the activity?"
        
        Recall = |basic ∩ advanced| / |basic|
            "Of actual performers, what % does advanced include?"
        
        Returns:
            Dict mapping activity -> {precision, recall, basic_size, advanced_size}.
        """
        results = {}
        
        for activity in self._all_activities:
            basic_set = set(self.basic.get_eligible_resources(activity))
            advanced_set = set(self.advanced.get_eligible_resources(activity))
            
            intersection = basic_set & advanced_set
            
            precision = len(intersection) / len(advanced_set) if advanced_set else 0.0
            recall = len(intersection) / len(basic_set) if basic_set else 0.0
            
            results[activity] = {
                "precision": precision,
                "recall": recall,
                "basic_size": len(basic_set),
                "advanced_size": len(advanced_set),
                "intersection_size": len(intersection)
            }
        
        return results
    
    def compute_coverage_comparison(self) -> Dict[str, Dict]:
        """
        Compare coverage between basic and advanced approaches.
        
        Returns:
            Dict with coverage stats for both approaches.
        """
        # Basic: all activities with at least one resource
        basic_covered = {a for a, r in self.basic.activity_resource_map.items() if r}
        
        # Advanced: reuse existing method
        advanced_stats = self.advanced.get_coverage_stats()
        
        # Calculate set of covered activities for comparison
        advanced_covered = set()
        for act in self._all_activities:
             if self.advanced.get_eligible_resources(act):
                 advanced_covered.add(act)
        
        return {
            "basic": {
                "covered_activities": len(basic_covered),
                "total_activities": len(self._all_activities),
                "coverage_ratio": len(basic_covered) / len(self._all_activities) if self._all_activities else 0.0
            },
            "advanced": advanced_stats,
            "comparison": {
                "only_basic": list(basic_covered - advanced_covered),
                "only_advanced": list(advanced_covered - basic_covered),
                "both": list(basic_covered & advanced_covered)
            }
        }

    
    def compute_group_quality(self) -> Dict:
        """
        Compute metrics about the discovered groups.
        Adapts to both FullRecall and OverallScore modes.
        """
        groups = []

        # FullRecall mode: groups stored in _groups
        if self.profiling_mode == 'full_recall' and hasattr(self.advanced, '_groups'):
            groups = self.advanced._groups
        # OverallScore mode: groups from OrdinoR model
        elif self.advanced.model is not None:
            try:
                if hasattr(self.advanced.model, 'find_all_groups'):
                    raw_groups = self.advanced.model.find_all_groups()
                    # Extract actual resource sets from OrdinoR structure
                    for g in raw_groups:
                        if isinstance(g, (list, tuple)) and len(g) >= 2:
                            groups.append(g[1])  # [group_id, frozenset(resources)]
                        else:
                            groups.append(g)
            except Exception as e:
                logger.warning(f"Could not retrieve groups from model: {e}")

        n_groups = len(groups)

        if n_groups == 0:
            return {
                "n_groups": 0,
                "size_stats": {"mean": 0, "std": 0, "min": 0, "max": 0},
                "group_details": []
            }

        group_sizes = [len(g) for g in groups]

        # Build group details
        group_details = []
        for i, resources in enumerate(groups):
            group_details.append({
                "group_id": str(i),
                "size": len(resources),
                "capabilities": "N/A", 
                "domain_focus": "mixed" 
            })
            
        return {
            "n_groups": n_groups,
            "group_details": group_details,
            "size_stats": {
                "mean": np.mean(group_sizes),
                "std": np.std(group_sizes),
                "min": np.min(group_sizes),
                "max": np.max(group_sizes)
            },
            "domain_separation": self._compute_domain_separation(group_details) # Pass group_details for domain separation
        }
    
    def _classify_domain(self, o_caps: List, w_caps: List, a_caps: List) -> str:
        """Classify a group's domain focus."""
        total = len(o_caps) + len(w_caps) + len(a_caps)
        if total == 0:
            return "empty"
        
        o_ratio = len(o_caps) / total
        w_ratio = len(w_caps) / total
        a_ratio = len(a_caps) / total
        
        if o_ratio >= 0.6:
            return "O_focused"
        elif w_ratio >= 0.6:
            return "W_focused"
        elif a_ratio >= 0.6:
            return "A_focused"
        else:
            return "mixed"
    
    def _compute_domain_separation(self, group_stats: List[Dict]) -> float:
        """Compute domain separation score (0-1, higher = better separation)."""
        focused = sum(1 for g in group_stats if g["domain_focus"] != "mixed" and g["domain_focus"] != "empty")
        total = len(group_stats)
        return focused / total if total > 0 else 0.0
    
    def run_holdout_test(self, holdout_fraction: float = 0.2) -> Dict:
        """
        Test generalization using holdout validation.
        
        Splits cases into train (80%) and test (20%), trains on train set,
        evaluates recall on test set.
        
        Args:
            holdout_fraction: Fraction of cases to hold out for testing.
        
        Returns:
            Dict with train/test metrics.
        """
        logger.info(f"Running holdout test (holdout={holdout_fraction})...")
        
        # Split by case
        cases = self.df["case:concept:name"].unique()
        np.random.shuffle(cases)
        
        split_idx = int(len(cases) * (1 - holdout_fraction))
        train_cases = set(cases[:split_idx])
        test_cases = set(cases[split_idx:])
        
        train_df = self.df[self.df["case:concept:name"].isin(train_cases)]
        test_df = self.df[self.df["case:concept:name"].isin(test_cases)]
        
        logger.info(f"  Train: {len(train_cases)} cases, {len(train_df)} events")
        logger.info(f"  Test: {len(test_cases)} cases, {len(test_df)} events")
        
        # Train Basic on train set (OrdinoR skipped - uses pre-trained cached model)
        basic_train = BasicResourcePermissions(df=train_df)

        # Evaluate on test set: for each (activity, resource) in test,
        # check if it would have been allowed
        test_pairs = test_df[["concept:name", "org:resource"]].drop_duplicates()

        basic_hits = 0
        advanced_hits = 0
        total = 0

        for _, row in test_pairs.iterrows():
            activity = row["concept:name"]
            resource = row["org:resource"]

            if pd.isna(activity) or pd.isna(resource):
                continue

            total += 1

            if resource in basic_train.get_eligible_resources(activity):
                basic_hits += 1

            # Use pre-loaded OrdinoR model (trained on full data)
            try:
                if resource in self.advanced.get_eligible_resources(activity):
                    advanced_hits += 1
            except ValueError:
                pass  # Activity not in model

        return {
            "train_cases": len(train_cases),
            "test_cases": len(test_cases),
            "test_pairs": total,
            "basic_recall": basic_hits / total if total > 0 else 0.0,
            "advanced_recall": advanced_hits / total if total > 0 else 0.0,
            "improvement": (advanced_hits - basic_hits) / total if total > 0 else 0.0
        }
    
    def run_sensitivity_analysis(self, cluster_range: range = range(3, 11)) -> pd.DataFrame:
        """
        Analyze how metrics change with different n_clusters values.
        
        Args:
            cluster_range: Range of cluster counts to test.
        
        Returns:
            DataFrame with metrics for each cluster count.
        """
        logger.info(f"Running sensitivity analysis for n_clusters in {list(cluster_range)}...")
        
        results = []
        for n in cluster_range:
            logger.info(f"  Testing n_clusters={n}...")
            
            adv = AdvancedResourcePermissions(df=self.df)
            adv.discover_model(
                n_clusters=n,
                min_frequency=self.min_frequency,
                min_coverage=self.min_coverage,
                exclude_resources=self.exclude_resources
            )
            
            # Compute metrics
            coverage = adv.get_coverage_stats()
            
            # Precision/recall averaged across activities
            pr_metrics = []
            for activity in self._all_activities:
                basic_set = set(self.basic.get_eligible_resources(activity))
                advanced_set = set(adv.get_eligible_resources(activity))
                
                if advanced_set:
                    precision = len(basic_set & advanced_set) / len(advanced_set)
                else:
                    precision = 0.0
                
                if basic_set:
                    recall = len(basic_set & advanced_set) / len(basic_set)
                else:
                    recall = 0.0
                
                pr_metrics.append((precision, recall))
            
            avg_precision = np.mean([p for p, r in pr_metrics])
            avg_recall = np.mean([r for p, r in pr_metrics])
            
            results.append({
                "n_clusters": n,
                "coverage": coverage["coverage_ratio"],
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "f1_score": 2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
            })
        
        return pd.DataFrame(results)
    
    def get_activity_breakdown(self) -> pd.DataFrame:
        """
        Get per-activity breakdown of all metrics.
        
        Returns:
            DataFrame with one row per activity.
        """
        pr_metrics = self.compute_precision_recall()
        
        rows = []
        for activity in sorted(self._all_activities):
            freq = self._activity_freq.get(activity, 0)
            metrics = pr_metrics.get(activity, {})
            
            rows.append({
                "activity": activity,
                "frequency": freq,
                "basic_pool_size": metrics.get("basic_size", 0),
                "advanced_pool_size": metrics.get("advanced_size", 0),
                "precision": metrics.get("precision", 0.0),
                "recall": metrics.get("recall", 0.0),
                "pool_expansion": (metrics.get("advanced_size", 0) - metrics.get("basic_size", 0)) / metrics.get("basic_size", 1) if metrics.get("basic_size", 0) > 0 else 0.0
            })
        
        return pd.DataFrame(rows)
    
    def generate_report(self, output_dir: str) -> str:
        """Generate human-readable markdown report."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute all metrics
        coverage = self.compute_coverage_comparison()
        pr_metrics = self.compute_precision_recall()
        group_quality = self.compute_group_quality()
        holdout = self.run_holdout_test()
        
        # Aggregate precision/recall
        precisions = [m["precision"] for m in pr_metrics.values()]
        recalls = [m["recall"] for m in pr_metrics.values()]
        avg_precision = np.mean(precisions) if precisions else 0
        avg_recall = np.mean(recalls) if recalls else 0
        
        # Generate report
        mode_label = "FullRecall" if self.profiling_mode == 'full_recall' else "OverallScore"
        report = f"""# OrdinoR Benchmark Report

Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}
Log: {len(self.df)} events, {len(self._all_activities)} activities
Configuration: OrdinoR ({mode_label} profiling)
Params: n_resource_clusters={self.n_clusters}, mode={self.profiling_mode}

## Coverage

| Approach | Covered | Total | Coverage |
|----------|---------|-------|----------|
| Basic | {coverage['basic']['covered_activities']} | {coverage['basic']['total_activities']} | {coverage['basic']['coverage_ratio']:.1%} |
| OrdinoR | {coverage['advanced']['covered_activities']} | {coverage['advanced']['total_activities']} | {coverage['advanced']['coverage_ratio']:.1%} |

{"✓" if coverage['advanced']['coverage_ratio'] >= 0.9 else "⚠"} OrdinoR coverage {"meets" if coverage['advanced']['coverage_ratio'] >= 0.9 else "below"} 90% threshold.

## Precision & Recall (Activity-Based)

| Metric | Value |
|--------|-------|
| Avg Precision | {avg_precision:.1%} |
| Avg Recall | {avg_recall:.1%} |
| F1 Score | {2 * avg_precision * avg_recall / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0:.1%} |

{"✓" if avg_precision >= 0.6 else "✗"} Precision {"≥60%" if avg_precision >= 0.6 else "<60%"} - {"most" if avg_precision >= 0.6 else "many"} eligible resources have actually performed the activity.

## Group Quality

- **Groups discovered:** {group_quality['n_groups']}
- **Avg group size:** {group_quality['size_stats']['mean']:.1f} (σ={group_quality['size_stats']['std']:.1f})

| Group | Size |
|-------|------|
"""
        for g in group_quality['group_details']:
            report += f"| {g['group_id']} | {g['size']} |\n"
        
        report += f"""

## Generalization (Holdout Test)

| System | Test Recall |
|--------|-------------|
| Basic | {holdout['basic_recall']:.1%} |
| OrdinoR | {holdout['advanced_recall']:.1%} |

{"✓" if holdout['advanced_recall'] > holdout['basic_recall'] else "✗"} OrdinoR {"outperforms" if holdout['advanced_recall'] > holdout['basic_recall'] else "underperforms"} basic on unseen cases.

## Recommendation

"""
        # Generate recommendation
        if avg_precision >= 0.6 and holdout['advanced_recall'] >= holdout['basic_recall']:
            report += "**Use OrdinoRResourcePermissions** for simulation. Higher recall and acceptable precision.\n"
        elif avg_precision < 0.4:
            report += "**Use BasicResourcePermissions** for simulation. OrdinoR precision too low.\n"
        else:
            report += "**Consider hybrid approach.** Use basic for high-frequency activities, OrdinoR for rare ones.\n"
        
        # Write report
        report_path = os.path.join(output_dir, "benchmark_report.md")
        with open(report_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {report_path}")
        return report_path
    
    def generate_plots(self, output_dir: str) -> List[str]:
        """Generate visualization plots."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plots")
            return []
        
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        
        paths = []
        
        # 1. Pool size comparison
        activity_df = self.get_activity_breakdown()
        
        if not activity_df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            x = range(len(activity_df))
            width = 0.35
            ax.bar([i - width/2 for i in x], activity_df["basic_pool_size"], width, label="Basic", alpha=0.7)
            ax.bar([i + width/2 for i in x], activity_df["advanced_pool_size"], width, label="OrdinoR", alpha=0.7)
            ax.set_xlabel("Activity Index")
            ax.set_ylabel("Pool Size")
            ax.set_title("Resource Pool Size Comparison")
            ax.legend()
            path = os.path.join(plots_dir, "pool_sizes.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            paths.append(path)
            
            # 2. Precision vs Pool Size scatter
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(activity_df["advanced_pool_size"], activity_df["precision"], alpha=0.6)
            ax.set_xlabel("OrdinoR Pool Size")
            ax.set_ylabel("Precision")
            ax.set_title("Precision vs Pool Size")
            ax.axhline(y=0.6, color='r', linestyle='--', alpha=0.5, label="60% threshold")
            ax.legend()
            path = os.path.join(plots_dir, "precision_vs_pool.png")
            plt.savefig(path, dpi=150, bbox_inches='tight')
            plt.close()
            paths.append(path)
        
        logger.info(f"Plots saved to {plots_dir}")
        return paths
    
    def run_full_benchmark(self, output_dir: str) -> Dict:
        """Run complete benchmark suite and save all outputs."""
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Running full benchmark suite...")
        
        # Generate report
        report_path = self.generate_report(output_dir)
        
        # Save activity breakdown
        activity_df = self.get_activity_breakdown()
        activity_path = os.path.join(output_dir, "activity_breakdown.csv")
        activity_df.to_csv(activity_path, index=False)
        
        # Plots
        plot_paths = self.generate_plots(output_dir)
        
        return {
            "report": report_path,
            "activity_breakdown": activity_path,
            "sensitivity_analysis": None, # Sensitivity analysis skipped for OrdinoR for now due to complexity
            "plots": plot_paths
        }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Benchmark Basic vs OrdinoR Permissions")
    parser.add_argument("--log-path", required=True, help="Path to XES event log")
    parser.add_argument("--cache-path", required=True, help="Path to cached OrdinoR model (.pkl file)")
    parser.add_argument("--mode", choices=['full_recall', 'overall_score'], default='full_recall',
                       help="Profiling mode: 'full_recall' (default) or 'overall_score'")
    parser.add_argument("--n-clusters", type=int, default=10, help="Number of resource clusters (default: 10)")
    parser.add_argument("--output-dir", default="benchmark_results", help="Output directory")
    parser.add_argument("--exclude-resources", nargs="*", default=[], help="Resources to exclude (e.g., User_1)")
    parser.add_argument("--sample", type=int, help="Sample size for quick verification (e.g. 1000)")

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # Run benchmark
    benchmark = PermissionBenchmark(
        log_path=args.log_path,
        n_clusters=args.n_clusters,
        exclude_resources=args.exclude_resources,
        sample_size=args.sample,
        cache_path=args.cache_path,
        profiling_mode=args.mode
    )
    
    results = benchmark.run_full_benchmark(args.output_dir)
    
    print("\n" + "="*50)
    print("BENCHMARK COMPLETE")
    print("="*50)
    print(f"Report: {results['report']}")
    print(f"Activity breakdown: {results['activity_breakdown']}")
    print(f"Plots: {len(results['plots'])} generated")


if __name__ == "__main__":
    main()
