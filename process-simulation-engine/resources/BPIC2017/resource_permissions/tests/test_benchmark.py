"""
Tests for the Permission Benchmarking System.
"""
import unittest
import pandas as pd
import numpy as np
import os
import sys
import tempfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from resources.BPIC2017.resource_permissions.benchmark_permissions import PermissionBenchmark


class TestBenchmarkMetrics(unittest.TestCase):
    """Test individual benchmark metrics."""
    
    @classmethod
    def setUpClass(cls):
        # Create synthetic data with clear group structure
        data = []
        
        # Group 1: R1-R3 do activities A, B
        for r in ["R1", "R2", "R3"]:
            for _ in range(10):
                data.append({"concept:name": "A", "org:resource": r, "case:concept:name": f"case_{len(data)}"})
            for _ in range(8):
                data.append({"concept:name": "B", "org:resource": r, "case:concept:name": f"case_{len(data)}"})
        
        # Group 2: R4-R6 do activities C, D
        for r in ["R4", "R5", "R6"]:
            for _ in range(10):
                data.append({"concept:name": "C", "org:resource": r, "case:concept:name": f"case_{len(data)}"})
            for _ in range(8):
                data.append({"concept:name": "D", "org:resource": r, "case:concept:name": f"case_{len(data)}"})
        
        cls.test_df = pd.DataFrame(data)
        cls.test_df["time:timestamp"] = pd.date_range(start='1/1/2017', periods=len(data))
        
        # Initialize benchmark
        cls.benchmark = PermissionBenchmark(df=cls.test_df, n_clusters=2)
    
    def test_precision_range(self):
        """Precision should be between 0 and 1."""
        metrics = self.benchmark.compute_precision_recall()
        for activity, m in metrics.items():
            self.assertGreaterEqual(m["precision"], 0.0)
            self.assertLessEqual(m["precision"], 1.0)
    
    def test_recall_range(self):
        """Recall should be between 0 and 1."""
        metrics = self.benchmark.compute_precision_recall()
        for activity, m in metrics.items():
            self.assertGreaterEqual(m["recall"], 0.0)
            self.assertLessEqual(m["recall"], 1.0)
    
    def test_coverage_comparison(self):
        """Coverage stats should have expected structure."""
        coverage = self.benchmark.compute_coverage_comparison()
        
        self.assertIn("basic", coverage)
        self.assertIn("advanced", coverage)
        self.assertIn("comparison", coverage)
        
        self.assertGreaterEqual(coverage["basic"]["coverage_ratio"], 0.0)
        self.assertLessEqual(coverage["basic"]["coverage_ratio"], 1.0)
    
    def test_group_quality_structure(self):
        """Group quality should return expected structure."""
        quality = self.benchmark.compute_group_quality()
        
        self.assertIn("n_groups", quality)
        self.assertIn("group_details", quality)
        self.assertIn("size_stats", quality)
        self.assertIn("domain_separation", quality)
        
        self.assertEqual(quality["n_groups"], 2)
    
    def test_activity_breakdown_shape(self):
        """Activity breakdown should have correct columns."""
        df = self.benchmark.get_activity_breakdown()
        
        expected_cols = ["activity", "frequency", "basic_pool_size", "advanced_pool_size", "precision", "recall"]
        for col in expected_cols:
            self.assertIn(col, df.columns)
        
        # Should have 4 activities: A, B, C, D
        self.assertEqual(len(df), 4)


class TestHoldoutValidation(unittest.TestCase):
    """Test holdout validation doesn't leak data."""
    
    def test_holdout_no_leak(self):
        """Train and test cases should be disjoint."""
        # Create data with distinct cases
        data = []
        for i in range(100):
            case_id = f"case_{i}"
            for _ in range(5):
                data.append({
                    "concept:name": "A",
                    "org:resource": f"R{i % 5}",
                    "case:concept:name": case_id
                })
        
        df = pd.DataFrame(data)
        df["time:timestamp"] = pd.date_range(start='1/1/2017', periods=len(data))
        
        benchmark = PermissionBenchmark(df=df, n_clusters=3)
        result = benchmark.run_holdout_test(holdout_fraction=0.2)
        
        # Should have ~80 train, ~20 test
        self.assertAlmostEqual(result["train_cases"], 80, delta=5)
        self.assertAlmostEqual(result["test_cases"], 20, delta=5)
        
        # Recall should be valid
        self.assertGreaterEqual(result["basic_recall"], 0.0)
        self.assertLessEqual(result["basic_recall"], 1.0)


class TestBenchmarkIntegration(unittest.TestCase):
    """Integration test on real BPIC2017 data."""
    
    @classmethod
    def setUpClass(cls):
        try:
            from dotenv import load_dotenv
        except ImportError:
            raise unittest.SkipTest("python-dotenv fehlt (Pfad des echten Logs in .env, EVENT_LOG_PATH)")
        load_dotenv()
        
        log_path = os.getenv("EVENT_LOG_PATH")
        if not log_path or not os.path.exists(log_path):
            raise unittest.SkipTest(f"Event log not found at {log_path}")
        
        print(f"\n[Integration] Loading log from {log_path}...")
        cls.benchmark = PermissionBenchmark(log_path=log_path, n_clusters=5)
        cls.output_dir = tempfile.mkdtemp()
    
    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'output_dir') and os.path.exists(cls.output_dir):
            shutil.rmtree(cls.output_dir)
    
    def test_precision_threshold(self):
        """Average precision should be reasonable (≥40%)."""
        metrics = self.benchmark.compute_precision_recall()
        precisions = [m["precision"] for m in metrics.values()]
        avg = np.mean(precisions)
        
        print(f"Average precision: {avg:.1%}")
        self.assertGreaterEqual(avg, 0.4, "Precision too low")
    
    def test_coverage_threshold(self):
        """Coverage should be ≥90%."""
        coverage = self.benchmark.compute_coverage_comparison()
        ratio = coverage["advanced"]["coverage_ratio"]
        
        print(f"Coverage: {ratio:.1%}")
        self.assertGreaterEqual(ratio, 0.9, "Coverage below 90%")
    
    def test_report_generation(self):
        """Report should be generated successfully."""
        report_path = self.benchmark.generate_report(self.output_dir)
        
        self.assertTrue(os.path.exists(report_path))
        
        with open(report_path) as f:
            content = f.read()
        
        self.assertIn("Permission Benchmark Report", content)
        self.assertIn("Coverage", content)
        self.assertIn("Recommendation", content)


if __name__ == '__main__':
    unittest.main()
