import unittest
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from resources.resource_permissions.resource_permissions import BasicResourcePermissions

class TestBasicResourcePermissions(unittest.TestCase):
    """
    Unit tests using synthetic data for fast feedback.
    """

    def setUp(self):
        # Create a synthetic DataFrame for testing
        data = {
            "case:concept:name": ["c1", "c1", "c1", "c2", "c2"],
            "concept:name": ["A_Create", "A_Create", "A_Validating", "W_Call", "A_Create"],
            "org:resource": ["User_1", "User_2", "User_1", "User_3", "User_1"],
            "time:timestamp": pd.date_range(start='1/1/2017', periods=5)
        }
        self.test_df = pd.DataFrame(data)

    def test_initialization_with_df(self):
        rp = BasicResourcePermissions(df=self.test_df)
        self.assertIsNotNone(rp.activity_resource_map)

    def test_initialization_error(self):
        with self.assertRaises(ValueError):
            BasicResourcePermissions()

    def test_mapping_correctness(self):
        rp = BasicResourcePermissions(df=self.test_df)
        
        # A_Create was done by User_1 and User_2
        eligible_create = rp.get_eligible_resources("A_Create")
        self.assertEqual(set(eligible_create), {"User_1", "User_2"})
        
        # A_Validating was done by User_1
        eligible_val = rp.get_eligible_resources("A_Validating")
        self.assertEqual(set(eligible_val), {"User_1"})

        # W_Call was done by User_3
        eligible_call = rp.get_eligible_resources("W_Call")
        self.assertEqual(set(eligible_call), {"User_3"})

    def test_unknown_activity(self):
        rp = BasicResourcePermissions(df=self.test_df)
        eligible = rp.get_eligible_resources("Unknown_Activity")
        self.assertEqual(eligible, [])

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=["case:concept:name", "concept:name", "org:resource", "time:timestamp"])
        rp = BasicResourcePermissions(df=empty_df)
        self.assertEqual(rp.get_eligible_resources("Any"), [])

    def test_missing_columns(self):
        bad_df = pd.DataFrame({"wrong_col": [1, 2, 3]})
        with self.assertRaises(ValueError):
            BasicResourcePermissions(df=bad_df)


class TestResourcePermissionsRealLog(unittest.TestCase):
    """
    Tests against the actual BPIC2017 event log using known oracles.
    Skipped if the log file is not present.
    """
    
    rp = None

    @classmethod
    def setUpClass(cls):
        try:
            from dotenv import load_dotenv
        except ImportError:
            raise unittest.SkipTest("python-dotenv fehlt (Pfad des echten Logs in .env, EVENT_LOG_PATH)")
        load_dotenv()
        
        log_path = os.getenv("EVENT_LOG_PATH")
        
        if not log_path or not os.path.exists(log_path):
            raise unittest.SkipTest(f"Event log not found at {log_path} (set EVENT_LOG_PATH in .env)")
        
        print(f"\n[RealLog] Loading log from {log_path} (this may take a while)...")
        # Load once for all tests
        cls.rp = BasicResourcePermissions(log_path=log_path)

    def test_oracle_create_application(self):
        # Oracle: 111 resources (source: Celonis Analysis)
        activity = "A_Create Application"
        resources = self.rp.get_eligible_resources(activity)
        self.assertEqual(len(resources), 111, f"Expected 111 resources for {activity}, found {len(resources)}")

    def test_oracle_assess_potential_fraud(self):
        # Oracle: 56 resources (source: Celonis Analysis)
        activity = "W_Assess potential fraud"
        resources = self.rp.get_eligible_resources(activity)
        self.assertEqual(len(resources), 56, f"Expected 56 resources for {activity}, found {len(resources)}")

    def test_oracle_validate_application(self):
        # Oracle: 133 resources (source: Celonis Analysis)
        activity = "W_Validate application"
        resources = self.rp.get_eligible_resources(activity)
        self.assertEqual(len(resources), 133, f"Expected 133 resources for {activity}, found {len(resources)}")


if __name__ == '__main__':
    unittest.main()
