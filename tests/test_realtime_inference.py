import unittest
from pathlib import Path

from src.models.realtime_inference import (
    DEFAULT_BASELINE_CHECKPOINT,
    MBTARealtimeAdapter,
    RealtimeDelayPredictor,
    benchmark_predictor,
    build_demo_historical_frame,
    build_demo_live_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RealtimeInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.history = build_demo_historical_frame(num_rows=120)
        cls.live_records = build_demo_live_records(count=8)
        cls.predictor = RealtimeDelayPredictor.bootstrap(
            checkpoint_path=DEFAULT_BASELINE_CHECKPOINT,
            historical_data=cls.history,
            sample_size=120,
            device="cpu",
        )

    def test_baseline_checkpoint_exists(self):
        self.assertTrue(DEFAULT_BASELINE_CHECKPOINT.exists())

    def test_baseline_realtime_prediction_smoke(self):
        result = self.predictor.predict_one(self.live_records[0])
        self.assertEqual(result.feature_version, "v1")
        self.assertEqual(result.checkpoint_name, DEFAULT_BASELINE_CHECKPOINT.name)
        self.assertEqual(set(result.feature_values.keys()), set(self.predictor.artifacts.feature_columns))
        self.assertTrue(abs(result.predicted_delay_minutes) < 1000)

    def test_latency_benchmark_returns_summary(self):
        summary = benchmark_predictor(self.predictor, self.live_records, iterations=5, warmup=1)
        self.assertGreater(summary["calls"], 0)
        self.assertGreater(summary["avg_latency_ms"], 0)
        self.assertGreaterEqual(summary["p95_latency_ms"], summary["p50_latency_ms"])
        self.assertGreater(summary["throughput_qps"], 0)

    def test_mbta_adapter_normalizes_json_api_payload(self):
        adapter = MBTARealtimeAdapter()
        payload = {
            "data": [
                {
                    "attributes": {
                        "departure_time": "2026-01-06T07:00:00-05:00",
                        "direction_id": 1,
                        "status": "On time",
                    },
                    "relationships": {
                        "route": {"data": {"id": "22"}},
                        "stop": {"data": {"id": "70091"}},
                        "vehicle": {"data": {"id": "vehicle-1"}},
                    },
                }
            ]
        }

        records = adapter.normalize_records(payload)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["route_id"], "22")
        self.assertEqual(records[0]["stop_id"], "70091")
        self.assertEqual(records[0]["vehicle_id"], "vehicle-1")

    def test_gitignore_blocks_local_agent_workspace(self):
        gitignore_text = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        self.assertIn(".agents/", gitignore_text)
        self.assertIn("models/delay_predictor_mlp_v2_realtime_bundle_stage2.pt", gitignore_text)


if __name__ == "__main__":
    unittest.main()
