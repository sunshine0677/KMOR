import unittest

from static_correctness_experiment import (
    COMPLETE,
    QueryInstance,
    RouteRecord,
    SolverResult,
    compare_results,
    record_from_dict,
    record_to_dict,
    summarize_group,
)


class StaticCorrectnessPathTests(unittest.TestCase):
    def make_result(self, method, full_path):
        route = RouteRecord(
            served_count=1,
            cost=3.0,
            served_requests=("r1",),
            route_events=(("pickup", "r1"), ("dropoff", "r1"), ("target", "")),
            key_path=("s", "p", "d", "t"),
            hops=3,
            full_path=full_path,
        )
        return SolverResult(
            method=method,
            status=COMPLETE,
            frontier=[route],
            routes=[route],
            preprocess_time=0.0,
            search_time=0.0,
            expanded_states=0,
            generated_states=0,
            feasibility_checks=0,
            feasibility_pruned=0,
            heuristic_pruned=0,
            dp_pruned=0,
            peak_open_size=0,
            hit_max_states=False,
            timed_out=False,
        )

    def test_physical_path_mismatch_is_not_hidden_by_event_match(self):
        query = QueryInstance("q1", "OL", 1, 1, "s", "t", 10.0, ())
        reference = self.make_result("Exhaustive", ("s", "p", "d", "t"))
        method = self.make_result("DPKMOR", ("s", "x", "p", "d", "t"))
        row = compare_results(query, reference, method, 1, 1e-6)

        self.assertTrue(row["route_match"])
        self.assertTrue(row["key_path_match"])
        self.assertFalse(row["physical_path_match"])
        self.assertFalse(row["concrete_topk_match"])
        self.assertEqual("recorded", row["physical_path_source"])

    def test_legacy_rows_are_unverified_not_zero_percent_matches(self):
        query = QueryInstance("q1", "OL", 1, 1, "s", "t", 10.0, ())
        original = self.make_result("Exhaustive", None)
        legacy = self.make_result("DPKMOR", None)
        row = compare_results(query, original, legacy, 1, 1e-6)
        summary = summarize_group([row], "OL", 1, 1, row["comparison"])

        self.assertEqual("", row["physical_path_match"])
        self.assertEqual("missing", row["physical_path_source"])
        self.assertEqual(0, summary["physical_path_compared_pairs"])
        self.assertEqual(0, summary["physical_path_recorded_pairs"])

    def test_route_record_roundtrip_preserves_full_path(self):
        route = self.make_result("DPKMOR", ("s", "x", "p", "d", "t")).routes[0]
        self.assertEqual(route, record_from_dict(record_to_dict(route)))
        legacy_payload = record_to_dict(self.make_result("DPKMOR", None).routes[0])
        self.assertNotIn("full_path", legacy_payload)
        self.assertIsNone(record_from_dict(legacy_payload).full_path)


if __name__ == "__main__":
    unittest.main()
