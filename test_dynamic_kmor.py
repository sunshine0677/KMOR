import random
import time
import unittest
from unittest.mock import patch

from Dpkmor import (
    Candidate,
    Graph,
    Request,
    RideSharePlanner,
    SearchCertificate,
    candidate_identity,
    select_top_k_candidates,
)
from dynamic_kmor import (
    DynamicKMORManager,
    EdgeUpdate,
    InsertionLabel,
    RequestArrival,
    RequestCancel,
    RequestStatus,
    TimedEdge,
    TimeDependentGraph,
    UpdateStrategy,
    run_paired_event,
)
from dynamic_kinetic_tree import KineticTreeKMORManager
from dynamic_rhls import RHLSTopKManager
from dynamic_td_insertion import (
    TDInsertionKMORManager,
    TDTopKInsertionKMORManager,
)
from dynamic_experiment_suite import route_diversity_metrics, validate_manager
from dynamic_paired_experiment import (
    aggregate_edge_updates_by_subtype,
    aggregate_events,
    classify_event_subtype,
    comparison_status,
    variant_runtime_settings,
)
from generate_dynamic_workloads import generate_diverse_corridors, generate_requests


class DynamicKMORTests(unittest.TestCase):
    def make_graph(self) -> Graph:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 2.0)
        graph.add_edge("1", "2", 2.0)
        graph.add_edge("2", "3", 2.0)
        return graph

    def make_kinetic_tree_manager(self) -> KineticTreeKMORManager:
        request = Request(
            "r1",
            "1",
            "2",
            demand=1,
            request_time=0.0,
            wait_limit=20.0,
        )
        return KineticTreeKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            strategy=UpdateStrategy.KINETIC_TREE.value,
            capacity=2,
            k=5,
            time_budget=100.0,
            detour_ratio=1.4,
            kinetic_tree_max_schedules=100,
            kinetic_tree_max_generated=1000,
        )

    def make_td_insertion_manager(self) -> TDInsertionKMORManager:
        request = Request(
            "r1",
            "1",
            "2",
            demand=1,
            request_time=0.0,
            wait_limit=20.0,
        )
        return TDInsertionKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            strategy=UpdateStrategy.TD_INSERTION.value,
            capacity=2,
            k=5,
            time_budget=100.0,
            detour_ratio=1.4,
        )

    def make_td_topk_insertion_manager(self) -> TDTopKInsertionKMORManager:
        request = Request(
            "r1",
            "1",
            "2",
            demand=1,
            request_time=0.0,
            wait_limit=20.0,
        )
        return TDTopKInsertionKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            strategy=UpdateStrategy.TD_TOPK_INSERTION.value,
            capacity=2,
            k=5,
            time_budget=100.0,
            detour_ratio=1.4,
            kinetic_tree_max_generated=1000,
        )

    def make_rhls_topk_manager(self) -> RHLSTopKManager:
        request = Request(
            "r1",
            "1",
            "2",
            demand=1,
            request_time=0.0,
            wait_limit=20.0,
        )
        return RHLSTopKManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            strategy=UpdateStrategy.RHLS_TOPK.value,
            capacity=2,
            k=5,
            time_budget=100.0,
            detour_ratio=1.4,
            rhls_pool_limit=50,
            rhls_max_rounds=4,
            rhls_max_generated=1000,
        )

    def test_rhls_uses_exact_initial_archive_then_local_search(self) -> None:
        manager = self.make_rhls_topk_manager()
        initialization = manager.initialize(0.0)

        self.assertEqual("RHLSTopK", initialization.update_mode)
        self.assertTrue(manager.route_archive)
        self.assertFalse(initialization.fallback_recompute)

        result = manager.handle_event(
            RequestArrival(Request("r2", "1", "3", 1, 0.5, 20.0), 0.5)
        )

        self.assertEqual("RHLSTopK", result.update_mode)
        self.assertGreater(result.generated_schedules, 0)
        self.assertIn("rhls_iterations=", result.note)
        self.assertLessEqual(len(manager.route_archive), 50)
        self.assertFalse(result.fallback_recompute)
        self.assertFalse(result.objective_complete)
        valid, error = validate_manager(manager)
        self.assertTrue(valid, error)

    def test_rhls_repairs_cancellation_and_edge_update(self) -> None:
        manager = self.make_rhls_topk_manager()
        manager.initialize(0.0)
        manager.handle_event(
            RequestArrival(Request("r2", "1", "3", 1, 0.5, 20.0), 0.5)
        )

        cancelled = manager.handle_event(RequestCancel("r2", 0.75))
        self.assertEqual(RequestStatus.CANCELLED, manager.request_status["r2"])
        self.assertEqual("RHLSTopK", cancelled.update_mode)
        self.assertTrue(
            all(
                name != "r2"
                for route in manager.route_archive
                for _, name in route.route_events
            )
        )

        updated = manager.handle_event(
            EdgeUpdate("traffic", "1", "2", 1.0, 8.0)
        )
        self.assertEqual("RHLSTopK", updated.update_mode)
        valid, error = validate_manager(manager)
        self.assertTrue(valid, error)

    def test_td_topk_insertion_keeps_a_bounded_candidate_beam(self) -> None:
        manager = self.make_td_topk_insertion_manager()
        initialization = manager.initialize(0.0)

        self.assertEqual("TDTopKInsertion", initialization.update_mode)
        self.assertLessEqual(len(manager.route_archive), 5)
        self.assertGreaterEqual(len(manager.available_routes), len(manager.routes))

        result = manager.handle_event(
            RequestArrival(Request("r2", "1", "3", 1, 0.5, 20.0), 0.5)
        )

        self.assertEqual("TDTopKInsertion", result.update_mode)
        self.assertLessEqual(len(manager.route_archive), 5)
        self.assertGreater(len(manager.available_routes), 1)
        self.assertEqual(
            len(manager.available_routes),
            len({candidate_identity(route) for route in manager.available_routes}),
        )
        valid, error = validate_manager(manager)
        self.assertTrue(valid, error)

    def test_kinetic_tree_exposes_nonformal_schedules_as_backups(self) -> None:
        manager = self.make_kinetic_tree_manager()

        manager.initialize(0.0)

        formal_ids = {candidate_identity(route) for route in manager.routes}
        self.assertTrue(manager.backup_routes)
        self.assertTrue(
            all(
                candidate_identity(route) not in formal_ids
                for route in manager.backup_routes
            )
        )
        self.assertGreater(len(manager.available_routes), len(manager.routes))

    def test_td_insertion_initialization_maintains_one_route(self) -> None:
        manager = self.make_td_insertion_manager()

        result = manager.initialize(0.0)

        self.assertEqual("TDInsertion", result.update_mode)
        self.assertEqual(1, len(manager.routes))
        self.assertEqual(["r1"], manager.routes[0].served_requests)
        valid, error = validate_manager(manager)
        self.assertTrue(valid, error)

    def test_td_insertion_arrival_selects_a_feasible_insertion(self) -> None:
        manager = self.make_td_insertion_manager()
        manager.initialize(0.0)

        result = manager.handle_event(
            RequestArrival(Request("r2", "1", "3", 1, 0.5, 20.0), 0.5)
        )

        self.assertEqual("TDInsertion", result.update_mode)
        self.assertEqual(1, len(manager.routes))
        self.assertIn("r2", manager.routes[0].served_requests)
        self.assertGreater(result.generated_schedules, 0)

    def test_td_insertion_cancellation_removes_request_stops(self) -> None:
        manager = self.make_td_insertion_manager()
        manager.initialize(0.0)

        manager.handle_event(RequestCancel("r1", 0.5))

        self.assertEqual(RequestStatus.CANCELLED, manager.request_status["r1"])
        self.assertTrue(
            all(name != "r1" for action, name in manager.routes[0].route_events)
        )

    def test_td_insertion_edge_update_revalidates_current_route(self) -> None:
        manager = self.make_td_insertion_manager()
        manager.initialize(0.0)
        old_cost = manager.routes[0].total_cost

        result = manager.handle_event(
            EdgeUpdate("traffic", "1", "2", 0.5, 8.0)
        )

        self.assertEqual("TDInsertion", result.update_mode)
        self.assertEqual(1, len(manager.routes))
        self.assertGreaterEqual(manager.routes[0].total_cost, old_cost)
        valid, error = validate_manager(manager)
        self.assertTrue(valid, error)

    def test_kinetic_tree_initialization_retains_optional_and_served_plans(self) -> None:
        manager = self.make_kinetic_tree_manager()

        result = manager.initialize(0.0)

        self.assertEqual("KineticTreeTopK", result.update_mode)
        self.assertTrue(any(not route.served_requests for route in manager.route_archive))
        self.assertTrue(any(route.served_requests == ["r1"] for route in manager.route_archive))
        valid, error = validate_manager(manager)
        self.assertTrue(valid, error)

    def test_kinetic_tree_arrival_inserts_request_into_retained_schedules(self) -> None:
        manager = self.make_kinetic_tree_manager()
        manager.initialize(0.0)
        arrival = RequestArrival(
            Request("r2", "1", "3", 1, 0.5, 20.0),
            0.5,
        )

        result = manager.handle_event(arrival)

        self.assertEqual("KineticTreeTopK", result.update_mode)
        self.assertGreater(result.inserted_routes, 0)
        self.assertTrue(
            any("r2" in route.served_requests for route in manager.route_archive)
        )

    def test_kinetic_tree_cancellation_removes_waiting_request_actions(self) -> None:
        manager = self.make_kinetic_tree_manager()
        manager.initialize(0.0)

        manager.handle_event(RequestCancel("r1", 0.5))

        self.assertEqual(RequestStatus.CANCELLED, manager.request_status["r1"])
        self.assertTrue(
            all(
                name != "r1"
                for route in manager.route_archive
                for _, name in route.route_events
            )
        )

    def test_kinetic_tree_edge_update_revalidates_schedule_tree(self) -> None:
        manager = self.make_kinetic_tree_manager()
        manager.initialize(0.0)
        old_costs = {
            tuple(route.route_events): route.total_cost
            for route in manager.route_archive
        }

        result = manager.handle_event(
            EdgeUpdate("traffic", "1", "2", 0.5, 8.0)
        )

        self.assertEqual("KineticTreeTopK", result.update_mode)
        self.assertTrue(manager.route_archive)
        new_costs = {
            tuple(route.route_events): route.total_cost
            for route in manager.route_archive
        }
        common = set(old_costs) & set(new_costs)
        self.assertTrue(common)
        self.assertTrue(any(new_costs[key] >= old_costs[key] for key in common))

    def test_edge_update_subtype_uses_pre_event_weight(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=1,
            time_budget=100.0,
        )
        increase = EdgeUpdate("traffic", "0", "1", 2.0, 4.0)
        self.assertEqual("WeightIncrease", classify_event_subtype(manager, increase))
        manager.graph.update_edge_weight("0", "1", 2.0, 4.0)
        restore = EdgeUpdate("restore", "0", "1", 3.0, 2.0)
        self.assertEqual(
            "WeightDecreaseOrRestore",
            classify_event_subtype(manager, restore),
        )

    def test_dynamic_aggregation_does_not_mix_k_values(self) -> None:
        rows = [
            {
                "scenario": "baseline",
                "dataset": "OL",
                "k": k,
                "status": "COMPLETE",
                "incremental_time": 1.0,
                "full_recompute_time": 2.0,
                "event_type": "EdgeUpdate",
                "event_subtype": subtype,
            }
            for k, subtype in (
                (1, "WeightIncrease"),
                (5, "WeightDecreaseOrRestore"),
            )
        ]
        aggregates = aggregate_events(rows)
        self.assertEqual({1, 5}, {row["k"] for row in aggregates})
        edge_aggregates = aggregate_edge_updates_by_subtype(rows)
        self.assertEqual(
            {"WeightIncrease", "WeightDecreaseOrRestore"},
            {row["event_subtype"] for row in edge_aggregates},
        )

    def test_incremental_only_disables_every_recovery_layer(self) -> None:
        settings = variant_runtime_settings(
            "incremental_only",
            local_reoptimization_enabled=True,
            backup_routes_enabled=True,
            adaptive_recompute_enabled=True,
        )

        self.assertFalse(settings["enable_local_reoptimization"])
        self.assertFalse(settings["enable_certified_recovery"])
        self.assertFalse(settings["require_incremental_certificate"])
        self.assertFalse(settings["enable_adaptive_recompute"])
        self.assertTrue(settings["enable_backup_routes"])

    def test_incremental_only_diagnostic_does_not_require_certificate(self) -> None:
        common = {
            "incremental_valid": True,
            "full_valid": True,
            "incremental_timed_out": False,
            "full_timed_out": False,
            "incremental_hit_max_states": False,
            "full_hit_max_states": False,
            "incremental_objective_complete": False,
            "incremental_concrete_complete": False,
            "full_objective_complete": True,
            "full_concrete_complete": True,
        }

        self.assertEqual(
            "INCOMPLETE",
            comparison_status(
                **common,
                require_incremental_certificate=True,
            ),
        )
        self.assertEqual(
            "COMPLETE",
            comparison_status(
                **common,
                require_incremental_certificate=False,
            ),
        )

    def test_time_dependent_graph_fork_isolates_edge_updates(self) -> None:
        original = TimeDependentGraph.from_static_graph(self.make_graph())
        branch = original.fork()

        branch.update_edge_weight("0", "1", 0.0, 9.0)
        self.assertEqual(2.0, original.edge_weight_at("0", "1", 0.0))
        self.assertEqual(9.0, branch.edge_weight_at("0", "1", 0.0))
        self.assertEqual(2.0, original.edge_weight_at("1", "0", 0.0))
        self.assertEqual(9.0, branch.edge_weight_at("1", "0", 0.0))

        original.update_edge_weight("1", "2", 0.0, 7.0)
        self.assertEqual(7.0, original.edge_weight_at("1", "2", 0.0))
        self.assertEqual(2.0, branch.edge_weight_at("1", "2", 0.0))

    def test_full_recompute_drops_event_dependent_tdsp_cache_by_default(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            strategy=UpdateStrategy.FULL_RECOMPUTE.value,
            capacity=2,
            k=1,
            time_budget=100.0,
        )
        manager.initialize(0.0)
        stale_key = ("unused", "path", 123.0)
        manager._dynamic_shortest_path_cache[stale_key] = (["unused", "path"], 1.0)

        manager.full_recompute(1.0, "Test")

        self.assertNotIn(stale_key, manager._dynamic_shortest_path_cache)

    def test_warm_full_recompute_can_retain_tdsp_cache_for_audit(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            strategy=UpdateStrategy.FULL_RECOMPUTE.value,
            capacity=2,
            k=1,
            time_budget=100.0,
            full_recompute_reuse_tdsp_cache=True,
        )
        manager.initialize(0.0)
        stale_key = ("unused", "path", 123.0)
        manager._dynamic_shortest_path_cache[stale_key] = (["unused", "path"], 1.0)

        manager.full_recompute(1.0, "Test")

        self.assertIn(stale_key, manager._dynamic_shortest_path_cache)

    def test_diverse_corridors_are_distinct_and_cost_bounded(self) -> None:
        graph = Graph(directed=False)
        for middle, weight in (("1", 1.0), ("2", 1.1), ("3", 1.2)):
            graph.add_edge("0", middle, weight)
            graph.add_edge(middle, "9", weight)

        corridors = generate_diverse_corridors(
            graph,
            "0",
            "9",
            corridor_count=3,
            max_cost_ratio=1.5,
        )

        self.assertEqual(3, len(corridors))
        self.assertEqual(3, len({tuple(path) for path, _, _ in corridors}))
        self.assertTrue(all(cost <= 3.0 for _, _, cost in corridors))

    def test_initial_requests_are_spread_across_available_corridors(self) -> None:
        graph = Graph(directed=False)
        for middle, weight in (("1", 1.0), ("2", 1.1), ("3", 1.2)):
            graph.add_edge("0", middle, weight)
            graph.add_edge(middle, "9", weight)
        corridors = generate_diverse_corridors(graph, "0", "9", 3, 1.5)

        initial, _ = generate_requests(
            "T",
            1,
            corridors[0][0],
            corridors[0][1],
            horizon=2.0,
            initial_count=3,
            arrival_count=0,
            rng=random.Random(1),
            corridors=corridors,
        )

        self.assertEqual({"1", "2", "3"}, {request.pickup for request in initial})

    def test_key_node_precompute_runs_one_dijkstra_per_source(self) -> None:
        class CountingGraph(Graph):
            def __init__(self) -> None:
                super().__init__(directed=False)
                self.source_searches = 0

            def shortest_paths_from(self, source, targets):
                self.source_searches += 1
                return super().shortest_paths_from(source, targets)

        graph = CountingGraph()
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        graph.add_edge("2", "3", 1.0)
        requests = [
            Request("r1", "1", "2", 1, 0.0, 10.0),
            Request("r2", "2", "3", 1, 0.0, 10.0),
        ]

        planner = RideSharePlanner(graph, "0", "3", requests, k=2)

        self.assertLessEqual(graph.source_searches, len(planner.key_nodes))

    def test_undirected_precompute_fills_new_key_node_with_one_search(self) -> None:
        class CountingGraph(Graph):
            def __init__(self) -> None:
                super().__init__(directed=False)
                self.source_searches = 0

            def shortest_paths_from(self, source, targets):
                self.source_searches += 1
                return super().shortest_paths_from(source, targets)

        graph = CountingGraph()
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        graph.add_edge("2", "3", 1.0)
        cache = {}
        RideSharePlanner(
            graph,
            "0",
            "2",
            [Request("old", "1", "2", 1, 0.0, 10.0)],
            shortest_path_cache=cache,
        )
        graph.source_searches = 0

        RideSharePlanner(
            graph,
            "0",
            "2",
            [
                Request("old", "1", "2", 1, 0.0, 10.0),
                Request("new", "3", "2", 1, 0.0, 10.0),
            ],
            shortest_path_cache=cache,
        )

        self.assertEqual(1, graph.source_searches)

    def test_pair_coverage_bound_preserves_concrete_topk(self) -> None:
        graph = Graph(directed=False)
        for left, right in (
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("2", "4"),
            ("3", "9"),
            ("4", "9"),
        ):
            graph.add_edge(left, right, 1.0)
        requests = [
            Request("r1", "1", "3", 1, 0.0, 1.5),
            Request("r2", "2", "4", 1, 0.0, 1.5),
        ]
        dynamic_travel = lambda left, right, _time: graph.shortest_path(
            left,
            right,
        )
        common = dict(
            capacity=2,
            k=5,
            time_budget=20.0,
            detour_ratio=2.0,
            dp_mode="bundled",
            pareto_only=True,
            time_dependent_shortest_path=dynamic_travel,
        )
        bounded = RideSharePlanner(
            graph,
            "0",
            "9",
            requests,
            enable_pair_coverage_bound=True,
            **common,
        )
        unbounded = RideSharePlanner(
            graph,
            "0",
            "9",
            requests,
            enable_pair_coverage_bound=False,
            **common,
        )

        self.assertEqual(1, bounded._max_pairwise_compatible(0b11))
        self.assertEqual(
            [candidate_identity(route) for route in unbounded.solve()],
            [candidate_identity(route) for route in bounded.solve()],
        )

    def test_seed_coverage_certificate_matches_unbounded_search(self) -> None:
        graph = Graph(directed=False)
        for left, right in (
            ("0", "1"),
            ("0", "2"),
            ("1", "3"),
            ("2", "4"),
            ("3", "9"),
            ("4", "9"),
        ):
            graph.add_edge(left, right, 1.0)
        requests = [
            Request("r1", "1", "3", 1, 0.0, 1.5),
            Request("r2", "2", "4", 1, 0.0, 1.5),
        ]
        seed = Candidate(
            key_path=["0", "1", "3", "9"],
            full_path=["0", "1", "3", "9"],
            served_requests=["r1"],
            pickup_info=[("r1", 0.0, 1.0, 1.0)],
            total_cost=3.0,
            total_time=3.0,
            hops=3,
            served_count=1,
            score=99997.0,
            route_events=[
                ("pickup", "r1"),
                ("dropoff", "r1"),
                ("target", ""),
            ],
        )
        dynamic_travel = lambda left, right, _time: graph.shortest_path(
            left,
            right,
        )
        common = dict(
            capacity=2,
            k=5,
            time_budget=20.0,
            detour_ratio=2.0,
            dp_mode="bundled",
            pareto_only=True,
            time_dependent_shortest_path=dynamic_travel,
            enable_pair_coverage_bound=False,
        )
        certified = RideSharePlanner(
            graph,
            "0",
            "9",
            requests,
            initial_candidates=[seed],
            enable_seed_coverage_certificate=True,
            **common,
        )
        reference = RideSharePlanner(
            graph,
            "0",
            "9",
            requests,
            enable_seed_coverage_certificate=False,
            **common,
        )

        self.assertEqual(1, certified._certified_coverage_cap)
        self.assertEqual(
            {candidate_identity(route) for route in reference.solve()},
            {candidate_identity(route) for route in certified.solve()},
        )

    def test_static_top_k_fills_slots_beyond_strict_pareto_front(self) -> None:
        candidates = [
            Candidate(
                key_path=["0", str(index), "9"],
                full_path=["0", str(index), "9"],
                served_requests=[f"r{index}"],
                pickup_info=[],
                total_cost=10.0 + index,
                total_time=10.0 + index,
                hops=2,
                served_count=1,
                score=100000.0 - index,
                route_events=[("pickup", f"r{index}"), ("dropoff", f"r{index}")],
            )
            for index in range(5)
        ]

        routes = select_top_k_candidates(candidates, 5)

        self.assertEqual(5, len(routes))
        self.assertEqual(5, len({tuple(route.route_events) for route in routes}))

    def test_static_planner_can_continue_with_onboard_request(self) -> None:
        request = Request("r1", "1", "3", 1, request_time=0.0, wait_limit=3.0)
        planner = RideSharePlanner(
            self.make_graph(),
            "2",
            "3",
            [request],
            capacity=2,
            k=1,
            initial_time=10.0,
            initial_onboard_names=["r1"],
            initial_pickup_times={"r1": 2.0},
            time_budget=20.0,
            enforce_passenger_detour=False,
        )

        routes = planner.solve()

        self.assertEqual(1, len(routes))
        self.assertEqual(["r1"], routes[0].served_requests)
        self.assertIn(("dropoff", "r1"), routes[0].route_events)

    def test_edge_update_persists_in_later_buckets(self) -> None:
        graph = TimeDependentGraph.from_static_graph(self.make_graph(), bucket_size=10.0)
        graph.update_edge_weight("1", "2", event_time=12.0, new_weight=7.0)

        weights_at_15 = dict(graph.neighbors_at("1", 15.0))
        weights_at_35 = dict(graph.neighbors_at("1", 35.0))

        self.assertEqual(7.0, weights_at_15["2"])
        self.assertEqual(7.0, weights_at_35["2"])

    def test_fifo_closure_prevents_overtaking_after_weight_drop(self) -> None:
        edge = TimedEdge("1", 10.0, {1: 20.0, 2: 10.0})

        arrival_before_recovery = 199.0 + edge.weight_at_time(199.0, 100.0)
        arrival_at_recovery = 200.0 + edge.weight_at_time(200.0, 100.0)

        self.assertEqual(219.0, arrival_before_recovery)
        self.assertEqual(220.0, arrival_at_recovery)
        self.assertLessEqual(arrival_before_recovery, arrival_at_recovery)

    def test_fifo_arrival_function_is_monotone_across_multiple_updates(self) -> None:
        edge = TimedEdge("1", 8.0, {1: 14.0, 2: 5.0, 3: 11.0, 4: 6.0})
        departures = [0.0, 99.9, 100.0, 150.0, 199.9, 200.0, 250.0, 400.0, 450.0]
        arrivals = [
            departure + edge.weight_at_time(departure, 100.0)
            for departure in departures
        ]

        self.assertTrue(
            all(left <= right + 1e-9 for left, right in zip(arrivals, arrivals[1:]))
        )

    def test_edge_update_keeps_parallel_edges_cost_consistent(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 3.0)
        graph.add_edge("0", "1", 5.0)
        dynamic = TimeDependentGraph.from_static_graph(graph, bucket_size=10.0)

        self.assertEqual(3.0, dynamic.edge_weight_at("0", "1", 0.0))
        dynamic.update_edge_weight("0", "1", event_time=10.0, new_weight=8.0)

        self.assertEqual(8.0, dynamic.edge_weight_at("0", "1", 15.0))
        self.assertTrue(
            all(
                edge.weight_at_time(15.0, dynamic.bucket_size) == 8.0
                for edge in dynamic.adj["0"]
                if edge.to_node == "1"
            )
        )

    def test_waiting_request_expires_before_replanning(self) -> None:
        request = Request("r1", "1", "3", 1, request_time=0.0, wait_limit=1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            capacity=2,
            k=1,
            time_budget=20.0,
        )

        manager.initialize(5.0)

        self.assertEqual(RequestStatus.EXPIRED, manager.request_status["r1"])
        self.assertNotIn("r1", manager.active_requests)

    def test_onboard_request_cannot_be_cancelled(self) -> None:
        request = Request("r1", "1", "3", 1, request_time=0.0, wait_limit=10.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            capacity=2,
            k=1,
            time_budget=20.0,
        )
        manager.initialize(0.0)

        result = manager.handle_event(RequestCancel("r1", event_time=3.0))

        self.assertEqual(RequestStatus.ONBOARD, manager.request_status["r1"])
        self.assertIn("ignored_already_onboard", result.note)
        self.assertEqual("IncrementalCancellationRepair", result.update_mode)
        self.assertEqual(0, result.removed_stops)

    def test_waiting_cancellation_removes_stops_without_full_recompute(self) -> None:
        request = Request("r1", "1", "2", 1, request_time=0.0, wait_limit=5.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        manager.initialize(0.0)

        def fail_full_recompute(*args, **kwargs):
            raise AssertionError("RequestCancel unexpectedly triggered FullRecompute")

        manager.full_recompute = fail_full_recompute  # type: ignore[method-assign]
        result = manager.handle_event(RequestCancel("r1", event_time=0.0))

        self.assertEqual(RequestStatus.CANCELLED, manager.request_status["r1"])
        self.assertEqual("IncrementalCancellationRepair", result.update_mode)
        self.assertFalse(result.fallback_recompute)
        self.assertEqual(2, result.removed_stops)
        self.assertGreaterEqual(result.affected_routes, 1)
        self.assertEqual(1, result.dependency_lookups)
        self.assertGreaterEqual(result.dependency_candidates, 1)
        self.assertEqual(0, manager.routes[0].served_count)
        self.assertNotIn("r1", manager.routes[0].served_requests)

    def test_route_dependency_index_tracks_edges_and_requests(self) -> None:
        request = Request("r1", "1", "2", 1, request_time=0.0, wait_limit=5.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        manager.initialize(0.0)
        manager.reset_event_diagnostics()

        request_records = manager.find_routes_serving_request("r1")
        edge_records = manager.find_affected_routes("1", "2", 0.0)

        self.assertTrue(request_records)
        self.assertTrue(edge_records)
        self.assertEqual(2, manager._event_diagnostics.dependency_lookups)
        self.assertGreaterEqual(
            manager._event_diagnostics.dependency_candidates,
            len(request_records) + len(edge_records),
        )

    def test_bundled_plan_dag_matches_individual_suffix_repairs(self) -> None:
        requests = [
            Request("a", "1", "2", 1, 0.0, 10.0),
            Request("b", "2", "3", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            requests,
            capacity=2,
            k=3,
            time_budget=30.0,
        )

        def plan(events, served):
            return Candidate(
                key_path=["0", "3"],
                full_path=["0", "3"],
                served_requests=list(served),
                pickup_info=[],
                total_cost=30.0,
                total_time=30.0,
                hops=1,
                served_count=len(served),
                score=0.0,
                route_events=list(events) + [("target", "")],
            )

        candidates = [
            plan([("pickup", "a"), ("dropoff", "a")], ["a"]),
            plan(
                [
                    ("pickup", "a"),
                    ("pickup", "b"),
                    ("dropoff", "a"),
                    ("dropoff", "b"),
                ],
                ["a", "b"],
            ),
        ]
        expected = [
            repaired
            for candidate in manager.route_plan_representatives(candidates)
            if (
                repaired := manager.repair_candidate_suffix(
                    manager.graph,
                    candidate,
                    0.0,
                )
            )
            is not None
        ]
        manager.reset_event_diagnostics()

        actual = manager.repair_candidates_bundled(candidates, 0.0)

        def result_map(values):
            return {
                tuple(candidate.route_events): (
                    candidate.total_cost,
                    tuple(candidate.full_path),
                    tuple(candidate.served_requests),
                )
                for candidate in values
            }

        self.assertEqual(result_map(expected), result_map(actual))
        self.assertEqual(1, manager._event_diagnostics.plan_dag_builds)
        self.assertGreater(
            manager._event_diagnostics.plan_prefix_transitions_saved,
            0,
        )

        manager.reset_event_diagnostics()
        repeated = manager.repair_candidates_bundled(candidates, 0.0)
        self.assertEqual(result_map(actual), result_map(repeated))
        self.assertEqual(1, manager._event_diagnostics.plan_dag_reuses)
        self.assertEqual(0, manager._event_diagnostics.plan_prefix_labels_evaluated)
        self.assertGreater(manager._event_diagnostics.plan_prefix_cache_hits, 0)

    def test_batched_request_insertion_matches_per_plan_dp(self) -> None:
        graph = Graph(directed=False)
        for left, right in zip("0123", "123"):
            graph.add_edge(left, right, 1.0)
        requests = [
            Request("a", "1", "2", 1, 0.0, 10.0),
            Request("b", "2", "3", 1, 0.0, 10.0),
            Request("new", "1", "2", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "3",
            requests,
            capacity=2,
            k=5,
            time_budget=20.0,
            detour_ratio=3.0,
        )

        def plan(name):
            return Candidate(
                key_path=["0", "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=[name],
                pickup_info=[],
                total_cost=3.0,
                total_time=3.0,
                hops=3,
                served_count=1,
                score=99997.0,
                route_events=[
                    ("pickup", name),
                    ("dropoff", name),
                    ("target", ""),
                ],
            )

        bases = [plan("a"), plan("b")]
        actual, _, _, hit_max_states = manager.dp_insert_request_batch(
            bases,
            manager.active_requests["new"],
            0.0,
        )

        self.assertFalse(hit_max_states)
        expected_frontier = {
            candidate_identity(candidate)
            for candidate in manager.select_top_k_repaired(
                [
                    candidate
                    for base in bases
                    for candidate in manager.dp_insert_request(
                        manager.graph,
                        base,
                        manager.active_requests["new"],
                        0.0,
                    )[0]
                ]
            )
        }
        actual_frontier = {
            candidate_identity(candidate)
            for candidate in manager.select_top_k_repaired(actual)
        }
        self.assertEqual(expected_frontier, actual_frontier)

    def test_large_insertion_batch_uses_exact_per_plan_fallback(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [
                Request("old", "1", "2", 1, 0.0, 10.0),
                Request("new", "1", "2", 1, 0.0, 10.0),
            ],
            capacity=2,
            k=2,
            time_budget=20.0,
            arrival_batch_sequence_limit=1,
        )
        base = Candidate(
            key_path=["0", "3"],
            full_path=["0", "1", "2", "3"],
            served_requests=["old"],
            pickup_info=[],
            total_cost=3.0,
            total_time=3.0,
            hops=3,
            served_count=1,
            score=99997.0,
            route_events=[
                ("pickup", "old"),
                ("dropoff", "old"),
                ("target", ""),
            ],
        )
        calls = 0
        original = manager.dp_insert_request

        def counting_insert(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        manager.dp_insert_request = counting_insert  # type: ignore[method-assign]
        candidates, _, _, hit_max_states = manager.dp_insert_request_batch(
            [base],
            manager.active_requests["new"],
            0.0,
        )

        self.assertEqual(1, calls)
        self.assertFalse(hit_max_states)
        self.assertTrue(candidates)

    def test_slow_edge_reopens_only_dependent_plan_prefix_branch(self) -> None:
        graph = Graph(directed=False)
        for left, right, weight in (
            ("0", "1", 1.0),
            ("1", "3", 1.0),
            ("1", "2", 1.0),
            ("2", "3", 1.0),
        ):
            graph.add_edge(left, right, weight)
        requests = [
            Request("a", "1", "3", 1, 0.0, 10.0),
            Request("b", "2", "2", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "3",
            requests,
            capacity=2,
            k=3,
            time_budget=30.0,
            detour_ratio=3.0,
        )

        def plan(events, served):
            return Candidate(
                key_path=["0", "3"],
                full_path=["0", "3"],
                served_requests=list(served),
                pickup_info=[],
                total_cost=30.0,
                total_time=30.0,
                hops=1,
                served_count=len(served),
                score=0.0,
                route_events=list(events) + [("target", "")],
            )

        candidates = [
            plan([("pickup", "a"), ("dropoff", "a")], ["a"]),
            plan(
                [
                    ("pickup", "a"),
                    ("pickup", "b"),
                    ("dropoff", "b"),
                    ("dropoff", "a"),
                ],
                ["a", "b"],
            ),
        ]
        initial = manager.repair_candidates_bundled(candidates, 0.0)
        self.assertEqual(2, len(initial))
        manager.reset_event_diagnostics()

        manager.apply_edge_update(EdgeUpdate("slow", "1", "3", 0.0, 5.0))
        repaired = manager.repair_candidates_bundled(candidates, 0.0)

        expected = [
            value
            for candidate in manager.route_plan_representatives(candidates)
            if (
                value := manager.repair_candidate_suffix(
                    manager.graph,
                    candidate,
                    0.0,
                )
            )
            is not None
        ]
        self.assertEqual(
            {
                tuple(value.route_events): (
                    value.total_cost,
                    tuple(value.full_path),
                )
                for value in expected
            },
            {
                tuple(value.route_events): (
                    value.total_cost,
                    tuple(value.full_path),
                )
                for value in repaired
            },
        )
        self.assertEqual(1, manager._event_diagnostics.plan_prefix_invalidations)
        self.assertEqual(1, manager._event_diagnostics.plan_prefix_labels_reopened)
        self.assertGreater(manager._event_diagnostics.plan_prefix_cache_hits, 1)

    def test_edge_update_repairs_suffix_without_full_recompute(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "3", 1.0)
        graph.add_edge("0", "2", 2.0)
        graph.add_edge("2", "3", 2.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph, bucket_size=10.0),
            "0",
            "3",
            [],
            capacity=2,
            k=1,
            time_budget=20.0,
        )
        manager.initialize(0.0)

        def fail_full_recompute(*args, **kwargs):
            raise AssertionError("EdgeUpdate unexpectedly triggered FullRecompute")

        manager.full_recompute = fail_full_recompute  # type: ignore[method-assign]
        result = manager.handle_event(EdgeUpdate("e13", "1", "3", 0.0, 10.0))

        self.assertEqual("IncrementalSuffixRepair", result.update_mode)
        self.assertFalse(result.fallback_recompute)
        self.assertEqual(0, result.expanded_states)
        self.assertEqual(1, result.affected_routes)
        self.assertEqual(["0", "2", "3"], manager.routes[0].full_path)
        self.assertEqual(4.0, result.best_cost)

    def test_same_state_slowdown_repairs_only_affected_route_plans(self) -> None:
        graph = Graph(directed=False)
        for u, v, weight in (
            ("0", "1", 1.0),
            ("1", "9", 1.0),
            ("0", "2", 2.0),
            ("2", "9", 2.0),
        ):
            graph.add_edge(u, v, weight)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [Request("b", "2", "9", 1, 0.0, 10.0)],
            k=2,
            time_budget=20.0,
        )

        def empty_route(path, cost):
            return Candidate(
                key_path=["0", "9"],
                full_path=path,
                served_requests=[],
                pickup_info=[],
                total_cost=cost,
                total_time=cost,
                hops=len(path) - 1,
                served_count=0,
                score=-cost,
                route_events=[("target", "")],
            )

        manager.routes = [
            empty_route(["0", "1", "9"], 2.0),
            Candidate(
                key_path=["0", "2", "9", "9"],
                full_path=["0", "2", "9"],
                served_requests=["b"],
                pickup_info=[("b", 0.0, 2.0, 2.0)],
                total_cost=4.0,
                total_time=4.0,
                hops=2,
                served_count=1,
                score=99996.0,
                route_events=[
                    ("pickup", "b"),
                    ("dropoff", "b"),
                    ("target", ""),
                ],
            ),
        ]
        manager.rebuild_route_records(0.0)

        result = manager.handle_event(
            EdgeUpdate("slow", "1", "9", event_time=0.0, new_weight=10.0)
        )

        self.assertEqual(1, result.affected_routes)
        self.assertEqual(1, result.repaired_routes)
        self.assertEqual(1, result.reused_routes)

    def test_all_retained_routes_affected_refreshes_local_frontier(self) -> None:
        request = Request("old", "1", "2", 1, 0.0, 10.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [request],
            capacity=2,
            k=2,
            time_budget=20.0,
            enable_backup_routes=False,
        )
        candidate = Candidate(
            key_path=["0", "1", "2", "3"],
            full_path=["0", "1", "2", "3"],
            served_requests=["old"],
            pickup_info=[("old", 0.0, 2.0, 2.0)],
            total_cost=6.0,
            total_time=6.0,
            hops=3,
            served_count=1,
            score=99994.0,
            route_events=[
                ("pickup", "old"),
                ("dropoff", "old"),
                ("target", ""),
            ],
        )
        manager.routes = [candidate]
        manager.route_archive = [candidate]
        local_calls = []
        manager.repair_candidates_bundled = (
            lambda candidates, planning_time: list(candidates)
        )

        def local_refresh(candidates, planning_time, **kwargs):
            local_calls.append(list(candidates))
            manager._last_local_search_certificate = SearchCertificate(
                search_exhausted=True,
                scope_complete=True,
                objective_complete=True,
                concrete_complete=True,
            )
            return list(candidates), True, 0, 0, False

        manager.local_reoptimize_frontier = local_refresh

        result = manager.incremental_edge_update(
            event_time=0.0,
            previous_routes=[candidate],
            affected_routes=1,
            note="edge",
            update_start=time.perf_counter(),
        )

        self.assertEqual(1, len(local_calls))
        self.assertEqual("IncrementalEdgeFrontierRefresh", result.update_mode)
        self.assertIn("all_retained_routes_affected", result.note)
        self.assertFalse(result.fallback_recompute)

    def test_committed_prefix_archive_contraction_refreshes_local_frontier(self) -> None:
        requests = [
            Request("a", "1", "2", 1, 0.0, 10.0),
            Request("b", "2", "3", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            requests,
            capacity=2,
            k=2,
            time_budget=20.0,
            enable_backup_routes=False,
        )

        def plan(name, pickup, dropoff):
            return Candidate(
                key_path=["0", pickup, dropoff, "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=[name],
                pickup_info=[(name, 0.0, 2.0, 2.0)],
                total_cost=6.0,
                total_time=6.0,
                hops=3,
                served_count=1,
                score=99994.0,
                route_events=[
                    ("pickup", name),
                    ("dropoff", name),
                    ("target", ""),
                ],
            )

        first = plan("a", "1", "2")
        second = plan("b", "2", "3")
        manager.routes = [first, second]
        manager.route_archive = [first, second]
        manager.repair_candidates_bundled = (
            lambda candidates, planning_time: [first]
        )
        local_calls = []

        def local_refresh(candidates, planning_time, **kwargs):
            local_calls.append(list(candidates))
            manager._last_local_search_certificate = SearchCertificate(
                search_exhausted=True,
                scope_complete=True,
                objective_complete=True,
                concrete_complete=True,
            )
            return [first, second], True, 0, 0, False

        manager.local_reoptimize_frontier = local_refresh

        result = manager.incremental_edge_update(
            event_time=1.0,
            previous_routes=[first, second],
            affected_routes=0,
            state_advanced=True,
            note="edge",
            update_start=time.perf_counter(),
        )

        self.assertEqual(1, len(local_calls))
        self.assertEqual("IncrementalEdgeFrontierRefresh", result.update_mode)
        self.assertIn("committed_prefix_archive_contraction", result.note)
        self.assertFalse(result.fallback_recompute)

    def test_formal_frontier_contraction_refreshes_local_frontier(self) -> None:
        requests = [
            Request("a", "1", "2", 1, 0.0, 10.0),
            Request("b", "2", "3", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            requests,
            capacity=2,
            k=5,
            time_budget=20.0,
            enable_backup_routes=False,
        )

        def plan(names, cost):
            events = []
            for name in names:
                events.extend((("pickup", name), ("dropoff", name)))
            events.append(("target", ""))
            return Candidate(
                key_path=["0", "1", "2", "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=list(names),
                pickup_info=[],
                total_cost=cost,
                total_time=cost,
                hops=3,
                served_count=len(names),
                score=len(names) * 100000.0 - cost,
                route_events=events,
            )

        high_coverage = plan(["a", "b"], 8.0)
        low_cost = plan(["a"], 6.0)
        newly_dominated = plan(["a"], 9.0)
        # Vehicle advancement may already project the live formal layer before
        # suffix repair begins. The pre-event cardinality must remain the
        # comparison certificate.
        manager.routes = [high_coverage]
        manager.route_archive = [high_coverage, low_cost]
        manager.repair_candidates_bundled = (
            lambda candidates, planning_time: [high_coverage, newly_dominated]
        )
        local_calls = []

        def local_refresh(candidates, planning_time, **kwargs):
            local_calls.append(list(candidates))
            manager._last_local_search_certificate = SearchCertificate(
                search_exhausted=True,
                scope_complete=True,
                objective_complete=True,
                concrete_complete=True,
            )
            return [high_coverage, low_cost], True, 0, 0, False

        manager.local_reoptimize_frontier = local_refresh

        result = manager.incremental_edge_update(
            event_time=0.0,
            previous_routes=[high_coverage, low_cost],
            affected_routes=0,
            previous_formal_count=2,
            note="edge",
            update_start=time.perf_counter(),
        )

        self.assertEqual(1, len(local_calls))
        self.assertEqual("IncrementalEdgeFrontierRefresh", result.update_mode)
        self.assertIn("formal_frontier_contraction", result.note)
        self.assertFalse(result.fallback_recompute)

    def test_slower_edge_invalidates_only_cached_paths_using_that_edge(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "3", 1.0)
        graph.add_edge("0", "2", 2.0)
        graph.add_edge("2", "3", 2.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "3",
            [],
        )
        manager.cached_dynamic_shortest_path("0", "3", 0.0)
        manager.cached_dynamic_shortest_path("0", "2", 0.0)
        manager.reset_event_diagnostics()

        manager.apply_edge_update(EdgeUpdate("e13", "1", "3", 0.0, 10.0))

        self.assertNotIn(("0", "3", 0.0), manager._dynamic_shortest_path_cache)
        self.assertIn(("0", "2", 0.0), manager._dynamic_shortest_path_cache)
        self.assertEqual(1, manager._event_diagnostics.tdsp_cache_invalidations)
        self.assertNotIn(
            ("0", "3", 0.0),
            manager._dynamic_path_keys_by_edge.get(
                manager.graph.edge_key("1", "3"), set()
            ),
        )

    def test_dynamic_shortest_path_diagnostics_track_exact_reuse(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
        )
        manager.reset_event_diagnostics()

        first = manager.cached_dynamic_shortest_path("0", "3", 0.0)
        second = manager.cached_dynamic_shortest_path("0", "3", 0.0)

        self.assertEqual(first, second)
        self.assertEqual(2, manager._event_diagnostics.tdsp_queries)
        self.assertEqual(1, manager._event_diagnostics.tdsp_cache_hits)
        self.assertEqual(1, manager._event_diagnostics.tdsp_lower_path_reuses)
        self.assertEqual(0, manager._event_diagnostics.tdsp_dijkstra_runs)

    def test_dynamic_shortest_path_diagnostics_track_td_dijkstra(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "3", 1.0)
        graph.add_edge("0", "2", 2.0)
        graph.add_edge("2", "3", 2.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "3",
            [],
        )
        manager.apply_edge_update(EdgeUpdate("e13", "1", "3", 0.0, 10.0))
        manager.reset_event_diagnostics()

        path, cost = manager.cached_dynamic_shortest_path("0", "3", 0.0)

        self.assertEqual(["0", "2", "3"], path)
        self.assertEqual(4.0, cost)
        self.assertEqual(1, manager._event_diagnostics.tdsp_dijkstra_runs)
        self.assertEqual(0, manager._event_diagnostics.tdsp_lower_path_reuses)

    def test_faster_edge_invalidates_all_dynamic_path_cache(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "3", 1.0)
        graph.add_edge("0", "2", 2.0)
        graph.add_edge("2", "3", 2.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "3",
            [],
        )
        manager.cached_dynamic_shortest_path("0", "3", 0.0)
        manager.cached_dynamic_shortest_path("0", "2", 0.0)

        manager.apply_edge_update(EdgeUpdate("e02", "0", "2", 0.0, 0.5))

        self.assertFalse(manager._dynamic_shortest_path_cache)

    def test_deviation_path_cache_tracks_lower_bound_revision(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
        )
        blocked = {manager.graph.edge_key("1", "2")}
        first = manager.cached_deviation_shortest_path("0", "3", blocked)
        second = manager.cached_deviation_shortest_path("0", "3", blocked)
        first_lower_graph = manager._lower_bound_graph_cache
        self.assertEqual(first, second)
        self.assertEqual(1, len(manager._deviation_shortest_path_cache))

        manager.apply_edge_update(EdgeUpdate("e01", "0", "1", 0.0, 3.0))
        manager.cached_deviation_shortest_path("0", "3", blocked)
        self.assertEqual(1, len(manager._deviation_shortest_path_cache))

        manager.apply_edge_update(EdgeUpdate("e01", "0", "1", 0.0, 1.0))
        manager.cached_deviation_shortest_path("0", "3", blocked)
        self.assertEqual(1, len(manager._deviation_shortest_path_cache))
        self.assertIsNot(first_lower_graph, manager._lower_bound_graph_cache)

    def test_lower_bound_reuses_exact_shortest_path_corridor_segments(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
        )
        manager.cached_lower_bound_shortest_path("0", "3")

        path, cost = manager.cached_lower_bound_shortest_path("2", "0")

        self.assertEqual(["2", "1", "0"], path)
        self.assertEqual(4.0, cost)
        self.assertEqual(["0", "1", "2", "3"], manager._corridor_path)

    def test_request_arrival_uses_incremental_dp_without_full_recompute(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=3,
            time_budget=20.0,
        )
        manager.initialize(0.0)

        def fail_full_recompute(*args, **kwargs):
            raise AssertionError("RequestArrival unexpectedly triggered FullRecompute")

        manager.full_recompute = fail_full_recompute  # type: ignore[method-assign]
        request = Request("new", "1", "2", 1, request_time=0.0, wait_limit=5.0)
        result = manager.handle_event(RequestArrival(request, event_time=0.0))

        self.assertEqual("IncrementalDPInsertion", result.update_mode)
        self.assertFalse(result.fallback_recompute)
        self.assertGreater(result.expanded_states, 0)
        self.assertGreaterEqual(result.inserted_routes, 1)
        self.assertGreater(result.insertion_transitions_tested, 0)
        self.assertGreater(result.insertion_labels_generated, 0)
        self.assertGreater(result.tdsp_queries, 0)
        self.assertEqual(1, manager.routes[0].served_count)
        self.assertIn("new", manager.routes[0].served_requests)

    def test_internal_archive_does_not_leak_dominated_plan_to_output(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            k=2,
            archive_factor=3,
            enable_backup_routes=False,
        )
        candidates = [
            Candidate(
                key_path=["0", str(index), "3"],
                full_path=["0", str(index), "3"],
                served_requests=["r"],
                pickup_info=[],
                total_cost=cost,
                total_time=cost,
                hops=2,
                served_count=1,
                score=100000.0 - cost,
                route_events=[("pickup", "r"), ("dropoff", "r")],
            )
            for index, cost in ((1, 5.0), (2, 7.0))
        ]

        output = manager.maintain_top_k_routes(candidates, 0.0)

        self.assertEqual([5.0], [candidate.total_cost for candidate in output])
        self.assertEqual(2, len(manager.route_archive))

    def test_request_arrival_can_replace_one_waiting_request(self) -> None:
        graph = Graph(directed=False)
        for left, right in zip("01234", "1234"):
            graph.add_edge(left, right, 1.0)
        old = Request("old", "1", "4", 1, 0.0, 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "4",
            [old],
            capacity=1,
            k=2,
            time_budget=20.0,
            replacement_limit_per_route=2,
            adaptive_recompute_on_failed_arrival=False,
        )
        manager.initialize(0.0)
        new = Request("new", "2", "3", 1, 0.0, 2.0)

        result = manager.handle_event(RequestArrival(new, 0.0))

        self.assertGreaterEqual(result.inserted_routes, 1)
        self.assertTrue(
            any("new" in candidate.served_requests for candidate in manager.route_archive)
        )

    def test_arrival_companion_layer_can_restore_coverage_gain(self) -> None:
        requests = [
            Request(name, "1", "2", 1, 0.0, 10.0)
            for name in ("a", "b", "c")
        ]
        requests.append(Request("companion", "1", "2", 1, 0.0, 3.0))
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            requests,
            capacity=4,
            k=2,
            time_budget=20.0,
            arrival_companion_limit=2,
            enable_arrival_local_reoptimization=False,
            adaptive_recompute_on_failed_arrival=False,
        )

        def candidate(names):
            events = []
            for name in names:
                events.extend((("pickup", name), ("dropoff", name)))
            events.append(("target", ""))
            return Candidate(
                key_path=["0", "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=list(names),
                pickup_info=[],
                total_cost=4.0,
                total_time=4.0,
                hops=3,
                served_count=len(names),
                score=len(names) * 100000.0 - 4.0,
                route_events=events,
            )

        high = candidate(["a", "b", "c"])
        low = candidate(["a", "b"])
        manager.routes = [high, low]
        manager.route_archive = [high, low]
        manager.repair_candidate_suffix = lambda graph, route, planning_time: route
        manager.rebuild_route_records = lambda planning_time: None

        def fake_insert(bases, request, planning_time):
            inserted = []
            for base in bases:
                served = list(base.served_requests)
                if request.name == "new" and len(served) == 2:
                    inserted.append(candidate([*served, "new"]))
                elif request.name == "companion" and "new" in served:
                    inserted.append(candidate([*served, "companion"]))
            return inserted, len(bases), 0, False

        manager.dp_insert_request_batch = fake_insert

        result = manager.handle_event(
            RequestArrival(Request("new", "1", "2", 1, 0.0, 10.0), 0.0)
        )

        self.assertEqual(4, result.best_served)
        self.assertTrue(
            any(candidate.served_count == 4 for candidate in manager.route_archive)
        )

    def test_coverage_stagnation_runs_exhaustive_local_reoptimization(self) -> None:
        old = Request("old", "1", "2", 1, 0.0, 3.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [old],
            capacity=1,
            k=2,
            time_budget=8.0,
            arrival_companion_limit=0,
            arrival_local_reopt_request_limit=4,
        )
        manager.initialize(0.0)
        new = Request("new", "1", "2", 1, 0.0, 3.0)

        paired = run_paired_event(manager, RequestArrival(new, 0.0))
        incremental = paired.incremental_result
        full = paired.full_result

        self.assertEqual("IncrementalLocalReoptimization", incremental.update_mode)
        self.assertEqual(1, incremental.local_reoptimization_runs)
        self.assertEqual(2, incremental.local_reoptimization_requests)
        self.assertGreater(incremental.local_reoptimization_expanded, 0)
        self.assertFalse(incremental.fallback_recompute)
        self.assertEqual(full.best_served, incremental.best_served)
        self.assertAlmostEqual(full.best_cost, incremental.best_cost)
        self.assertEqual(
            {
                candidate_identity(candidate)
                for candidate in paired.full_manager.routes
            },
            {
                candidate_identity(candidate)
                for candidate in paired.incremental_manager.routes
            },
        )

    def test_saturated_archive_refreshes_arrival_after_coverage_gain(self) -> None:
        old = Request("old", "1", "2", 1, 0.0, 5.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [old],
            capacity=2,
            k=2,
            time_budget=12.0,
            arrival_local_reopt_request_limit=4,
        )
        manager.initialize(0.0)
        # Model a previous bounded archive whose next concrete plans were not
        # retained. Coverage growth alone must not be treated as a proof that
        # the new plan-level frontier is complete.
        manager.route_archive_saturated = True
        new = Request("new", "1", "2", 1, 0.0, 5.0)

        paired = run_paired_event(manager, RequestArrival(new, 0.0))
        incremental = paired.incremental_result

        self.assertGreater(incremental.best_served, 1)
        self.assertEqual("IncrementalLocalReoptimization", incremental.update_mode)
        self.assertIn("prior_archive_saturated", incremental.note)
        self.assertEqual(1, incremental.local_reoptimization_runs)
        self.assertEqual(
            {
                candidate_identity(candidate)
                for candidate in paired.full_manager.routes
            },
            {
                candidate_identity(candidate)
                for candidate in paired.incremental_manager.routes
            },
        )

    def test_underfilled_plan_frontier_refreshes_after_coverage_gain(self) -> None:
        old = Request("old", "1", "2", 1, 0.0, 8.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [old],
            capacity=2,
            k=10,
            time_budget=20.0,
            arrival_local_reopt_request_limit=4,
        )
        manager.initialize(0.0)
        new = Request("new", "1", "2", 1, 0.0, 8.0)

        paired = run_paired_event(manager, RequestArrival(new, 2.0))
        incremental = paired.incremental_result

        self.assertEqual("IncrementalLocalReoptimization", incremental.update_mode)
        self.assertIn("mixed_onboard_waiting_order", incremental.note)
        self.assertEqual(1, incremental.local_reoptimization_runs)
        self.assertFalse(incremental.fallback_recompute)
        self.assertEqual(
            {
                candidate_identity(candidate)
                for candidate in paired.full_manager.routes
            },
            {
                candidate_identity(candidate)
                for candidate in paired.incremental_manager.routes
            },
        )

    def test_cancellation_refills_released_plan_with_waiting_request(self) -> None:
        graph = Graph(directed=False)
        for left, right in zip("01234", "1234"):
            graph.add_edge(left, right, 1.0)
        old = Request("old", "1", "4", 1, 0.0, 2.0)
        spare = Request("spare", "2", "3", 1, 0.0, 10.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "4",
            [old, spare],
            capacity=1,
            k=2,
            time_budget=20.0,
            refill_request_limit=2,
        )
        old_plan = Candidate(
            key_path=["0", "1", "4", "4"],
            full_path=["0", "1", "2", "3", "4"],
            served_requests=["old"],
            pickup_info=[("old", 0.0, 1.0, 1.0)],
            total_cost=4.0,
            total_time=4.0,
            hops=4,
            served_count=1,
            score=99996.0,
            route_events=[("pickup", "old"), ("dropoff", "old"), ("target", "")],
        )
        manager.routes = [old_plan]
        manager.route_archive = [old_plan]
        manager.rebuild_route_records(0.0)

        result = manager.handle_event(RequestCancel("old", 0.0))

        self.assertEqual("IncrementalCancellationRefill", result.update_mode)
        self.assertTrue(
            any("spare" in candidate.served_requests for candidate in manager.routes)
        )

    def test_cancellation_refill_can_insert_two_waiting_requests(self) -> None:
        graph = Graph(directed=False)
        for left, right in zip("01234", "1234"):
            graph.add_edge(left, right, 1.0)
        old = Request("old", "1", "4", 1, 0.0, 2.0)
        spare_one = Request("spare-one", "1", "2", 1, 0.0, 10.0)
        spare_two = Request("spare-two", "2", "3", 1, 0.0, 10.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "4",
            [old, spare_one, spare_two],
            capacity=1,
            k=2,
            time_budget=20.0,
            cancel_refill_depth=2,
            adaptive_recompute_on_archive_depletion=False,
        )
        old_plan = Candidate(
            key_path=["0", "1", "4", "4"],
            full_path=["0", "1", "2", "3", "4"],
            served_requests=["old"],
            pickup_info=[("old", 0.0, 1.0, 1.0)],
            total_cost=4.0,
            total_time=4.0,
            hops=4,
            served_count=1,
            score=99996.0,
            route_events=[("pickup", "old"), ("dropoff", "old"), ("target", "")],
        )
        manager.routes = [old_plan]
        manager.route_archive = [old_plan]
        manager.rebuild_route_records(0.0)

        manager.handle_event(RequestCancel("old", 0.0))

        self.assertTrue(
            any(
                {"spare-one", "spare-two"}.issubset(candidate.served_requests)
                for candidate in manager.route_archive
            )
        )

    def test_cancellation_prefilters_each_waiting_request_once(self) -> None:
        old = Request("old", "1", "2", 1, 0.0, 10.0)
        spare_one = Request("spare-one", "1", "2", 1, 0.0, 10.0)
        spare_two = Request("spare-two", "1", "2", 1, 0.0, 10.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [old, spare_one, spare_two],
            k=2,
            time_budget=20.0,
            cancel_refill_depth=1,
            adaptive_recompute_on_archive_depletion=False,
        )

        def candidate(names):
            events = []
            for name in names:
                events.extend((("pickup", name), ("dropoff", name)))
            events.append(("target", ""))
            return Candidate(
                key_path=["0", "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=list(names),
                pickup_info=[],
                total_cost=4.0,
                total_time=4.0,
                hops=3,
                served_count=len(names),
                score=len(names) * 100000.0 - 4.0,
                route_events=events,
            )

        routes = [candidate(["old"]), candidate(["old", "spare-one"])]
        manager.request_status["old"] = RequestStatus.CANCELLED
        manager.active_requests.pop("old")
        manager.repair_candidate_suffix = (
            lambda graph, route, planning_time: route
        )
        manager.rebuild_route_records = lambda planning_time: None

        compatibility_checks = []

        def compatible(request, planning_time):
            compatibility_checks.append(request.name)
            return True

        manager.is_request_compatible = compatible
        manager.dp_insert_request = lambda *args, **kwargs: ([], 1, 0)
        manager.incremental_request_cancel(
            event_time=0.0,
            request_name="old",
            previous_routes=routes,
            cancellation_applied=True,
            note="old",
            update_start=time.perf_counter(),
        )

        self.assertEqual(["spare-one", "spare-two"], compatibility_checks)

    def test_cancellation_archive_depletion_triggers_adaptive_recompute(self) -> None:
        old = Request("old", "1", "2", 1, 0.0, 10.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [old],
            capacity=2,
            k=5,
            time_budget=20.0,
            archive_depletion_ratio=0.5,
        )
        candidates = [
            Candidate(
                key_path=["0", "1", "2", "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=["old"],
                pickup_info=[("old", 0.0, 2.0, 2.0)],
                total_cost=6.0 + index,
                total_time=6.0 + index,
                hops=3,
                served_count=1,
                score=99994.0 - index,
                route_events=[
                    ("pickup", "old"),
                    ("dropoff", "old"),
                    ("target", ""),
                ],
            )
            for index in range(5)
        ]
        manager.routes = candidates[:1]
        manager.route_archive = candidates
        manager.rebuild_route_records(0.0)

        result = manager.handle_event(RequestCancel("old", 0.0))

        self.assertTrue(result.fallback_recompute)
        self.assertTrue(result.adaptive_recompute)
        self.assertEqual("AdaptiveFullRecompute", result.update_mode)
        self.assertIn("archive_depleted_5_to_1", result.note)

    def test_dynamic_top_k_remains_distinct_across_events(self) -> None:
        graph = Graph(directed=False)
        for u, v, weight in (
            ("0", "1", 1.0),
            ("1", "2", 1.0),
            ("2", "3", 1.0),
            ("0", "4", 1.0),
            ("4", "2", 1.0),
            ("1", "5", 1.0),
            ("5", "3", 1.0),
        ):
            graph.add_edge(u, v, weight)
        requests = [
            Request("a", "1", "2", 1, 0.0, 10.0),
            Request("b", "4", "3", 1, 0.0, 10.0),
            Request("c", "2", "5", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "3",
            requests,
            capacity=2,
            k=5,
            time_budget=20.0,
            max_states=10000,
            strict_static_semantics=False,
        )

        manager.initialize(0.0)
        self.assertEqual(5, len(manager.routes))
        manager.handle_event(
            RequestArrival(Request("d", "1", "5", 1, 0.0, 10.0), event_time=0.0)
        )
        self.assertEqual(5, len(manager.routes))
        manager.handle_event(EdgeUpdate("e12", "1", "2", 0.0, 1.5))
        self.assertEqual(5, len(manager.routes))
        manager.handle_event(RequestCancel("d", event_time=0.0))
        self.assertEqual(5, len(manager.routes))
        self.assertEqual(
            5,
            len({candidate_identity(route) for route in manager.routes}),
        )

    def test_dynamic_top_k_uses_distinct_physical_paths_when_no_requests(self) -> None:
        graph = Graph(directed=False)
        for branch in range(1, 6):
            graph.add_edge("0", str(branch), float(branch))
            graph.add_edge(str(branch), "9", 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [],
            k=5,
            time_budget=20.0,
            topk_deviation_attempts=40,
            strict_static_semantics=False,
        )

        manager.initialize(0.0)

        self.assertEqual(5, len(manager.routes))
        self.assertEqual(
            5,
            len({tuple(route.full_path) for route in manager.routes}),
        )
        self.assertEqual(
            5,
            len(
                {
                    tuple(segment.edge_key for segment in record.segments)
                    for record in manager.route_records
                }
            ),
        )
        self.assertTrue(all(route.served_count == 0 for route in manager.routes))
        metrics = route_diversity_metrics(manager)
        self.assertEqual(1, metrics["unique_service_plans"])
        self.assertEqual(5, metrics["unique_physical_paths"])
        self.assertTrue(metrics["topk_complete"])

    def test_constraint_validator_rejects_corrupted_physical_path(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            k=1,
            time_budget=20.0,
        )
        manager.initialize(0.0)
        manager.routes[0].full_path = ["0", "3"]

        valid, error = validate_manager(manager)

        self.assertFalse(valid)
        self.assertIn("full path differs", error)

    def test_request_arrival_runs_dp_once_per_distinct_stop_plan(self) -> None:
        graph = Graph(directed=False)
        for branch in range(1, 6):
            graph.add_edge("0", str(branch), float(branch))
            graph.add_edge(str(branch), "9", 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [],
            capacity=2,
            k=5,
            time_budget=20.0,
            topk_deviation_attempts=40,
            strict_static_semantics=False,
        )
        manager.initialize(0.0)
        self.assertEqual(5, len(manager.routes))
        self.assertEqual(
            1,
            len(
                {
                    tuple(manager.remaining_route_events(route))
                    for route in manager.routes
                }
            ),
        )

        calls = 0
        original = manager.dp_insert_request_batch

        def counting_insert(*args, **kwargs):
            nonlocal calls
            calls += 1
            return original(*args, **kwargs)

        manager.dp_insert_request_batch = counting_insert  # type: ignore[method-assign]
        manager.handle_event(
            RequestArrival(
                Request("new", "1", "9", 1, request_time=0.0, wait_limit=10.0),
                event_time=0.0,
            )
        )

        self.assertEqual(1, calls)
        self.assertEqual(5, len(manager.routes))

    def test_request_arrival_deduplicates_converged_replacement_plans(self) -> None:
        requests = [
            Request(name, "1", "2", 1, 0.0, 20.0)
            for name in ("a", "b", "c")
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            requests,
            capacity=4,
            k=5,
            time_budget=30.0,
            replacement_limit_per_route=2,
            arrival_companion_limit=0,
            adaptive_recompute_on_failed_arrival=False,
        )

        def candidate(names):
            events = []
            for name in names:
                events.extend((("pickup", name), ("dropoff", name)))
            events.append(("target", ""))
            return Candidate(
                key_path=["0", "3"],
                full_path=["0", "1", "2", "3"],
                served_requests=list(names),
                pickup_info=[],
                total_cost=4.0,
                total_time=4.0,
                hops=3,
                served_count=len(names),
                score=len(names) * 100000.0 - 4.0,
                route_events=events,
            )

        first = candidate(["a", "c"])
        second = candidate(["b", "c"])
        manager.routes = [first, second]
        manager.route_archive = [first, second]
        manager.advance_vehicle = lambda event_time: None
        manager.repair_candidate_suffix = (
            lambda graph, route, planning_time: route
        )
        manager.rebuild_route_records = lambda planning_time: None

        attempted_plans = []

        def fake_insert(bases, request, planning_time):
            inserted = []
            for base in manager.route_plan_representatives(bases):
                remaining = tuple(manager.remaining_route_events(base))
                attempted_plans.append((request.name, remaining))
                names = [name for action, name in remaining if action == "pickup"]
                inserted.append(candidate([*names, request.name]))
            return inserted, len(inserted), 0, False

        manager.dp_insert_request_batch = fake_insert
        manager.handle_event(
            RequestArrival(
                Request("new", "1", "2", 1, request_time=0.0, wait_limit=20.0),
                event_time=0.0,
            )
        )

        self.assertEqual(5, len(attempted_plans))
        self.assertEqual(5, len(set(attempted_plans)))
        self.assertTrue(
            any("new" in route.served_requests for route in manager.route_archive)
        )

    def test_strict_dynamic_mode_uses_final_static_semantics(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            k=5,
            time_budget=20.0,
        )

        self.assertTrue(manager.strict_static_semantics)
        self.assertEqual("bundled", manager.planner_params["dp_mode"])
        self.assertTrue(manager.planner_params["pareto_only"])
        self.assertTrue(manager.planner_params["enforce_passenger_detour"])
        self.assertFalse(manager.planner_params["enable_compatibility_pruning"])

    def test_strict_dynamic_top_k_does_not_add_dominated_fillers(self) -> None:
        graph = Graph(directed=False)
        for branch in range(1, 6):
            graph.add_edge("0", str(branch), float(branch))
            graph.add_edge(str(branch), "9", 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [],
            k=5,
            time_budget=20.0,
            topk_deviation_attempts=40,
        )

        manager.initialize(0.0)

        self.assertEqual(1, len(manager.routes))
        self.assertEqual(["0", "1", "9"], manager.routes[0].full_path)
        self.assertEqual([], manager.backup_routes)
        self.assertEqual(1, len(manager.available_routes))
        self.assertGreater(manager.last_deviation_attempts, 0)

    def test_strict_dynamic_mode_adds_bounded_diverse_backup_routes(self) -> None:
        graph = Graph(directed=False)
        for branch in range(1, 6):
            graph.add_edge("0", str(branch), 1.0 + branch * 0.01)
            graph.add_edge(str(branch), "9", 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [],
            k=5,
            time_budget=20.0,
            backup_cost_regret_limit=0.05,
            backup_deviation_attempts=10,
        )

        result = manager.initialize(0.0)

        self.assertEqual(1, len(manager.routes))
        self.assertEqual(1, len(manager.route_archive))
        self.assertEqual(4, len(manager.backup_routes))
        self.assertEqual(5, len(manager.available_routes))
        self.assertTrue(result.available_topk_complete)
        self.assertLessEqual(result.max_backup_cost_regret, 0.05 + 1e-12)
        self.assertEqual(
            5,
            len(
                {
                    tuple(candidate.full_path)
                    for candidate in manager.available_routes
                }
            ),
        )
        archive_ids = {
            candidate_identity(candidate) for candidate in manager.route_archive
        }
        self.assertTrue(
            all(
                candidate_identity(candidate) not in archive_ids
                for candidate in manager.backup_routes
            )
        )
        self.assertEqual(5, len(manager.current_candidate_pool()))

    def test_strict_dynamic_mode_reuses_valid_concrete_backup_paths(self) -> None:
        graph = Graph(directed=False)
        for branch in range(1, 6):
            graph.add_edge("0", str(branch), 1.0 + branch * 0.01)
            graph.add_edge(str(branch), "9", 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [],
            k=5,
            time_budget=20.0,
            backup_cost_regret_limit=0.05,
            backup_deviation_attempts=10,
        )
        manager.initialize(0.0)
        expected_paths = {
            tuple(candidate.full_path) for candidate in manager.available_routes
        }
        manager.reset_event_diagnostics()

        manager.routes = manager.maintain_top_k_routes(
            manager.route_archive,
            0.0,
        )

        self.assertEqual(
            expected_paths,
            {tuple(candidate.full_path) for candidate in manager.available_routes},
        )
        self.assertEqual(5, len(manager.available_routes))
        self.assertEqual(0, manager._event_diagnostics.deviation_queries)
        self.assertEqual(0, manager._event_diagnostics.deviation_dijkstra_runs)

    def test_backup_event_orders_fill_sparse_graph_without_physical_detours(self) -> None:
        graph = Graph(directed=False)
        for node in range(6):
            graph.add_edge(str(node), str(node + 1), 1.0)
        requests = [
            Request("a", "1", "4", 1, 0.0, 20.0),
            Request("b", "2", "3", 1, 0.0, 20.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "6",
            requests,
            capacity=2,
            k=5,
            time_budget=30.0,
            detour_ratio=2.0,
            backup_cost_regret_limit=1.0,
            backup_deviation_attempts=0,
            backup_event_order_attempts=20,
        )

        result = manager.initialize(0.0)

        self.assertEqual(1, len(manager.routes))
        self.assertEqual(4, len(manager.backup_routes))
        self.assertEqual(5, len(manager.available_routes))
        self.assertTrue(result.available_topk_complete)
        event_orders = {
            tuple(candidate.route_events) for candidate in manager.available_routes
        }
        self.assertEqual(5, len(event_orders))
        self.assertTrue(all(validate_manager(manager)[0] for _route in event_orders))

    def test_preserved_backup_path_can_resume_inside_first_leg(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        graph.add_edge("2", "9", 1.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [],
            k=1,
            time_budget=20.0,
        )
        candidate = Candidate(
            key_path=["0", "9"],
            full_path=["0", "1", "2", "9"],
            served_requests=[],
            pickup_info=[],
            total_cost=3.0,
            total_time=3.0,
            hops=3,
            served_count=0,
            score=-3.0,
            route_events=[("target", "")],
        )
        manager.vehicle.node = "1"
        manager.vehicle.planning_time = 1.0

        repaired = manager.repair_candidate_suffix(
            manager.graph,
            candidate,
            1.0,
            preserve_physical_path=True,
        )

        self.assertIsNotNone(repaired)
        assert repaired is not None
        self.assertEqual(["1", "2", "9"], repaired.full_path)
        self.assertEqual(3.0, repaired.total_cost)
        self.assertEqual(0, manager._event_diagnostics.tdsp_queries)

    def test_current_candidate_pool_deduplicates_archive_and_backups(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            k=2,
            time_budget=20.0,
        )
        manager.initialize(0.0)
        route = manager.routes[0]
        manager.route_archive = [route]
        manager.backup_routes = [route]

        self.assertEqual([route], manager.current_candidate_pool())

    def test_suffix_repair_and_validator_enforce_beta_detour(self) -> None:
        graph = Graph(directed=False)
        for u, v, weight in (
            ("0", "1", 1.0),
            ("1", "2", 1.0),
            ("1", "3", 1.0),
            ("3", "2", 1.0),
            ("2", "4", 1.0),
        ):
            graph.add_edge(u, v, weight)
        requests = [
            Request("r", "1", "2", 1, 0.0, 10.0),
            Request("s", "3", "2", 1, 0.0, 10.0),
        ]
        candidate = Candidate(
            key_path=["0", "1", "3", "2", "2", "4"],
            full_path=["0", "1", "3", "2", "4"],
            served_requests=["r", "s"],
            pickup_info=[],
            total_cost=4.0,
            total_time=4.0,
            hops=4,
            served_count=2,
            score=199996.0,
            route_events=[
                ("pickup", "r"),
                ("pickup", "s"),
                ("dropoff", "r"),
                ("dropoff", "s"),
                ("target", ""),
            ],
        )
        strict = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "4",
            requests,
            capacity=2,
            k=5,
            detour_ratio=1.2,
            time_budget=20.0,
        )
        legacy = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "4",
            requests,
            capacity=2,
            k=5,
            detour_ratio=1.2,
            time_budget=20.0,
            strict_static_semantics=False,
        )

        self.assertIsNone(
            strict.repair_candidate_suffix(strict.graph, candidate, 0.0)
        )
        repaired = legacy.repair_candidate_suffix(legacy.graph, candidate, 0.0)
        self.assertIsNotNone(repaired)
        assert repaired is not None
        legacy.routes = [repaired]
        legacy.rebuild_route_records(0.0)
        legacy.planner_params["enforce_passenger_detour"] = True

        valid, error = validate_manager(legacy)

        self.assertFalse(valid)
        self.assertIn("violates beta detour for r", error)

    def test_insertion_label_resource_equality_requires_matching_pickup_times(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [Request("r", "1", "3", 1, 0.0, 10.0)],
            k=5,
        )

        def label(arrival_time: float, pickup_time: float, branch: str) -> InsertionLabel:
            return InsertionLabel(
                base_index=1,
                new_phase=1,
                node="2",
                arrival_time=arrival_time,
                load=1,
                onboard=frozenset({"r"}),
                key_path=("0", branch, "2"),
                full_path=("0", branch, "2"),
                route_events=(("pickup", "r"),),
                pickup_times={"r": pickup_time},
                served_names=(),
            )

        equal_cost_better_resource = label(5.0, 2.0, "a")
        equal_cost_tighter_resource = label(5.0, 1.0, "b")
        self.assertFalse(
            manager._same_insertion_resources(
                equal_cost_better_resource,
                equal_cost_tighter_resource,
            )
        )
        self.assertTrue(
            manager._same_insertion_resources(
                equal_cost_better_resource,
                label(5.0, 2.0, "c"),
            )
        )
        self.assertFalse(
            manager._same_insertion_resources(
                equal_cost_better_resource,
                label(5.0 + 1e-10, 2.0, "d"),
            )
        )

    def test_insertion_keeps_later_label_when_future_wait_equalizes_cost(self) -> None:
        graph = Graph(directed=False)
        for left, right in zip(range(5), range(1, 6)):
            graph.add_edge(str(left), str(right), 1.0)
        first = Request("a", "1", "2", 1, 0.0, 100.0)
        future = Request("b", "3", "4", 1, 100.0, 0.0)
        inserted = Request("n", "0", "1", 1, 0.0, 100.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0", "5", [first, future, inserted],
            capacity=2, k=20, detour_ratio=10.0,
            time_budget=110.0, max_states=10000,
        )
        base = Candidate(
            key_path=["0"], full_path=["0"], served_requests=[],
            pickup_info=[], total_cost=0.0, total_time=0.0,
            hops=0, served_count=0, score=0.0,
            route_events=[
                ("pickup", "a"), ("dropoff", "a"),
                ("pickup", "b"), ("dropoff", "b"),
            ],
        )

        candidates, _, _ = manager.dp_insert_request(
            manager.graph, base, inserted, 0.0
        )
        by_events = {tuple(candidate.route_events): candidate for candidate in candidates}
        early = (
            ("pickup", "n"), ("dropoff", "n"),
            ("pickup", "a"), ("dropoff", "a"),
            ("pickup", "b"), ("dropoff", "b"), ("target", ""),
        )
        late = (
            ("pickup", "a"), ("pickup", "n"),
            ("dropoff", "n"), ("dropoff", "a"),
            ("pickup", "b"), ("dropoff", "b"), ("target", ""),
        )
        self.assertIn(early, by_events)
        self.assertIn(late, by_events)
        self.assertAlmostEqual(by_events[early].total_cost, by_events[late].total_cost)
        self.assertNotEqual(
            candidate_identity(by_events[early]),
            candidate_identity(by_events[late]),
        )

    def test_strict_arrival_screening_uses_only_necessary_conditions(self) -> None:
        graph = Graph(directed=False)
        for u, v, weight in (
            ("0", "9", 1.0),
            ("0", "1", 1.0),
            ("1", "2", 1.0),
            ("2", "9", 1.0),
        ):
            graph.add_edge(u, v, weight)
        request = Request("r", "1", "2", 1, 0.0, 10.0)
        strict = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [request],
            detour_ratio=1.2,
            time_budget=10.0,
        )
        legacy = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(graph),
            "0",
            "9",
            [request],
            detour_ratio=1.2,
            time_budget=10.0,
            strict_static_semantics=False,
        )

        self.assertTrue(strict.is_request_compatible(request, 0.0))
        self.assertFalse(legacy.is_request_compatible(request, 0.0))

    def test_request_arrival_respects_wait_limit(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        manager.initialize(0.0)
        request = Request("late", "1", "2", 1, request_time=0.0, wait_limit=0.5)

        result = manager.handle_event(RequestArrival(request, event_time=0.0))

        self.assertEqual("IncrementalDPInsertion", result.update_mode)
        self.assertEqual(0, result.inserted_routes)
        self.assertNotIn("late", manager.routes[0].served_requests)

    def test_request_arrival_respects_current_vehicle_load(self) -> None:
        onboard_request = Request("r1", "1", "3", 2, request_time=0.0, wait_limit=5.0)
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [onboard_request],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        manager.initialize(0.0)
        new_request = Request("new", "2", "3", 1, request_time=3.0, wait_limit=1.0)

        result = manager.handle_event(RequestArrival(new_request, event_time=3.0))

        self.assertEqual(RequestStatus.ONBOARD, manager.request_status["r1"])
        self.assertEqual(0, result.inserted_routes)
        self.assertNotIn("new", manager.routes[0].served_requests)

    def test_event_during_committed_edge_does_not_move_planning_time_backwards(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=1,
            time_budget=20.0,
        )
        manager.initialize(0.0)
        request = Request("new", "2", "3", 1, request_time=1.0, wait_limit=10.0)
        first = manager.handle_event(RequestArrival(request, event_time=1.0))

        second = manager.handle_event(RequestCancel("unknown", event_time=1.5))

        self.assertEqual(2.0, first.planning_time)
        self.assertEqual(2.0, second.planning_time)
        self.assertTrue(second.in_transit)

    def test_time_dependent_shortest_path_uses_future_edge_weight(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "3", 1.0)
        graph.add_edge("0", "2", 2.0)
        graph.add_edge("2", "3", 2.0)
        dynamic_graph = TimeDependentGraph.from_static_graph(graph, bucket_size=1.0)
        dynamic_graph.update_edge_weight("1", "3", event_time=1.0, new_weight=10.0)

        path, travel_time = dynamic_graph.shortest_path("0", "3", depart_time=0.0)

        self.assertEqual(["0", "2", "3"], path)
        self.assertEqual(4.0, travel_time)

    def test_cloned_strategies_process_same_event_independently(self) -> None:
        base = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        base.initialize(0.0)
        request = Request("new", "1", "2", 1, request_time=0.0, wait_limit=5.0)
        event = RequestArrival(request, event_time=0.0)

        paired = run_paired_event(base, event)
        incremental = paired.incremental_manager
        full = paired.full_manager
        incremental_result = paired.incremental_result
        full_result = paired.full_result

        self.assertNotIn("new", base.active_requests)
        self.assertEqual("incremental", incremental_result.strategy)
        self.assertEqual("full_recompute", full_result.strategy)
        self.assertEqual("IncrementalDPInsertion", incremental_result.update_mode)
        self.assertEqual("FullRecomputeStrategy", full_result.update_mode)
        self.assertEqual(incremental_result.best_served, full_result.best_served)
        self.assertEqual(incremental_result.best_cost, full_result.best_cost)
        self.assertEqual(
            incremental.request_status["new"],
            full.request_status["new"],
        )

    def test_paired_commit_mode_uses_only_one_manager_clone(self) -> None:
        base = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        base.initialize(0.0)
        original_clone = base.clone
        clone_calls = 0

        def counted_clone(strategy=None):
            nonlocal clone_calls
            clone_calls += 1
            return original_clone(strategy)

        base.clone = counted_clone  # type: ignore[method-assign]
        request = Request("new", "1", "2", 1, request_time=0.0, wait_limit=5.0)

        paired = run_paired_event(
            base,
            RequestArrival(request, 0.0),
            commit_incremental=True,
        )

        self.assertEqual(1, clone_calls)
        self.assertIs(base, paired.incremental_manager)
        self.assertIn("new", base.active_requests)

    def test_paired_event_supports_full_first_counterbalancing(self) -> None:
        base = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=2,
            time_budget=20.0,
        )
        base.initialize(0.0)
        event = RequestArrival(
            Request("new", "1", "2", 1, request_time=0.0, wait_limit=5.0),
            event_time=0.0,
        )

        paired = run_paired_event(
            base,
            event,
            execution_order="full_first",
        )

        self.assertEqual("full_first", paired.execution_order)
        self.assertEqual(
            paired.incremental_result.best_served,
            paired.full_result.best_served,
        )

    def test_periodic_strategy_recomputes_at_configured_interval(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            strategy=UpdateStrategy.PERIODIC_RECOMPUTE,
            periodic_recompute_interval=2,
            k=2,
            time_budget=20.0,
        )
        manager.initialize(0.0)

        first = manager.handle_event(RequestCancel("unknown-1", 0.0))
        second = manager.handle_event(RequestCancel("unknown-2", 0.0))

        self.assertEqual("IncrementalCancellationRepair", first.update_mode)
        self.assertEqual("PeriodicFullRecompute", second.update_mode)
        self.assertEqual("periodic_recompute", second.strategy)

    def test_paired_event_preserves_periodic_online_strategy(self) -> None:
        base = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            strategy=UpdateStrategy.PERIODIC_RECOMPUTE,
            periodic_recompute_interval=1,
            k=2,
            time_budget=20.0,
        )
        base.initialize(0.0)

        paired = run_paired_event(
            base,
            RequestCancel("unknown", 0.0),
            commit_incremental=True,
            online_strategy=UpdateStrategy.PERIODIC_RECOMPUTE,
        )

        self.assertIs(base, paired.incremental_manager)
        self.assertEqual("periodic_recompute", paired.incremental_result.strategy)
        self.assertEqual(
            "PeriodicFullRecompute",
            paired.incremental_result.update_mode,
        )
        self.assertEqual(1, base.processed_event_count)

    def test_certified_arrival_returns_a_global_certificate(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [],
            capacity=2,
            k=2,
            time_budget=20.0,
            enable_certified_recovery=True,
        )
        initialization = manager.initialize(0.0)
        result = manager.handle_event(
            RequestArrival(
                Request("new", "1", "2", 1, 0.0, 5.0),
                0.0,
            )
        )

        self.assertTrue(initialization.objective_complete)
        self.assertTrue(initialization.concrete_complete)
        self.assertTrue(result.scope_complete)
        self.assertTrue(result.objective_complete)
        self.assertTrue(result.concrete_complete)

    def test_incomplete_local_scope_uses_certified_full_recompute(self) -> None:
        initial = [
            Request("a", "1", "2", 1, 0.0, 10.0),
            Request("b", "1", "2", 1, 0.0, 10.0),
        ]
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            initial,
            capacity=2,
            k=2,
            time_budget=20.0,
            arrival_local_reopt_request_limit=1,
            enable_certified_recovery=True,
        )
        manager.initialize(0.0)
        result = manager.handle_event(
            RequestArrival(
                Request("c", "1", "2", 1, 0.0, 10.0),
                0.0,
            )
        )

        self.assertEqual("CertifiedFullRecompute", result.update_mode)
        self.assertTrue(result.fallback_recompute)
        self.assertTrue(result.scope_complete)
        self.assertTrue(result.objective_complete)
        self.assertTrue(result.concrete_complete)

    def test_truncated_local_certificate_falls_back_for_each_event_type(self) -> None:
        events = (
            RequestArrival(Request("new", "1", "2", 1, 0.0, 10.0), 0.0),
            RequestCancel("a", 0.0),
            EdgeUpdate("e", "1", "2", 0.0, 3.0),
        )
        for event in events:
            with self.subTest(event=type(event).__name__):
                manager = DynamicKMORManager(
                    TimeDependentGraph.from_static_graph(self.make_graph()),
                    "0", "3", [Request("a", "1", "2", 1, 0.0, 10.0)],
                    capacity=2, k=2, time_budget=20.0,
                    max_states=5000, arrival_local_reopt_max_states=5000,
                    enable_certified_recovery=True,
                )
                self.assertTrue(manager.initialize(0.0).completed)
                real_local_search = manager.local_reoptimize_frontier

                def truncated_local_search(*args, **kwargs):
                    result = real_local_search(*args, **kwargs)
                    manager._last_local_search_certificate = SearchCertificate(
                        search_exhausted=False,
                        hit_state_limit=True,
                        scope_complete=True,
                        unresolved_count=1,
                        objective_complete=True,
                        concrete_complete=True,
                    )
                    return result

                with patch.object(
                    manager,
                    "local_reoptimize_frontier",
                    side_effect=truncated_local_search,
                ):
                    result = manager.handle_event(event)

                self.assertTrue(result.fallback_recompute)
                self.assertEqual("full", result.certificate_stage)
                self.assertTrue(result.completed)
                self.assertFalse(result.hit_max_states)

    def test_truncated_local_and_full_search_reports_incomplete(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0", "3", [Request("a", "1", "2", 1, 0.0, 10.0)],
            capacity=2, k=2, time_budget=20.0,
            max_states=5000, arrival_local_reopt_max_states=1,
            enable_certified_recovery=True,
        )
        self.assertTrue(manager.initialize(0.0).completed)
        manager.planner_params["max_states"] = 1

        result = manager.handle_event(EdgeUpdate("e", "1", "2", 0.0, 3.0))

        self.assertTrue(result.fallback_recompute)
        self.assertEqual("full", result.certificate_stage)
        self.assertTrue(result.hit_max_states)
        self.assertFalse(result.completed)

    def test_state_limited_full_recompute_is_not_certified_complete(self) -> None:
        manager = DynamicKMORManager(
            TimeDependentGraph.from_static_graph(self.make_graph()),
            "0",
            "3",
            [Request("a", "1", "2", 1, 0.0, 10.0)],
            capacity=2,
            k=2,
            time_budget=20.0,
            max_states=1,
            enable_certified_recovery=True,
        )
        result = manager.initialize(0.0)

        self.assertTrue(result.hit_max_states)
        self.assertFalse(
            result.objective_complete and result.concrete_complete
        )


if __name__ == "__main__":
    unittest.main()
