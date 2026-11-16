import random
import unittest

from Dpkmor import (
    Candidate,
    DPTransition,
    Graph,
    Request,
    RideSharePlanner,
    SearchState,
    candidate_identity,
    select_top_k_candidates,
)


class StaticKMORCorrectnessTests(unittest.TestCase):
    def make_detour_graph(self) -> Graph:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "4", 1.0)
        graph.add_edge("4", "5", 1.0)
        graph.add_edge("1", "2", 1.0)
        graph.add_edge("2", "3", 1.0)
        graph.add_edge("3", "4", 1.0)
        return graph

    def make_detour_requests(self):
        return [
            Request("r1", "1", "4", 1, request_time=0.0, wait_limit=1.1),
            Request("r2", "2", "3", 1, request_time=0.0, wait_limit=2.1),
        ]

    def solve_detour_case(self, **params):
        common = {
            "capacity": 2,
            "k": 5,
            "time_budget": 20.0,
            "detour_ratio": 1.5,
            "max_states": 10000,
            "pareto_only": True,
        }
        common.update(params)
        planner = RideSharePlanner(
            self.make_detour_graph(),
            "0",
            "5",
            self.make_detour_requests(),
            **common,
        )
        return planner, planner.solve()

    def test_passenger_detour_is_a_hard_constraint(self) -> None:
        _, constrained = self.solve_detour_case()
        _, unconstrained = self.solve_detour_case(enforce_passenger_detour=False)

        self.assertEqual(1, constrained[0].served_count)
        self.assertEqual(2, unconstrained[0].served_count)

    def test_no_pruning_still_enforces_passenger_detour(self) -> None:
        _, routes = self.solve_detour_case(enable_pruning=False)

        self.assertEqual(1, routes[0].served_count)

    def test_legacy_ablation_flag_enables_state_bounds(self) -> None:
        legacy = RideSharePlanner(
            self.make_detour_graph(),
            "0",
            "5",
            self.make_detour_requests(),
            aggressive_served_pruning=True,
        )
        explicit = RideSharePlanner(
            self.make_detour_graph(),
            "0",
            "5",
            self.make_detour_requests(),
            aggressive_served_pruning=True,
            enable_state_bounds=False,
        )

        self.assertTrue(legacy.enable_state_bounds)
        self.assertFalse(explicit.enable_state_bounds)

    def test_state_bound_pruning_switch_and_counter(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("S", "P", 1.0)
        graph.add_edge("P", "T", 1.0)
        graph.add_edge("P", "D", 3.0)
        graph.add_edge("D", "T", 3.0)
        request = Request("r1", "P", "D", 1, request_time=0.0, wait_limit=10.0)
        common = {
            "capacity": 1,
            "k": 2,
            "time_budget": 8.0,
            "detour_ratio": 3.0,
            "enable_compatibility_pruning": False,
        }
        bounded = RideSharePlanner(
            graph,
            "S",
            "T",
            [request],
            enable_pruning=True,
            enable_state_bounds=True,
            **common,
        )
        unbounded = RideSharePlanner(
            graph,
            "S",
            "T",
            [request],
            enable_pruning=False,
            enable_state_bounds=True,
            **common,
        )
        late_state = SearchState(
            node="S",
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=3.0,
            key_path=("S",),
            pickup_times=(-1.0,),
        )

        bounded_transitions = bounded._build_dp_transitions(late_state, [])
        unbounded_transitions = unbounded._build_dp_transitions(late_state, [])

        self.assertNotIn(
            "pickup",
            [transition.successor.route_events[-1][0] for transition in bounded_transitions],
        )
        self.assertIn(
            "pickup",
            [transition.successor.route_events[-1][0] for transition in unbounded_transitions],
        )
        self.assertEqual(1, bounded.stats.state_bound_pruned_states)
        self.assertEqual(1, bounded.stats.heuristic_pruned_states)
        self.assertEqual(0, bounded.stats.pareto_bound_pruned_states)
        self.assertEqual(0, unbounded.stats.state_bound_pruned_states)

    def test_target_node_is_not_complete_with_onboard_passenger(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        request = Request("r1", "1", "2", 1, request_time=0.0, wait_limit=2.0)
        planner = RideSharePlanner(
            graph,
            "0",
            "1",
            [request],
            capacity=1,
            k=2,
            time_budget=10.0,
            detour_ratio=1.0,
            pareto_only=True,
        )

        routes = planner.solve()

        served_route = next(route for route in routes if route.served_count == 1)
        self.assertIn(("pickup", "r1"), served_route.route_events)
        self.assertIn(("dropoff", "r1"), served_route.route_events)
        self.assertEqual(("target", ""), served_route.route_events[-1])

    def test_dp_dominance_keeps_incomparable_onboard_slack(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 10.0)
        request = Request("r1", "0", "1", 1, request_time=0.0, wait_limit=10.0)
        planner = RideSharePlanner(
            graph,
            "0",
            "1",
            [request],
            detour_ratio=2.0,
            k=2,
        )
        early_but_used_slack = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=10.0,
            key_path=("0",),
            pickup_times=(0.0,),
        )
        later_with_slack = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=11.0,
            key_path=("0", "0"),
            pickup_times=(9.0,),
        )

        self.assertFalse(planner._state_dominates(early_but_used_slack, later_with_slack))
        self.assertFalse(planner._state_dominates(later_with_slack, early_but_used_slack))

        frontier = {}
        self.assertTrue(planner._should_enqueue(early_but_used_slack, frontier))
        self.assertTrue(planner._should_enqueue(later_with_slack, frontier))
        self.assertEqual(2, len(frontier[planner._state_signature(early_but_used_slack)]))

    def test_dp_dominance_is_safe_under_future_waiting(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 10.0)
        request = Request("r1", "0", "1", 1, request_time=0.0, wait_limit=10.0)
        planner = RideSharePlanner(graph, "0", "1", [request], detour_ratio=2.0)
        earlier_but_earlier_pickup = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=10.0,
            key_path=("0",),
            pickup_times=(1.0,),
        )
        later_but_later_pickup = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=11.0,
            key_path=("0", "0"),
            pickup_times=(2.0,),
        )

        # Both have ridden for nine units now, but a future common wait would
        # leave the first passenger with the earlier absolute dropoff deadline.
        self.assertFalse(
            planner._state_dominates(
                earlier_but_earlier_pickup,
                later_but_later_pickup,
            )
        )

    def test_dp_keeps_k_dominated_histories_for_top_k_routes(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        requests = [
            Request("r1", "0", "1", 1, 0.0, 10.0),
            Request("r2", "0", "1", 1, 100.0, 10.0),
        ]
        planner = RideSharePlanner(graph, "0", "1", requests, k=2)
        better = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=10.0,
            key_path=("0",),
            pickup_times=(10.0, -1.0),
        )
        worse_but_distinct = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=11.0,
            key_path=("0", "0"),
            pickup_times=(9.0, -1.0),
        )

        self.assertTrue(planner._state_dominates(better, worse_but_distinct))
        frontier = {}
        self.assertTrue(planner._should_enqueue(better, frontier))
        self.assertTrue(planner._should_enqueue(worse_but_distinct, frontier))
        self.assertEqual(2, len(frontier[planner._state_signature(better)]))

    def test_dp_prunes_strict_cost_dominance_after_all_releases(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        request = Request("r1", "0", "1", 1, 0.0, 10.0)
        planner = RideSharePlanner(graph, "0", "1", [request], k=5)
        better = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=10.0,
            key_path=("0",),
            pickup_times=(10.0,),
        )
        worse = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=11.0,
            key_path=("0", "0"),
            pickup_times=(9.0,),
        )

        frontier = {}
        self.assertTrue(planner._should_enqueue(better, frontier))
        self.assertFalse(planner._should_enqueue(worse, frontier))
        self.assertEqual(1, len(frontier[planner._state_signature(better)]))

    def test_static_pareto_selection_does_not_fill_with_dominated_routes(self) -> None:
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

        routes = select_top_k_candidates(candidates, 5, pareto_only=True)

        self.assertEqual(1, len(routes))
        self.assertEqual(10.0, routes[0].total_cost)

    def test_pareto_selection_tolerates_roundoff_noise(self) -> None:
        better_coverage = Candidate(
            key_path=["0", "1"],
            full_path=["0", "1"],
            served_requests=["r1", "r2"],
            pickup_info=[],
            total_cost=10.0 + 5e-10,
            total_time=10.0 + 5e-10,
            hops=1,
            served_count=2,
            score=0.0,
            route_events=[("target", "")],
        )
        lower_coverage = Candidate(
            key_path=["0", "2", "1"],
            full_path=["0", "2", "1"],
            served_requests=["r1"],
            pickup_info=[],
            total_cost=10.0,
            total_time=10.0,
            hops=2,
            served_count=1,
            score=0.0,
            route_events=[("pickup", "r1"), ("target", "")],
        )

        routes = select_top_k_candidates(
            [better_coverage, lower_coverage],
            2,
            pareto_only=True,
        )

        self.assertEqual([better_coverage], routes)

    def test_top_k_order_uses_tolerance_before_route_tie_breaking(self) -> None:
        lexicographically_first = Candidate(
            key_path=["0", "1"],
            full_path=["0", "1"],
            served_requests=["r1"],
            pickup_info=[],
            total_cost=10.0 + 5e-10,
            total_time=10.0 + 5e-10,
            hops=1,
            served_count=1,
            score=0.0,
            route_events=[("pickup", "a"), ("target", "")],
        )
        numerically_first = Candidate(
            key_path=["0", "2"],
            full_path=["0", "2"],
            served_requests=["r2"],
            pickup_info=[],
            total_cost=10.0,
            total_time=10.0,
            hops=1,
            served_count=1,
            score=0.0,
            route_events=[("pickup", "b"), ("target", "")],
        )

        routes = select_top_k_candidates(
            [numerically_first, lexicographically_first],
            2,
            pareto_only=True,
            numeric_tolerance=1e-9,
        )

        self.assertEqual(lexicographically_first, routes[0])

    def test_dp_merges_equal_resource_states_without_losing_top_k_routes(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        requests = [
            Request("r1", "1", "2", 1, 0.0, 10.0),
            Request("r2", "1", "2", 1, 0.0, 10.0),
        ]
        common = {
            "capacity": 2,
            "k": 10,
            "time_budget": 10.0,
            "detour_ratio": 2.0,
            "pareto_only": True,
        }
        dpkmor = RideSharePlanner(
            graph,
            "0",
            "2",
            requests,
            enable_dp=True,
            dp_mode="bundled",
            **common,
        )
        kprun = RideSharePlanner(
            graph,
            "0",
            "2",
            requests,
            enable_dp=False,
            **common,
        )

        dpkmor_routes = dpkmor.solve()
        kprun_routes = kprun.solve()
        route_key = lambda route: (
            route.served_count,
            round(route.total_cost, 9),
            tuple(route.route_events),
            tuple(route.key_path),
        )

        self.assertEqual(
            [route_key(route) for route in kprun_routes],
            [route_key(route) for route in dpkmor_routes],
        )
        self.assertLess(dpkmor.stats.expanded_states, kprun.stats.expanded_states)
        self.assertLess(dpkmor.stats.feasibility_checks, kprun.stats.feasibility_checks)

    def test_bundled_dag_reuses_one_resource_node_and_recovers_both_histories(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        requests = [
            Request("r1", "1", "2", 1, 0.0, 10.0),
            Request("r2", "1", "2", 1, 0.0, 10.0),
        ]
        planner = RideSharePlanner(
            graph,
            "0",
            "2",
            requests,
            capacity=2,
            k=5,
            time_budget=10.0,
            detour_ratio=2.0,
            dp_mode="bundled",
        )

        initial = SearchState(
            node="0",
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=0.0,
            key_path=("0",),
            pickup_times=(-1.0, -1.0),
            route_events=(),
        )
        pickup_r1 = SearchState(
            node="1",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=1.0,
            key_path=("0", "1"),
            pickup_times=(1.0, -1.0),
            route_events=(("pickup", "r1"),),
        )
        pickup_r2 = SearchState(
            node="1",
            picked_mask=2,
            served_mask=0,
            load=1,
            time_used=1.0,
            key_path=("0", "1"),
            pickup_times=(-1.0, 1.0),
            route_events=(("pickup", "r2"),),
        )
        both_after_r1 = SearchState(
            node="1",
            picked_mask=3,
            served_mask=0,
            load=2,
            time_used=1.0,
            key_path=("0", "1", "1"),
            pickup_times=(1.0, 1.0),
            route_events=(("pickup", "r1"), ("pickup", "r2")),
        )
        both_after_r2 = SearchState(
            node="1",
            picked_mask=3,
            served_mask=0,
            load=2,
            time_used=1.0,
            key_path=("0", "1", "1"),
            pickup_times=(1.0, 1.0),
            route_events=(("pickup", "r2"), ("pickup", "r1")),
        )

        planner._history_from_state(initial)
        queue = []
        frontier = {}
        counter = planner._enqueue_dp_dag(
            queue, 0, frontier, initial, None, None
        )
        initial_node = planner._bundle_resource_index[
            planner._bundle_resource_key(initial)
        ]

        transition_r1 = DPTransition(pickup_r1, 0, 1)
        transition_r2 = DPTransition(pickup_r2, 1, 1)
        counter = planner._enqueue_dp_dag(
            queue, counter, frontier, pickup_r1, initial_node, transition_r1
        )
        counter = planner._enqueue_dp_dag(
            queue, counter, frontier, pickup_r2, initial_node, transition_r2
        )
        parent_r1 = planner._bundle_resource_index[
            planner._bundle_resource_key(pickup_r1)
        ]
        parent_r2 = planner._bundle_resource_index[
            planner._bundle_resource_key(pickup_r2)
        ]

        counter = planner._enqueue_dp_dag(
            queue,
            counter,
            frontier,
            both_after_r1,
            parent_r1,
            DPTransition(both_after_r1, 1, 0),
        )
        generated_before_merge = planner.stats.generated_states
        counter = planner._enqueue_dp_dag(
            queue,
            counter,
            frontier,
            both_after_r2,
            parent_r2,
            DPTransition(both_after_r2, 0, 0),
        )

        merged = planner._bundle_resource_index[
            planner._bundle_resource_key(both_after_r1)
        ]
        histories = planner._dag_top_histories(merged, initial_node, {})
        event_orders = {
            planner._materialize_history(history_id)[2]
            for history_id in histories
        }

        self.assertEqual(generated_before_merge, planner.stats.generated_states)
        self.assertEqual(2, len(merged.incoming))
        self.assertEqual(
            {
                (("pickup", "r1"), ("pickup", "r2")),
                (("pickup", "r2"), ("pickup", "r1")),
            },
            event_orders,
        )

    def test_lazy_recovery_top_k_ids_are_distinct_concrete_routes(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("s", "p", 1.0)
        graph.add_edge("p", "d", 1.0)
        graph.add_edge("d", "t", 1.0)
        requests = [Request(f"r{index}", "p", "d", 1, 0.0, 20.0) for index in range(3)]
        options = dict(
            capacity=3,
            k=2,
            time_budget=10.0,
            detour_ratio=5.0,
            max_states=100000,
            dp_mode="bundled",
        )
        planner = RideSharePlanner(graph, "s", "t", requests, enable_dp=True, **options)
        original_recover = planner._dag_top_histories
        terminal_checks = []

        def all_histories(node, initial_node, memo):
            key = id(node)
            if key not in memo:
                if node is initial_node:
                    memo[key] = {0}
                else:
                    memo[key] = {
                        planner._extend_history(history_id, transition)
                        for parent, transition in node.incoming
                        for history_id in all_histories(parent, initial_node, memo)
                    }
            return memo[key]

        def checked_recover(node, initial_node, memo):
            retained = original_recover(node, initial_node, memo)
            if node.state.node == "t" and node.state.load == 0:
                all_ids = all_histories(node, initial_node, {})
                identities = [
                    (
                        planner._materialize_history(history_id)[2],
                        planner._materialize_history(history_id)[0],
                    )
                    for history_id in all_ids
                ]
                self.assertEqual(len(identities), len(set(identities)))
                expected = tuple(sorted(all_ids, key=planner._history_rank)[: planner.k])
                self.assertEqual(expected, retained)
                terminal_checks.append(len(all_ids))
            return retained

        planner._dag_top_histories = checked_recover
        bundled = planner.solve()
        kprun = RideSharePlanner(graph, "s", "t", requests, enable_dp=False, **options).solve()

        self.assertTrue(any(count > planner.k for count in terminal_checks))
        self.assertEqual(
            [candidate_identity(route) for route in kprun],
            [candidate_identity(route) for route in bundled],
        )

    def test_dpkmor_and_kprun_match_on_small_instance(self) -> None:
        _, dpkmor = self.solve_detour_case(enable_dp=True)
        _, kprun = self.solve_detour_case(enable_dp=False)

        dpkmor_objectives = {
            (route.served_count, round(route.total_cost, 9)) for route in dpkmor
        }
        kprun_objectives = {
            (route.served_count, round(route.total_cost, 9)) for route in kprun
        }
        self.assertEqual(kprun_objectives, dpkmor_objectives)

    def test_optimistic_coverage_counts_onboard_requests(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        request = Request("r1", "0", "1", 1, 0.0, 1.0)
        planner = RideSharePlanner(graph, "0", "1", [request])
        state = SearchState(
            node="0",
            picked_mask=1,
            served_mask=0,
            load=1,
            time_used=0.0,
            key_path=("0",),
            pickup_times=(0.0,),
        )

        self.assertEqual(1, planner._best_possible_served(state))

    def test_optimistic_coverage_excludes_expired_unpicked_request(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        request = Request("r1", "0", "1", 1, 0.0, 1.0)
        planner = RideSharePlanner(
            graph,
            "0",
            "1",
            [request],
            time_budget=10.0,
        )
        expired_state = SearchState(
            node="1",
            picked_mask=0,
            served_mask=0,
            load=0,
            time_used=2.0,
            key_path=("0", "1"),
            pickup_times=(-1.0,),
        )

        self.assertEqual(0, planner._best_possible_served(expired_state))

    def test_bundled_incumbents_preserve_concrete_top_k(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("0", "1", 1.0)
        graph.add_edge("1", "2", 1.0)
        graph.add_edge("2", "3", 1.0)
        requests = [
            Request(name, "1", "2", 1, 0.0, 10.0)
            for name in ("a", "b", "c")
        ]
        common = {
            "capacity": 2,
            "k": 5,
            "time_budget": 20.0,
            "detour_ratio": 2.0,
            "max_states": 10000,
            "pareto_only": True,
            "dp_mode": "bundled",
        }
        baseline = RideSharePlanner(
            graph,
            "0",
            "3",
            requests,
            **common,
        ).solve()
        seeded = RideSharePlanner(
            graph,
            "0",
            "3",
            requests,
            initial_candidates=baseline[:2],
            **common,
        ).solve()
        bounded = RideSharePlanner(
            graph,
            "0",
            "3",
            requests,
            initial_candidates=baseline,
            enable_topk_incumbent_bound=True,
            **common,
        ).solve()

        self.assertEqual(5, len(baseline))
        self.assertEqual(
            {candidate_identity(candidate) for candidate in baseline},
            {candidate_identity(candidate) for candidate in seeded},
        )
        self.assertEqual(
            {candidate_identity(candidate) for candidate in baseline},
            {candidate_identity(candidate) for candidate in bounded},
        )

    def test_exact_kprun_and_dpkmor_match_on_random_small_instances(self) -> None:
        graph = Graph(directed=False)
        for node in range(5):
            graph.add_edge(str(node), str(node + 1), 1.0)
        graph.add_edge("0", "2", 1.5)
        graph.add_edge("2", "4", 1.5)
        graph.add_edge("1", "4", 2.2)
        graph.add_edge("3", "5", 1.4)

        rng = random.Random(20260711)
        for case_index in range(12):
            requests = []
            for request_index in range(4):
                pickup = rng.randint(0, 4)
                dropoff = rng.randint(pickup + 1, 5)
                requests.append(
                    Request(
                        f"c{case_index}_r{request_index}",
                        str(pickup),
                        str(dropoff),
                        1,
                        request_time=float(rng.randint(0, 2)),
                        wait_limit=float(rng.randint(2, 7)),
                    )
                )

            common = {
                "capacity": 2,
                "k": 20,
                "time_budget": 15.0,
                "detour_ratio": [1.0, 1.3, 1.8][case_index % 3],
                "max_states": 100000,
                "pareto_only": True,
                "enable_state_bounds": False,
                "aggressive_served_pruning": False,
            }

            objective_frontiers = []
            for enable_pruning, enable_dp in [
                (False, False),
                (True, False),
                (True, True),
            ]:
                planner = RideSharePlanner(
                    graph,
                    "0",
                    "5",
                    requests,
                    enable_pruning=enable_pruning,
                    enable_dp=enable_dp,
                    **common,
                )
                routes = planner.solve()
                self.assertLess(planner.stats.expanded_states, planner.max_states)
                # A route-level top-k can be filled by several physically
                # different routes sharing one objective pair. Correctness of
                # the DP merge is therefore checked against the complete
                # nondominated objective frontier, not only the first k route
                # identities returned to the caller.
                objective_frontiers.append(
                    {
                        (route.served_count, round(route.total_cost, 9))
                        for route in planner.last_complete_frontier
                    }
                )

            self.assertEqual(objective_frontiers[0], objective_frontiers[1])
            self.assertEqual(objective_frontiers[0], objective_frontiers[2])

    def test_objective_equivalence_does_not_certify_concrete_plans(self) -> None:
        graph = Graph(directed=False)
        graph.add_edge("S", "P", 1.0)
        graph.add_edge("P", "D", 1.0)
        graph.add_edge("D", "T", 1.0)
        request = Request("r", "P", "D", 1, 0.0, 10.0)
        planner = RideSharePlanner(
            graph,
            "S",
            "T",
            [request],
            k=5,
            time_budget=20.0,
            dp_mode="bundled",
        )
        unresolved = SearchState(
            node="T",
            picked_mask=1,
            served_mask=1,
            load=0,
            time_used=3.0,
            key_path=("T",),
            pickup_times=(1.0,),
        )
        equivalent = Candidate(
            key_path=["S", "P", "D", "T"],
            full_path=["S", "P", "D", "T"],
            served_requests=["r"],
            pickup_info=[],
            total_cost=3.0,
            total_time=3.0,
            hops=3,
            served_count=1,
            score=99997.0,
            route_events=[("pickup", "r"), ("dropoff", "r"), ("target", "")],
        )

        certificate = planner._build_search_certificate(
            [unresolved],
            [equivalent],
        )

        self.assertTrue(certificate.objective_complete)
        self.assertFalse(certificate.concrete_complete)
        self.assertFalse(certificate.search_exhausted)

    def test_incomplete_scope_invalidates_both_certificates(self) -> None:
        _, routes = self.solve_detour_case(dp_mode="bundled")
        planner, _ = self.solve_detour_case(dp_mode="bundled")
        certificate = planner.search_certificate.for_scope(False)

        self.assertTrue(routes)
        self.assertFalse(certificate.scope_complete)
        self.assertFalse(certificate.objective_complete)
        self.assertFalse(certificate.concrete_complete)

    def test_state_limited_search_is_not_reported_as_completed(self) -> None:
        planner, _ = self.solve_detour_case(
            dp_mode="bundled",
            max_states=1,
        )

        self.assertTrue(planner.stats.hit_max_states)
        self.assertTrue(planner.search_certificate.hit_state_limit)
        self.assertGreater(planner.search_certificate.unresolved_count, 0)
        self.assertFalse(planner.search_certificate.completed)


if __name__ == "__main__":
    unittest.main()
