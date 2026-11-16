"""Rolling-horizon local-search baseline adapted to plan-level Top-k KMOR.

The baseline starts from the same exact initial archive as the formal dynamic
method.  After initialization it only repairs retained plans and explores
their local service-event neighborhoods.  It deliberately does not invoke a
completeness certificate, local-completion search, or full recomputation.
"""

from __future__ import annotations

import time
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Union

from Dpkmor import Candidate, Request, candidate_identity
from dynamic_kmor import (
    DynamicEvent,
    DynamicKMORManager,
    DynamicUpdateResult,
    EdgeUpdate,
    RequestArrival,
    RequestCancel,
    RequestStatus,
    TimeDependentGraph,
    UpdateStrategy,
)
from dynamic_kinetic_tree import KineticTreeKMORManager, Plan


class RHLSTopKManager(KineticTreeKMORManager):
    """Maintain Top-k plans with event-driven rolling-horizon local search.

    The candidate pool is intentionally wider than the returned Top-k.  The
    two safety limits are audited rather than used as an unreported runtime
    shortcut.
    """

    def __init__(
        self,
        graph: TimeDependentGraph,
        source: str,
        target: str,
        initial_requests: Sequence[Request],
        **planner_params: object,
    ):
        output_k = max(1, int(planner_params.get("k", 5)))
        self.rhls_pool_limit = max(
            output_k,
            int(planner_params.pop("rhls_pool_limit", max(500, output_k * 100))),
        )
        self.rhls_max_rounds = max(
            1,
            int(planner_params.pop("rhls_max_rounds", 6)),
        )
        self.rhls_max_generated = max(
            self.rhls_pool_limit,
            int(planner_params.pop("rhls_max_generated", 100000)),
        )
        planner_params["kinetic_tree_max_schedules"] = self.rhls_pool_limit
        planner_params["kinetic_tree_max_generated"] = self.rhls_max_generated
        super().__init__(graph, source, target, initial_requests, **planner_params)
        self.rhls_last_iterations = 0
        self.rhls_last_moves = 0
        self.rhls_last_peak_pool = 0
        self.rhls_iteration_capped = False

    def _schedule_maintenance_enabled(self) -> bool:
        return self.strategy == UpdateStrategy.RHLS_TOPK

    def _baseline_update_mode(self) -> str:
        return "RHLSTopK"

    def _initialization_note(self) -> str:
        return "shared exact initial archive"

    @staticmethod
    def _insertions(plan: Plan, request_name: str) -> Iterable[Plan]:
        size = len(plan)
        for pickup_index in range(size + 1):
            with_pickup = (
                plan[:pickup_index]
                + (("pickup", request_name),)
                + plan[pickup_index:]
            )
            for dropoff_index in range(pickup_index + 1, size + 2):
                yield (
                    with_pickup[:dropoff_index]
                    + (("dropoff", request_name),)
                    + with_pickup[dropoff_index:]
                )

    def _waiting_names_in_plan(self, plan: Plan) -> List[str]:
        pickups = {
            name
            for action, name in plan
            if action == "pickup"
            and self.request_status.get(name) == RequestStatus.WAITING
        }
        dropoffs = {name for action, name in plan if action == "dropoff"}
        return sorted(pickups & dropoffs)

    def _unserved_waiting_names(self, plan: Plan) -> List[str]:
        planned = {name for _, name in plan}
        return sorted(
            name
            for name in self.active_requests
            if self.request_status.get(name) == RequestStatus.WAITING
            and name not in planned
        )

    @staticmethod
    def _remove_pair(plan: Plan, request_name: str) -> Plan:
        return tuple(event for event in plan if event[1] != request_name)

    def _neighbor_plans(
        self,
        seeds: Sequence[Candidate],
        seen: Set[Plan],
        remaining_limit: int,
    ) -> List[Plan]:
        """Generate insertion, relocation, reinsertion, and exchange moves."""
        generated: List[Plan] = []

        def retain(plan: Plan) -> bool:
            if plan in seen:
                return True
            seen.add(plan)
            generated.append(plan)
            self.rhls_last_moves += 1
            return len(generated) < remaining_limit

        plans = sorted(
            {
                tuple(self.remaining_route_events(candidate))
                for candidate in seeds
            }
        )
        for plan in plans:
            served_waiting = self._waiting_names_in_plan(plan)
            unserved_waiting = self._unserved_waiting_names(plan)

            # Insert currently unserved requests into every legal pair of
            # positions. This includes the new request after an arrival.
            for request_name in unserved_waiting:
                for neighbor in self._insertions(plan, request_name):
                    if not retain(neighbor):
                        return generated

            # Relocate/reinsert one waiting pickup-delivery pair. Moving the
            # pair also covers local precedence-safe event-order adjustments.
            for request_name in served_waiting:
                reduced = self._remove_pair(plan, request_name)
                for neighbor in self._insertions(reduced, request_name):
                    if neighbor == plan:
                        continue
                    if not retain(neighbor):
                        return generated

            # Direct exchanges allow the search to cross an intermediate
            # lower-coverage plan without requiring that plan to be retained.
            for removed_name in served_waiting:
                reduced = self._remove_pair(plan, removed_name)
                for inserted_name in unserved_waiting:
                    for neighbor in self._insertions(reduced, inserted_name):
                        if not retain(neighbor):
                            return generated
        return generated

    def _run_local_search(self, planning_time: float) -> Tuple[int, int, int]:
        self.rhls_last_iterations = 0
        self.rhls_last_moves = 0
        self.rhls_last_peak_pool = len(self.kinetic_schedule_pool)
        self.rhls_iteration_capped = False
        seen: Set[Plan] = {
            tuple(self.remaining_route_events(candidate))
            for candidate in self.kinetic_schedule_pool
        }
        feasible_total = 0
        evicted_total = 0
        expanded_total = 0

        for round_index in range(self.rhls_max_rounds):
            remaining = self.rhls_max_generated - self.rhls_last_moves
            if remaining <= 0:
                self.kinetic_generation_capped = True
                break
            before_ids = {
                candidate_identity(candidate)
                for candidate in self.kinetic_schedule_pool
            }
            neighbors = self._neighbor_plans(
                self.kinetic_schedule_pool,
                seen,
                remaining,
            )
            if not neighbors:
                break
            self.rhls_last_iterations = round_index + 1
            if self.rhls_last_moves >= self.rhls_max_generated:
                self.kinetic_generation_capped = True

            candidates = self._evaluate_plans(neighbors, planning_time)
            feasible_total += len(candidates)
            expanded_total += getattr(self, "_kinetic_last_expanded", 0)
            combined = [*self.kinetic_schedule_pool, *candidates]
            self._retain_schedule_pool(combined)
            evicted_total += self.kinetic_last_evicted
            self.rhls_last_peak_pool = max(
                self.rhls_last_peak_pool,
                len(combined),
                len(self.kinetic_schedule_pool),
            )
            after_ids = {
                candidate_identity(candidate)
                for candidate in self.kinetic_schedule_pool
            }
            if not (after_ids - before_ids):
                break
        else:
            self.rhls_iteration_capped = True

        return feasible_total, evicted_total, expanded_total

    def _result(self, **kwargs: object) -> DynamicUpdateResult:
        result = super()._result(**kwargs)
        result.update_mode = "RHLSTopK"
        result.peak_open_size = max(result.peak_open_size, self.rhls_last_peak_pool)
        result.note = result.note.replace("kt_pool=", "rhls_pool=")
        result.note += (
            f";rhls_iterations={self.rhls_last_iterations}"
            f";rhls_moves={self.rhls_last_moves}"
            f";rhls_peak_pool={self.rhls_last_peak_pool}"
            f";iteration_capped={self.rhls_iteration_capped}"
        )
        result.local_search_iterations = self.rhls_last_iterations
        result.local_search_moves = self.rhls_last_moves
        result.candidate_pool_peak_size = self.rhls_last_peak_pool
        result.local_search_iteration_capped = self.rhls_iteration_capped
        return result

    def initialize(self, current_time: float = 0.0) -> DynamicUpdateResult:
        if not self._schedule_maintenance_enabled():
            return super().initialize(current_time)

        # Initialization is outside per-event latency and is the only point at
        # which RHLS may use the complete DPKMOR solver.
        started = time.perf_counter()
        DynamicKMORManager.initialize(self, current_time)
        initial_pool = self.current_candidate_pool()
        self.kinetic_last_generated = 0
        self.kinetic_last_feasible = len(initial_pool)
        self.kinetic_generation_capped = False
        self._retain_schedule_pool(initial_pool)
        self.rhls_last_iterations = 0
        self.rhls_last_moves = 0
        self.rhls_last_peak_pool = len(self.kinetic_schedule_pool)
        self.rhls_iteration_capped = False
        result = self._result(
            event_type="Initialize",
            event_time=current_time,
            planning_time=current_time,
            update_start=started,
            note=self._initialization_note(),
        )
        return self.attach_event_diagnostics(result)

    def handle_event(self, event: DynamicEvent) -> DynamicUpdateResult:
        if not self._schedule_maintenance_enabled():
            return super().handle_event(event)

        update_start = time.perf_counter()
        retained_before = len(self.kinetic_schedule_pool)
        previous_ids = {
            candidate_identity(candidate)
            for candidate in self.kinetic_schedule_pool
        }

        repaired = super().handle_event(event)
        base_generated = repaired.generated_schedules
        base_feasible = repaired.feasible_schedules
        base_evicted = repaired.evicted_schedules
        base_expanded = repaired.expanded_states
        base_capped = repaired.schedule_generation_capped

        planning_time = max(event.event_time, self.vehicle.planning_time)
        local_feasible, local_evicted, local_expanded = self._run_local_search(
            planning_time
        )
        self.kinetic_last_generated = base_generated + self.rhls_last_moves
        self.kinetic_last_feasible = base_feasible + local_feasible
        self.kinetic_last_evicted = base_evicted + local_evicted
        self.kinetic_generation_capped = (
            base_capped or self.kinetic_generation_capped
        )
        self._kinetic_last_expanded = base_expanded + local_expanded

        final_ids = {
            candidate_identity(candidate)
            for candidate in self.kinetic_schedule_pool
        }
        note = ""
        if isinstance(event, EdgeUpdate):
            note = event.edge_id
        elif isinstance(event, RequestArrival):
            note = event.request.name
        elif isinstance(event, RequestCancel):
            note = event.request_name

        result = self._result(
            event_type=repaired.event_type,
            event_time=event.event_time,
            planning_time=planning_time,
            update_start=update_start,
            affected_routes=repaired.affected_routes,
            inserted_routes=len(final_ids - previous_ids),
            removed_stops=repaired.removed_stops,
            note=note,
        )
        return self.attach_event_context(result, retained_before)
