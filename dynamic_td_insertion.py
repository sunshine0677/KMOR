"""Time-dependent single-route insertion baseline for dynamic KMOR.

This is an independent adaptation of the insertion policy studied by Gong,
Zeng, and Chen (PVLDB 2024).  The original operator handles the arrival of one
request on one time-dependent route.  Here arrival keeps the relative order of
all committed stops and chooses the feasible pickup/drop-off insertion with
minimum route cost.  Small adapters remove cancelled stops and revalidate the
same stop order after an edge update.  The method intentionally maintains one
executable route; it is not a plan-level Top-k algorithm and is therefore used
as an effectiveness/latency baseline rather than an exactness oracle.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import List, Optional, Sequence, Tuple, Union

from Dpkmor import Candidate, Request, candidate_identity, select_top_k_candidates
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
from dynamic_kinetic_tree import KineticTreeKMORManager


PlanEvent = Tuple[str, str]
Plan = Tuple[PlanEvent, ...]


class TDTopKInsertionKMORManager(KineticTreeKMORManager):
    """Top-k beam adaptation of the time-dependent insertion policy.

    Unlike KineticTree, which retains a broad feasible schedule tree, this
    method keeps only the best ``k`` concrete schedules after every event.
    Each arrival expands all precedence-safe insertion pairs from that beam.
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
        planner_params["kinetic_tree_max_schedules"] = output_k
        super().__init__(
            graph,
            source,
            target,
            initial_requests,
            **planner_params,
        )

    def _schedule_maintenance_enabled(self) -> bool:
        return self.strategy == UpdateStrategy.TD_TOPK_INSERTION

    def _baseline_update_mode(self) -> str:
        return "TDTopKInsertion"

    def _initialization_note(self) -> str:
        return "time-dependent top-k beam insertion"


class TDInsertionKMORManager(DynamicKMORManager):
    """Maintain one route using precedence-safe time-dependent insertion."""

    def __init__(
        self,
        graph: TimeDependentGraph,
        source: str,
        target: str,
        initial_requests: Sequence[Request],
        **planner_params: object,
    ):
        super().__init__(graph, source, target, initial_requests, **planner_params)
        self.td_insertion_last_generated = 0
        self.td_insertion_last_feasible = 0
        self.td_insertion_last_removed = 0

    def clone(
        self,
        strategy: Optional[Union[UpdateStrategy, str]] = None,
    ) -> "TDInsertionKMORManager":
        return super().clone(strategy)  # type: ignore[return-value]

    @staticmethod
    def _stub(plan: Plan) -> Candidate:
        return Candidate(
            key_path=[],
            full_path=[],
            served_requests=[],
            pickup_info=[],
            total_cost=math.inf,
            total_time=math.inf,
            hops=0,
            served_count=0,
            score=-math.inf,
            route_events=list(plan),
        )

    def _current_plan(self) -> Plan:
        if not self.routes:
            return ()
        return tuple(self.remaining_route_events(self.routes[0]))

    @staticmethod
    def _insertions(plan: Plan, request_name: str) -> List[Plan]:
        generated: List[Plan] = []
        size = len(plan)
        for pickup_index in range(size + 1):
            with_pickup = (
                plan[:pickup_index]
                + (("pickup", request_name),)
                + plan[pickup_index:]
            )
            for dropoff_index in range(pickup_index + 1, size + 2):
                generated.append(
                    with_pickup[:dropoff_index]
                    + (("dropoff", request_name),)
                    + with_pickup[dropoff_index:]
                )
        return generated

    def _evaluate(self, plans: Sequence[Plan], planning_time: float) -> List[Candidate]:
        unique_plans = sorted(set(plans))
        self.td_insertion_last_generated += len(unique_plans)
        before = self._event_diagnostics.plan_prefix_labels_evaluated
        candidates = self.repair_candidates_bundled(
            [self._stub(plan) for plan in unique_plans],
            planning_time,
        )
        self.td_insertion_last_feasible += len(candidates)
        self._event_diagnostics.peak_open_size = max(
            self._event_diagnostics.peak_open_size,
            len(unique_plans),
        )
        self._td_insertion_last_expanded = (
            self._event_diagnostics.plan_prefix_labels_evaluated - before
        )
        return candidates

    def _commit(self, candidates: Sequence[Candidate]) -> None:
        selected = select_top_k_candidates(
            candidates,
            1,
            pareto_only=False,
            numeric_tolerance=float(
                self.planner_params.get("numeric_tolerance", 1e-9)
            ),
        )
        self.routes = list(selected)
        self.route_archive = list(selected)
        self.backup_routes = []
        self.available_routes = list(selected)
        self.route_archive_saturated = False

    def _repair_after_edge_update(
        self,
        plan: Plan,
        planning_time: float,
    ) -> List[Candidate]:
        candidates = self._evaluate([plan], planning_time)
        if candidates:
            return candidates

        # Preserve every onboard drop-off.  If traffic invalidates the route,
        # remove the smallest possible subset of still-waiting assignments.
        optional = sorted(
            {
                name
                for _action, name in plan
                if self.request_status.get(name) == RequestStatus.WAITING
            }
        )
        for remove_count in range(1, len(optional) + 1):
            reduced_plans = []
            for removed in itertools.combinations(optional, remove_count):
                removed_set = set(removed)
                reduced_plans.append(
                    tuple(event for event in plan if event[1] not in removed_set)
                )
            candidates = self._evaluate(reduced_plans, planning_time)
            if candidates:
                self.td_insertion_last_removed = remove_count
                return candidates
        return self._evaluate([()], planning_time)

    def _result(
        self,
        *,
        event_type: str,
        event_time: float,
        planning_time: float,
        update_start: float,
        affected_routes: int = 0,
        inserted_routes: int = 0,
        removed_stops: int = 0,
        note: str = "",
    ) -> DynamicUpdateResult:
        self.rebuild_route_records(planning_time)
        best = self.routes[0] if self.routes else None
        projected_served = self._status_count(RequestStatus.COMPLETED)
        projected_cost = self.vehicle.traveled_cost
        if best is not None:
            projected_served += best.served_count
            projected_cost += max(0.0, best.total_cost - planning_time)
        details = (
            f"td_generated={self.td_insertion_last_generated};"
            f"td_feasible={self.td_insertion_last_feasible};"
            f"td_removed={self.td_insertion_last_removed}"
        )
        if note:
            details = f"{note};{details}"
        return DynamicUpdateResult(
            event_type=event_type,
            event_time=event_time,
            active_requests=len(self.active_requests),
            route_count=len(self.routes),
            best_served=projected_served,
            best_cost=projected_cost if best is not None else math.inf,
            update_time=time.perf_counter() - update_start,
            recompute_time=0.0,
            expanded_states=getattr(self, "_td_insertion_last_expanded", 0),
            peak_open_size=self._event_diagnostics.peak_open_size,
            strategy=self.strategy.value,
            affected_routes=affected_routes,
            vehicle_node=self.vehicle.node,
            planning_time=planning_time,
            waiting_requests=self._status_count(RequestStatus.WAITING),
            onboard_requests=self._status_count(RequestStatus.ONBOARD),
            completed_requests=self._status_count(RequestStatus.COMPLETED),
            cancelled_requests=self._status_count(RequestStatus.CANCELLED),
            expired_requests=self._status_count(RequestStatus.EXPIRED),
            in_transit=self.vehicle.in_transit,
            update_mode="TDInsertion",
            repaired_routes=len(self.routes),
            inserted_routes=inserted_routes,
            removed_stops=removed_stops,
            search_exhausted=True,
            scope_complete=False,
            objective_complete=False,
            concrete_complete=False,
            completed=True,
            certificate_stage="not_applicable",
            archive_size=len(self.route_archive),
            backup_route_count=0,
            available_route_count=len(self.available_routes),
            available_topk_complete=False,
            generated_schedules=self.td_insertion_last_generated,
            feasible_schedules=self.td_insertion_last_feasible,
            evicted_schedules=0,
            schedule_generation_capped=False,
            note=details,
        )

    def initialize(self, current_time: float = 0.0) -> DynamicUpdateResult:
        if self.strategy != UpdateStrategy.TD_INSERTION:
            return super().initialize(current_time)
        self.reset_event_diagnostics()
        started = time.perf_counter()
        self.current_time = float(current_time)
        self.vehicle.current_time = float(current_time)
        self.vehicle.planning_time = float(current_time)
        planning_time = float(current_time)
        self.td_insertion_last_generated = 0
        self.td_insertion_last_feasible = 0
        self.td_insertion_last_removed = 0
        direct = self._evaluate([()], planning_time)
        self._commit(direct)
        for request in sorted(
            self.active_requests.values(),
            key=lambda item: (item.request_time, item.name),
        ):
            inserted = self._evaluate(
                self._insertions(self._current_plan(), request.name),
                planning_time,
            )
            inserted = [
                candidate
                for candidate in inserted
                if request.name in candidate.served_requests
            ]
            if inserted:
                self._commit(inserted)
        result = self._result(
            event_type="Initialize",
            event_time=current_time,
            planning_time=planning_time,
            update_start=started,
            note="time-dependent sequential insertion",
        )
        return self.attach_event_diagnostics(result)

    def handle_event(self, event: DynamicEvent) -> DynamicUpdateResult:
        if self.strategy != UpdateStrategy.TD_INSERTION:
            return super().handle_event(event)

        self.reset_event_diagnostics()
        started = time.perf_counter()
        retained_before = len(self.routes)
        self.td_insertion_last_generated = 0
        self.td_insertion_last_feasible = 0
        self.td_insertion_last_removed = 0
        self.current_time = event.event_time
        affected_routes = 0
        inserted_routes = 0
        removed_stops = 0
        note = ""

        if isinstance(event, EdgeUpdate):
            affected_routes = len(
                self.find_affected_routes(event.u, event.v, event.event_time)
            )
            self.advance_vehicle(event.event_time)
            plan = self._current_plan()
            self.apply_edge_update(event)
            event_type = "EdgeUpdate"
            note = event.edge_id
        elif isinstance(event, RequestArrival):
            self.advance_vehicle(event.event_time)
            plan = self._current_plan()
            self.active_requests[event.request.name] = event.request
            self.request_status[event.request.name] = RequestStatus.WAITING
            self.warm_request_lower_bound_paths(event.request)
            event_type = "RequestArrival"
            note = event.request.name
        elif isinstance(event, RequestCancel):
            affected_routes = len(
                self.find_routes_serving_request(event.request_name)
            )
            self.advance_vehicle(event.event_time)
            status = self.request_status.get(event.request_name)
            if status == RequestStatus.WAITING:
                self.request_status[event.request_name] = RequestStatus.CANCELLED
                self.active_requests.pop(event.request_name, None)
                removed_stops = 2 * affected_routes
                note = event.request_name
            elif status == RequestStatus.ONBOARD:
                note = f"{event.request_name}:ignored_already_onboard"
            else:
                note = (
                    f"{event.request_name}:"
                    f"ignored_{status.value.lower() if status else 'unknown'}"
                )
            plan = self._current_plan()
            event_type = "RequestCancel"
        else:
            raise TypeError(f"unsupported event: {event!r}")

        planning_time = max(event.event_time, self.vehicle.planning_time)
        self.expire_waiting_requests(planning_time)
        plan = tuple(
            event_item
            for event_item in plan
            if self.request_status.get(event_item[1])
            in {RequestStatus.WAITING, RequestStatus.ONBOARD}
        )
        previous_identity = candidate_identity(self.routes[0]) if self.routes else None

        if isinstance(event, RequestArrival):
            candidates = self._evaluate(
                self._insertions(plan, event.request.name),
                planning_time,
            )
            candidates = [
                candidate
                for candidate in candidates
                if event.request.name in candidate.served_requests
            ]
            if not candidates:
                candidates = self._evaluate([plan], planning_time)
        elif isinstance(event, EdgeUpdate):
            candidates = self._repair_after_edge_update(plan, planning_time)
        else:
            candidates = self._evaluate([plan], planning_time)
            if not candidates:
                candidates = self._repair_after_edge_update(plan, planning_time)

        self._commit(candidates)
        inserted_routes = int(
            bool(self.routes)
            and candidate_identity(self.routes[0]) != previous_identity
        )
        result = self._result(
            event_type=event_type,
            event_time=event.event_time,
            planning_time=planning_time,
            update_start=started,
            affected_routes=affected_routes,
            inserted_routes=inserted_routes,
            removed_stops=removed_stops,
            note=note,
        )
        return self.attach_event_context(result, retained_before)
