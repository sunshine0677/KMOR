"""Kinetic-tree baseline adapted to plan-level dynamic KMOR.

The original kinetic-tree method maintains feasible pickup/drop-off schedules
and inserts a new request into every legal pair of positions.  KMOR also
allows a request to remain unserved, so this adaptation retains both the old
schedule and every feasible insertion.  Physical paths and all feasibility
constraints are evaluated by :class:`DynamicKMORManager`; the baseline only
changes how concrete stop schedules are maintained.
"""

from __future__ import annotations

import math
import time
from typing import Iterable, List, Optional, Sequence, Tuple, Union

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


PlanEvent = Tuple[str, str]
Plan = Tuple[PlanEvent, ...]


class KineticTreeKMORManager(DynamicKMORManager):
    """Maintain a bounded kinetic tree of feasible concrete stop schedules.

    ``kinetic_tree_max_schedules`` bounds the retained terminal schedules, as
    practical kinetic-tree systems must do under exponential schedule growth.
    ``kinetic_tree_max_generated`` bounds raw insertion sequences considered
    for one request.  Both bounds are deterministic and are reported in the
    result note; they do not change the shared route-feasibility semantics.
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
        self.kinetic_tree_max_schedules = max(
            output_k,
            int(planner_params.pop("kinetic_tree_max_schedules", 5000)),
        )
        self.kinetic_tree_max_generated = max(
            self.kinetic_tree_max_schedules,
            int(planner_params.pop("kinetic_tree_max_generated", 100000)),
        )
        super().__init__(graph, source, target, initial_requests, **planner_params)
        self.kinetic_schedule_pool: List[Candidate] = []
        self.kinetic_last_generated = 0
        self.kinetic_last_feasible = 0
        self.kinetic_last_evicted = 0
        self.kinetic_generation_capped = False

    def _schedule_maintenance_enabled(self) -> bool:
        return self.strategy == UpdateStrategy.KINETIC_TREE

    def _baseline_update_mode(self) -> str:
        return "KineticTreeTopK"

    def _initialization_note(self) -> str:
        return "kinetic-tree initial schedules"

    def clone(
        self,
        strategy: Optional[Union[UpdateStrategy, str]] = None,
    ) -> "KineticTreeKMORManager":
        cloned = super().clone(strategy)
        cloned.kinetic_schedule_pool = list(self.kinetic_schedule_pool)
        return cloned

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

    @staticmethod
    def _without_target(candidate: Candidate) -> Plan:
        return tuple(
            (action, name)
            for action, name in candidate.route_events
            if action in {"pickup", "dropoff"}
        )

    def _current_plans(self) -> List[Plan]:
        candidates = self.kinetic_schedule_pool or self.route_archive or self.routes
        plans = {
            tuple(self.remaining_route_events(candidate))
            for candidate in candidates
        }
        return sorted(plans) or [()]

    def _insert_request(self, plans: Iterable[Plan], request_name: str) -> List[Plan]:
        """Return old plans plus all precedence-safe insertion positions."""
        generated = set()
        self.kinetic_last_generated = 0
        self.kinetic_generation_capped = False
        capped = False
        for plan in plans:
            generated.add(tuple(plan))
            if any(name == request_name for _, name in plan):
                continue
            size = len(plan)
            for pickup_index in range(size + 1):
                with_pickup = (
                    plan[:pickup_index]
                    + (("pickup", request_name),)
                    + plan[pickup_index:]
                )
                for dropoff_index in range(pickup_index + 1, size + 2):
                    expanded = (
                        with_pickup[:dropoff_index]
                        + (("dropoff", request_name),)
                        + with_pickup[dropoff_index:]
                    )
                    self.kinetic_last_generated += 1
                    generated.add(expanded)
                    if len(generated) >= self.kinetic_tree_max_generated:
                        capped = True
                        self.kinetic_generation_capped = True
                        break
                if capped:
                    break
            if capped:
                break
        return sorted(generated)

    def _evaluate_plans(
        self,
        plans: Sequence[Plan],
        planning_time: float,
    ) -> List[Candidate]:
        stubs = [self._stub(plan) for plan in plans]
        before = self._event_diagnostics.plan_prefix_labels_evaluated
        candidates = self.repair_candidates_bundled(stubs, planning_time)
        evaluated = (
            self._event_diagnostics.plan_prefix_labels_evaluated - before
        )
        self._event_diagnostics.peak_open_size = max(
            self._event_diagnostics.peak_open_size,
            len(plans),
        )
        self.kinetic_last_feasible = len(candidates)
        self._kinetic_last_expanded = evaluated
        return candidates

    def _retain_schedule_pool(self, candidates: Sequence[Candidate]) -> None:
        unique = {}
        for candidate in candidates:
            identity = candidate_identity(candidate)
            old = unique.get(identity)
            if old is None or candidate.total_cost < old.total_cost:
                unique[identity] = candidate
        all_candidates = list(unique.values())
        retained = select_top_k_candidates(
            all_candidates,
            self.kinetic_tree_max_schedules,
            pareto_only=False,
            numeric_tolerance=float(
                self.planner_params.get("numeric_tolerance", 1e-9)
            ),
        )
        self.kinetic_last_evicted = max(0, len(all_candidates) - len(retained))
        self.kinetic_schedule_pool = retained
        self.route_archive = list(retained)
        self.route_archive_saturated = self.kinetic_last_evicted > 0
        self.routes = select_top_k_candidates(
            retained,
            int(self.planner_params.get("k", 5)),
            pareto_only=True,
            numeric_tolerance=float(
                self.planner_params.get("numeric_tolerance", 1e-9)
            ),
        )
        operational = select_top_k_candidates(
            retained,
            int(self.planner_params.get("k", 5)),
            pareto_only=False,
            numeric_tolerance=float(
                self.planner_params.get("numeric_tolerance", 1e-9)
            ),
        )
        formal_ids = {candidate_identity(candidate) for candidate in self.routes}
        self.backup_routes = [
            candidate
            for candidate in operational
            if candidate_identity(candidate) not in formal_ids
        ]
        self.available_routes = list(self.routes)
        available_ids = set(formal_ids)
        for candidate in self.backup_routes:
            identity = candidate_identity(candidate)
            if identity in available_ids:
                continue
            self.available_routes.append(candidate)
            available_ids.add(identity)
            if len(self.available_routes) >= int(self.planner_params.get("k", 5)):
                break

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
            f"kt_pool={len(self.kinetic_schedule_pool)};"
            f"generated={self.kinetic_last_generated};"
            f"feasible={self.kinetic_last_feasible};"
            f"evicted={self.kinetic_last_evicted}"
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
            expanded_states=getattr(self, "_kinetic_last_expanded", 0),
            peak_open_size=len(self.kinetic_schedule_pool),
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
            update_mode=self._baseline_update_mode(),
            repaired_routes=len(self.kinetic_schedule_pool),
            inserted_routes=inserted_routes,
            removed_stops=removed_stops,
            search_exhausted=True,
            scope_complete=False,
            objective_complete=False,
            concrete_complete=False,
            completed=True,
            certificate_stage="not_applicable",
            archive_size=len(self.kinetic_schedule_pool),
            backup_route_count=len(self.backup_routes),
            available_route_count=len(self.available_routes),
            available_topk_complete=(
                len(self.available_routes)
                >= int(self.planner_params.get("k", 5))
            ),
            generated_schedules=self.kinetic_last_generated,
            feasible_schedules=self.kinetic_last_feasible,
            evicted_schedules=self.kinetic_last_evicted,
            schedule_generation_capped=self.kinetic_generation_capped,
            note=details,
        )

    def initialize(self, current_time: float = 0.0) -> DynamicUpdateResult:
        if not self._schedule_maintenance_enabled():
            return super().initialize(current_time)
        self.reset_event_diagnostics()
        started = time.perf_counter()
        self.current_time = float(current_time)
        self.vehicle.current_time = float(current_time)
        self.vehicle.planning_time = float(current_time)
        planning_time = float(current_time)
        candidates = self._evaluate_plans([()], planning_time)
        self._retain_schedule_pool(candidates)
        for request in sorted(
            self.active_requests.values(),
            key=lambda item: (item.request_time, item.name),
        ):
            plans = self._insert_request(self._current_plans(), request.name)
            candidates = self._evaluate_plans(plans, planning_time)
            self._retain_schedule_pool(candidates)
        result = self._result(
            event_type="Initialize",
            event_time=current_time,
            planning_time=planning_time,
            update_start=started,
            note=self._initialization_note(),
        )
        return self.attach_event_diagnostics(result)

    def handle_event(self, event: DynamicEvent) -> DynamicUpdateResult:
        if not self._schedule_maintenance_enabled():
            return super().handle_event(event)

        self.reset_event_diagnostics()
        update_start = time.perf_counter()
        retained_before = len(self.kinetic_schedule_pool)
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
            self.apply_edge_update(event)
            event_type = "EdgeUpdate"
            note = event.edge_id
        elif isinstance(event, RequestArrival):
            self.advance_vehicle(event.event_time)
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
            event_type = "RequestCancel"
        else:
            raise TypeError(f"unsupported event: {event!r}")

        planning_time = max(event.event_time, self.vehicle.planning_time)
        self.expire_waiting_requests(planning_time)
        plans = self._current_plans()
        if isinstance(event, RequestArrival):
            plans = self._insert_request(plans, event.request.name)
        else:
            self.kinetic_last_generated = 0
            self.kinetic_generation_capped = False
        previous_ids = {
            candidate_identity(candidate) for candidate in self.kinetic_schedule_pool
        }
        candidates = self._evaluate_plans(plans, planning_time)
        inserted_routes = sum(
            candidate_identity(candidate) not in previous_ids
            for candidate in candidates
        )
        self._retain_schedule_pool(candidates)
        result = self._result(
            event_type=event_type,
            event_time=event.event_time,
            planning_time=planning_time,
            update_start=update_start,
            affected_routes=affected_routes,
            inserted_routes=inserted_routes,
            removed_stops=removed_stops,
            note=note,
        )
        return self.attach_event_context(result, retained_before)

