"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Mid-time tournament with champion-challenger local refinement."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(env, beta)

        if beta <= 0.0:
            return self._deterministic_answer(env)

        try:
            scout_cost = max(1, self._forward_cost_from_time(env, 0.0, float(schedule["scout_time"]))) + 1
            affordable_roots = max(1, env.budget_left // max(1, scout_cost))
            root_count = min(int(schedule["roots"]), affordable_roots)
            root_ids = env.spawn(root_count)

            self._advance_batch(env, root_ids, float(schedule["scout_time"]))
            self._preview_batch(env, root_ids)

            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            keep_ids = self._select_particle_ids(
                ranked,
                base=int(schedule["keep_base"]),
                gap_threshold=float(schedule["scout_gap"]),
                uncertainty_threshold=float(schedule["uncertainty"]),
            )
            self._prune_active_except(env, root_ids, keep_ids)

            self._advance_and_preview(env, list(keep_ids), float(schedule["confirm_time"]))
            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            finalist_ids = self._select_particle_ids(
                ranked,
                base=int(schedule["finish_base"]),
                gap_threshold=float(schedule["confirm_gap"]),
                uncertainty_threshold=max(0.0, float(schedule["uncertainty"]) - 0.03),
            )
            self._finish_and_preview(env, list(finalist_ids))

            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            self._refine_contenders(env, ranked, schedule)
            self._late_expand(env, schedule)
        except (BudgetExceededError, InvalidActionError):
            return self._answer(env)

        return self._answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int | str]:
        grid_last = max(1, len(env.time_grid) - 1)
        scout_fraction = 0.38 + 0.10 * beta
        confirm_fraction = 0.74 + 0.08 * beta
        branch_fraction = 0.78 - 0.20 * beta

        return {
            "roots": min(8, 2 + int(7.0 * max(0.0, beta - 0.15))),
            "scout_time": self._grid_time(env, self._grid_index(grid_last, scout_fraction)),
            "confirm_time": self._grid_time(env, self._grid_index(grid_last, confirm_fraction)),
            "branch_time": self._grid_time(env, self._grid_index(grid_last, branch_fraction)),
            "keep_base": min(4, 1 + int(4.0 * beta)),
            "finish_base": min(4, 1 + int(4.0 * max(0.0, beta - 0.15))),
            "branch_anchors": 0
            + int(beta >= 0.42)
            + int(beta >= 0.72)
            + int(beta >= 0.94),
            "children": 0
            + int(beta >= 0.42)
            + int(beta >= 0.50)
            + int(beta >= 0.68)
            + int(beta >= 0.82)
            + int(beta >= 0.95),
            "late_finishes": 0 + int(beta >= 0.70) + int(beta >= 0.92),
            "scout_gap": 0.018 + 0.010 * beta,
            "confirm_gap": 0.012 + 0.010 * beta,
            "uncertainty": max(0.12, 0.28 - 0.12 * beta),
            "noise_policy": "fresh_noise"
            if beta < 0.42
            else ("inferred_noise" if beta < 0.80 else "mixed_noise"),
            "noise_strength": max(0.35, 0.95 - 0.50 * beta),
        }

    def _deterministic_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        root_id = env.spawn(1)[0]
        try:
            self._advance_particle(env, root_id, 1.0)
        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")
        return env.answer(rule="latest_active")

    def _advance_batch(self, env: FlowTTSEnv, particle_ids: list[int], target_time: float) -> None:
        for particle_id in particle_ids:
            self._advance_particle(env, particle_id, target_time)

    def _advance_and_preview(self, env: FlowTTSEnv, particle_ids: list[int], target_time: float) -> None:
        for particle_id in particle_ids:
            state = env.get_state()
            particle = state.particles.get(particle_id)
            if particle is None or particle.status == "pruned":
                continue
            advanced = self._advance_particle(env, particle_id, target_time)
            if env.budget_left <= 0:
                return
            if advanced and self._needs_preview(env, particle_id):
                env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _finish_and_preview(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            state = env.get_state()
            particle = state.particles.get(particle_id)
            if particle is None or particle.status == "pruned":
                continue
            advanced = self._advance_particle(env, particle_id, 1.0)
            if env.budget_left <= 0:
                return
            if (advanced or self._needs_preview(env, particle_id)) and env.budget_left > 0:
                env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _refine_contenders(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int | str],
    ) -> None:
        max_anchors = int(schedule["branch_anchors"])
        total_children = int(schedule["children"])
        if max_anchors <= 0 or total_children <= 0 or not ranked:
            return

        gap = self._score_gap(ranked)
        uncertainty = float(ranked[0].uncertainty or 0.0)
        close_threshold = float(schedule["confirm_gap"])

        if gap <= close_threshold * 0.5:
            total_children += 1
        if uncertainty >= float(schedule["uncertainty"]) and max_anchors > 1:
            total_children += 1

        contenders = [ranked[0]]
        if len(ranked) > 1 and max_anchors > 1:
            if gap <= close_threshold or uncertainty >= float(schedule["uncertainty"]):
                contenders.append(ranked[1])
        if len(ranked) > 2 and max_anchors > 2 and gap <= close_threshold * 0.5:
            contenders.append(ranked[2])

        child_cost = self._child_finish_preview_cost(env, float(schedule["branch_time"]))
        affordable_children = env.budget_left // max(1, child_cost)
        total_children = min(total_children, affordable_children)
        if total_children <= 0:
            return

        allocations = self._allocate_children(contenders, total_children, close_threshold, gap)
        for preview, num_children in allocations:
            if num_children <= 0 or env.budget_left < child_cost:
                continue
            try:
                child_ids = env.backward(
                    preview.id,
                    target_time=float(schedule["branch_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=num_children,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                return

            self._finish_and_preview(env, child_ids)

    def _late_expand(self, env: FlowTTSEnv, schedule: dict[str, float | int | str]) -> None:
        remaining = int(schedule["late_finishes"])
        if remaining <= 0:
            return

        while remaining > 0 and env.budget_left > 0:
            ranked = self._ranked_previews(env)
            if not ranked:
                return

            gap = self._score_gap(ranked)
            uncertainty = float(ranked[0].uncertainty or 0.0)
            if gap > float(schedule["confirm_gap"]) * 1.35 and uncertainty < float(schedule["uncertainty"]):
                return

            candidate_id = self._best_active_ranked_particle(env, ranked)
            if candidate_id is None:
                return

            finish_cost = self._finish_preview_cost(env, candidate_id)
            if finish_cost <= 0 or env.budget_left < finish_cost:
                return

            self._finish_and_preview(env, [candidate_id])
            remaining -= 1

    def _select_particle_ids(
        self,
        ranked: list[PreviewRecord],
        base: int,
        gap_threshold: float,
        uncertainty_threshold: float,
    ) -> set[int]:
        if not ranked:
            return set()

        keep = min(len(ranked), max(1, base))
        gap = self._score_gap(ranked)
        top_uncertainty = float(ranked[0].uncertainty or 0.0)

        if gap <= gap_threshold:
            keep += 1
        if top_uncertainty >= uncertainty_threshold:
            keep += 1

        keep = min(len(ranked), max(1, keep))
        return {preview.particle_id for preview in ranked[:keep]}

    def _allocate_children(
        self,
        contenders: list[PreviewRecord],
        total_children: int,
        close_threshold: float,
        gap: float,
    ) -> list[tuple[PreviewRecord, int]]:
        if not contenders or total_children <= 0:
            return []
        if len(contenders) == 1:
            return [(contenders[0], total_children)]

        if len(contenders) == 2:
            if gap <= close_threshold * 0.5:
                first = (total_children + 1) // 2
            else:
                first = max(1, (3 * total_children + 1) // 4)
            second = max(0, total_children - first)
            result = [(contenders[0], first)]
            if second > 0:
                result.append((contenders[1], second))
            return result

        allocations: list[tuple[PreviewRecord, int]] = []
        remaining = total_children
        for index, preview in enumerate(contenders):
            slots_left = len(contenders) - index
            count = max(1, remaining - (slots_left - 1))
            allocations.append((preview, count))
            remaining -= count
            if remaining <= 0:
                break
        return allocations

    def _preview_batch(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            if env.budget_left <= 0:
                return
            if self._needs_preview(env, particle_id):
                env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _prune_active_except(self, env: FlowTTSEnv, particle_ids: list[int], keep_ids: set[int]) -> None:
        state = env.get_state()
        losers = [
            particle_id
            for particle_id in particle_ids
            if particle_id not in keep_ids
            and particle_id in state.particles
            and state.particles[particle_id].status == "active"
        ]
        if not losers:
            return
        try:
            env.prune(losers)
        except InvalidActionError:
            return

    def _advance_particle(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> bool:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status == "pruned":
            return False

        start_time = float(particle.time)
        current_time = start_time
        for grid_time in env.time_grid:
            grid_time = float(grid_time)
            if grid_time > current_time + 1e-9 and grid_time <= target_time + 1e-9:
                env.forward(particle_id, target_time=grid_time, solver="euler")
                current_time = grid_time
        return current_time > start_time + 1e-9

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        latest_by_particle: dict[int, PreviewRecord] = {}
        for particle in state.particles.values():
            if particle.status == "pruned" or particle.last_preview_id is None:
                continue
            preview = state.previews.get(particle.last_preview_id)
            if preview is None or preview.score is None:
                continue
            latest_by_particle[particle.id] = preview
        return sorted(
            latest_by_particle.values(),
            key=lambda preview: (
                float(preview.score),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
            ),
            reverse=True,
        )

    def _needs_preview(self, env: FlowTTSEnv, particle_id: int) -> bool:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None:
            return False
        if particle.last_preview_id is None:
            return True
        preview = state.previews.get(particle.last_preview_id)
        if preview is None:
            return True
        return float(preview.time) + 1e-9 < float(particle.time)

    def _best_active_ranked_particle(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
    ) -> int | None:
        state = env.get_state()
        for preview in ranked:
            particle = state.particles.get(preview.particle_id)
            if particle is None:
                continue
            if particle.status == "active" and float(particle.time) < 1.0 - 1e-9:
                return particle.id
        return None

    def _score_gap(self, ranked: list[PreviewRecord]) -> float:
        if len(ranked) < 2:
            return float("inf")
        return float(ranked[0].score or 0.0) - float(ranked[1].score or 0.0)

    def _forward_cost_from_time(self, env: FlowTTSEnv, start_time: float, target_time: float) -> int:
        if target_time <= start_time:
            return 0
        return sum(
            1
            for grid_time in env.time_grid
            if float(grid_time) > start_time + 1e-9 and float(grid_time) <= target_time + 1e-9
        )

    def _child_finish_preview_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        return self._forward_cost_from_time(env, target_time, 1.0) + 1

    def _finish_preview_cost(self, env: FlowTTSEnv, particle_id: int) -> int:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status == "pruned":
            return 0
        forward_cost = self._forward_cost_from_time(env, float(particle.time), 1.0)
        preview_cost = 1 if self._needs_preview(env, particle_id) else 0
        return forward_cost + preview_cost

    def _grid_index(self, grid_last: int, fraction: float) -> int:
        fraction = min(max(float(fraction), 0.0), 1.0)
        return min(grid_last, max(1, int(grid_last * fraction + 1e-9)))

    def _grid_time(self, env: FlowTTSEnv, index: int) -> float:
        index = min(max(0, int(index)), len(env.time_grid) - 1)
        return float(env.time_grid[index])

    def _answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            try:
                return env.answer(rule="best_preview_score")
            except InvalidActionError:
                pass
        return env.answer(rule="latest_active")
