"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled scout, confirm, prune, and selective backward refine."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        if beta <= 0.0:
            return self._deterministic_anchor(env)

        schedule = self._schedule(beta, env)
        scout_cost = self._steps_to(env, 0.0, schedule["first_probe_time"]) + 1
        scout_count = max(1, min(schedule["scouts"], self._budget_cap(env, scout_cost)))

        root_ids = env.spawn(scout_count)
        try:
            self._advance_many(env, root_ids, schedule["first_probe_time"])
            self._preview_many(env, root_ids)
        except (BudgetExceededError, InvalidActionError):
            return self._fallback_answer(env)

        ranked = self._ranked_previews(env)
        if not ranked:
            return self._fallback_answer(env)

        keep_count = self._keep_count(beta, ranked, schedule)
        keep_ids = {preview.particle_id for preview in ranked[:keep_count]}
        self._prune_active(env, root_ids, keep_ids)

        confirm_ids = [pid for pid in root_ids if pid in keep_ids and self._is_active(env, pid)]
        if confirm_ids and schedule["second_probe_time"] > schedule["first_probe_time"]:
            try:
                self._advance_many(env, confirm_ids, schedule["second_probe_time"])
                self._preview_many(env, confirm_ids)
            except (BudgetExceededError, InvalidActionError):
                return self._fallback_answer(env)

        ranked = self._ranked_previews(env)
        if not ranked:
            return self._fallback_answer(env)

        finish_count = min(len(ranked), self._finish_count(beta, ranked, schedule))
        finish_ids = [preview.particle_id for preview in ranked[:finish_count]]
        self._prune_active(env, root_ids, set(finish_ids))
        self._finish_and_preview(env, finish_ids)

        ranked = self._ranked_previews(env)
        if not ranked:
            return self._fallback_answer(env)

        if beta >= schedule["branch_min_beta"] and env.budget_left > 0:
            self._branch_and_refine(env, ranked, schedule, beta)

        return env.answer(rule="best_preview_score")

    def _deterministic_anchor(self, env: FlowTTSEnv) -> AnswerRecord:
        root_id = env.spawn(1)[0]
        try:
            self._finish_to(env, root_id, 1.0)
        except (BudgetExceededError, InvalidActionError):
            pass
        return env.answer(rule="latest_active")

    def _schedule(self, beta: float, env: FlowTTSEnv) -> dict[str, float | int]:
        last_idx = max(1, len(env.time_grid) - 1)
        return {
            "scouts": 1
            + int(beta > 0.0)
            + int(beta >= 0.33)
            + int(beta >= 0.66)
            + int(beta >= 0.9),
            "first_probe_time": self._grid_time(
                env,
                self._grid_index(last_idx, 0.18 + 0.10 * beta),
            ),
            "second_probe_time": self._grid_time(
                env,
                self._grid_index(last_idx, 0.50 + 0.18 * beta),
            ),
            "child_time": self._grid_time(
                env,
                self._grid_index(last_idx, 0.82 - 0.22 * beta),
            ),
            "branch_min_beta": 0.35,
            "branch_base": int(beta >= 0.35)
            + int(beta >= 0.75)
            + int(beta >= 0.95),
            "finish_base": 1 + int(beta >= 0.45) + int(beta >= 0.85),
            "keep_base": 1 + int(beta >= 0.40) + int(beta >= 0.80),
            "gap_threshold": 0.012 + 0.018 * beta,
            "noise_policy": "inferred_noise" if beta < 0.55 else "mixed_noise",
            "noise_strength": max(0.25, min(0.85, 0.84 - 0.44 * beta)),
        }

    def _keep_count(
        self,
        beta: float,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> int:
        top_uncertainty = float(ranked[0].uncertainty or 0.0)
        keep = int(schedule["keep_base"])
        if top_uncertainty > 0.45 - 0.08 * beta:
            keep += 1
        if self._score_gap(ranked) <= float(schedule["gap_threshold"]):
            keep += 1
        return max(1, min(len(ranked), keep))

    def _finish_count(
        self,
        beta: float,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> int:
        finish = int(schedule["finish_base"])
        if self._score_gap(ranked) <= float(schedule["gap_threshold"]):
            finish += 1
        if float(ranked[0].uncertainty or 0.0) > 0.50 - 0.10 * beta:
            finish += 1
        return max(1, min(len(ranked), finish))

    def _branch_and_refine(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        beta: float,
    ) -> None:
        branch_count = int(schedule["branch_base"])
        if self._score_gap(ranked) <= float(schedule["gap_threshold"]):
            branch_count += 1
        branch_count = max(0, min(len(ranked), branch_count))
        if branch_count <= 0 or env.budget_left <= 0:
            return

        anchor_ids = [preview.id for preview in ranked[:branch_count]]
        extra_child = 1 if beta >= 0.9 and self._score_gap(ranked) <= float(schedule["gap_threshold"]) else 0
        for index, anchor_id in enumerate(anchor_ids):
            if env.budget_left <= 0:
                return
            num_children = 2 if index == 0 and extra_child else 1
            child_cost = self._estimate_finish_cost(env, schedule["child_time"]) + 1
            if env.budget_left < child_cost:
                return
            try:
                child_ids = env.backward(
                    anchor_id,
                    target_time=float(schedule["child_time"]),
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=num_children,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                return
            self._finish_and_preview(env, child_ids)

    def _fallback_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="latest_active")

    def _finish_and_preview(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            if not self._is_active(env, particle_id):
                if env.budget_left > 0:
                    try:
                        env.preview(particle_id, mode="clean_anchor", scorer="default")
                    except (BudgetExceededError, InvalidActionError):
                        return
                continue
            try:
                self._finish_to(env, particle_id, 1.0)
                if env.budget_left > 0:
                    env.preview(particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                return

    def _advance_many(self, env: FlowTTSEnv, particle_ids: list[int], target_time: float) -> None:
        for particle_id in particle_ids:
            self._advance_to(env, particle_id, target_time)

    def _preview_many(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _advance_to(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> None:
        state = env.get_state()
        current_time = float(state.particles[particle_id].time)
        for grid_time in env.time_grid:
            if grid_time > current_time + 1e-9 and grid_time <= target_time + 1e-9:
                env.forward(particle_id, target_time=float(grid_time), solver="euler")
                current_time = float(grid_time)

    def _finish_to(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> None:
        self._advance_to(env, particle_id, target_time)

    def _prune_active(
        self,
        env: FlowTTSEnv,
        particle_ids: list[int],
        keep_ids: set[int],
    ) -> None:
        losers = [
            particle_id
            for particle_id in particle_ids
            if particle_id not in keep_ids and self._is_active(env, particle_id)
        ]
        if losers:
            env.prune(losers)

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        latest: dict[int, PreviewRecord] = {}
        for particle in state.particles.values():
            if particle.status == "pruned" or particle.last_preview_id is None:
                continue
            preview = state.previews.get(particle.last_preview_id)
            if preview is not None and preview.score is not None:
                latest[particle.id] = preview
        return sorted(
            latest.values(),
            key=lambda preview: (
                float(preview.score or float("-inf")),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
            ),
            reverse=True,
        )

    def _score_gap(self, ranked: list[PreviewRecord]) -> float:
        if len(ranked) < 2:
            return float("inf")
        return float(ranked[0].score or 0.0) - float(ranked[1].score or 0.0)

    def _estimate_finish_cost(self, env: FlowTTSEnv, target_time: float) -> int:
        state = env.get_state()
        active = [particle for particle in state.particles.values() if particle.status == "active"]
        if not active:
            return 1
        current_time = min(float(particle.time) for particle in active)
        return max(1, self._steps_to(env, current_time, target_time))

    def _steps_to(self, env: FlowTTSEnv, start_time: float, target_time: float) -> int:
        return sum(1 for grid_time in env.time_grid if grid_time > start_time + 1e-9 and grid_time <= target_time + 1e-9)

    def _grid_index(self, last_idx: int, fraction: float) -> int:
        idx = int(round(last_idx * float(fraction)))
        return min(max(1, idx), last_idx)

    def _grid_time(self, env: FlowTTSEnv, index: int) -> float:
        index = min(max(0, int(index)), len(env.time_grid) - 1)
        return float(env.time_grid[index])

    def _budget_cap(self, env: FlowTTSEnv, per_item_cost: int) -> int:
        per_item_cost = max(1, int(per_item_cost))
        return max(1, int(env.budget_left // per_item_cost))

    def _is_active(self, env: FlowTTSEnv, particle_id: int) -> bool:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        return bool(particle is not None and particle.status == "active")
