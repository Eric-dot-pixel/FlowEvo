"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Preview-gated multi-root scout and refine controller."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        initial_budget = max(0, int(env.budget_left))
        target_nfe = self._target_nfe(initial_budget, beta)

        try:
            schedule = self._schedule(beta, env)
            if beta <= 0.0:
                root_id = env.spawn(1)[0]
                self._finish(env, root_id)
                return env.answer(rule="latest_active")

            root_ids = env.spawn(max(1, int(schedule["scouts"])))
            self._advance_batch(env, root_ids, float(schedule["scout_time"]))
            self._preview_batch(env, root_ids)
            ranked = self._ranked_previews(env)
            if not ranked:
                return self._fallback_answer(env)

            keep_ids = self._select_keep_ids(ranked, schedule)
            self._prune_roots(env, root_ids, keep_ids)

            if float(schedule["confirm_time"]) > float(schedule["scout_time"]):
                confirm_ids = self._confirm_roots(env, keep_ids, float(schedule["confirm_time"]))
                self._preview_batch(env, confirm_ids)
                ranked = self._ranked_previews(env)
                if not ranked:
                    return self._fallback_answer(env)

            if env.budget_left <= 0:
                return env.answer(rule="best_preview_score")

            self._branch_and_refine(env, ranked, schedule, beta)

            if env.budget_left > 0 and self._spent(env, initial_budget) < target_nfe:
                self._top_off(env, ranked, schedule, beta, initial_budget, target_nfe)

        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")

        return env.answer(rule="best_preview_score")

    def _schedule(self, beta: float, env: FlowTTSEnv) -> dict[str, float | int]:
        grid_last = max(1, len(env.time_grid) - 1)
        scout_stage = self._grid_time(env, self._grid_index(grid_last, 0.16 + 0.14 * beta))
        confirm_stage = self._grid_time(env, self._grid_index(grid_last, 0.54 + 0.16 * beta))
        branch_stage = self._grid_time(env, self._grid_index(grid_last, 0.66 + 0.12 * beta))
        finish_stage = self._grid_time(env, self._grid_index(grid_last, 0.92 + 0.06 * beta))

        scouts = 1 + int(beta > 0.0) + int(beta >= 0.20) + int(beta >= 0.45) + int(beta >= 0.70) + int(beta >= 0.90)
        keep_base = 1 + int(beta >= 0.30) + int(beta >= 0.60) + int(beta >= 0.90)
        branch_base = 0
        if beta >= 0.25:
            branch_base += 1
        if beta >= 0.65:
            branch_base += 1
        if beta >= 0.90:
            branch_base += 1

        return {
            "scouts": scouts,
            "scout_time": scout_stage,
            "confirm_time": confirm_stage,
            "branch_time": branch_stage,
            "finish_time": finish_stage,
            "keep_base": keep_base,
            "branch_base": branch_base,
            "score_gap": 0.010 + 0.015 * beta,
            "uncertainty": 0.46 - 0.12 * beta,
            "children": 1 + int(beta >= 0.50) + int(beta >= 0.85),
            "noise_policy": "inferred_noise" if beta < 0.55 else "mixed_noise",
            "noise_strength": 0.80 - 0.35 * beta,
            "confirm_children": 1 + int(beta >= 0.90),
        }

    def _finish(self, env: FlowTTSEnv, particle_id: int) -> None:
        state = env.get_state()
        current_time = state.particles[particle_id].time
        for target_time in env.time_grid:
            if target_time > current_time:
                env.forward(particle_id, target_time=target_time, solver="euler")
                current_time = target_time

    def _advance_batch(self, env: FlowTTSEnv, particle_ids: list[int], target_time: float) -> None:
        for particle_id in particle_ids:
            self._advance_to(env, particle_id, target_time)

    def _preview_batch(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            if env.budget_left <= 0:
                return
            env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _select_keep_ids(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> set[int]:
        keep = int(schedule["keep_base"])
        if self._score_gap(ranked) <= float(schedule["score_gap"]):
            keep += 1
        if float(ranked[0].uncertainty or 0.0) >= float(schedule["uncertainty"]):
            keep += 1
        keep = max(1, min(len(ranked), keep))
        return {preview.particle_id for preview in ranked[:keep]}

    def _prune_roots(self, env: FlowTTSEnv, root_ids: list[int], keep_ids: set[int]) -> None:
        losers = [particle_id for particle_id in root_ids if particle_id not in keep_ids]
        if losers:
            env.prune(losers)

    def _confirm_roots(self, env: FlowTTSEnv, keep_ids: set[int], target_time: float) -> list[int]:
        confirmed: list[int] = []
        for particle_id in list(keep_ids):
            if env.budget_left <= 0:
                break
            try:
                self._advance_to(env, particle_id, target_time)
                confirmed.append(particle_id)
            except (BudgetExceededError, InvalidActionError):
                continue
        return confirmed

    def _branch_and_refine(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        beta: float,
    ) -> None:
        branch_count = int(schedule["branch_base"])
        if self._score_gap(ranked) <= float(schedule["score_gap"]):
            branch_count += 1
        if float(ranked[0].uncertainty or 0.0) >= float(schedule["uncertainty"]):
            branch_count += 1
        branch_count = max(0, min(len(ranked), branch_count))
        if branch_count <= 0:
            return

        branch_ids = [preview.id for preview in ranked[:branch_count]]
        target_time = float(schedule["branch_time"])
        finish_time = float(schedule["finish_time"])
        for index, anchor_id in enumerate(branch_ids):
            if env.budget_left <= 0:
                return
            num_children = int(schedule["children"])
            if index > 0 and beta < 0.75:
                num_children = 1
            if index == 0 and beta >= 0.9 and self._score_gap(ranked) <= float(schedule["score_gap"]):
                num_children += 1
            try:
                child_ids = env.backward(
                    anchor_id,
                    target_time=target_time,
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=num_children,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                return

            for child_id in child_ids:
                if env.budget_left <= 0:
                    return
                try:
                    self._advance_to(env, child_id, finish_time)
                    env.preview(child_id, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return

    def _top_off(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        beta: float,
        initial_budget: int,
        target_nfe: int,
    ) -> None:
        if not ranked or env.budget_left <= 0:
            return
        if self._spent(env, initial_budget) >= target_nfe:
            return
        gap = self._score_gap(ranked)
        top = ranked[0]
        if beta < 0.65 and gap > float(schedule["score_gap"]) and float(top.uncertainty or 0.0) < float(schedule["uncertainty"]):
            return

        topoffs = 1 + int(beta >= 0.85)
        for round_index in range(topoffs):
            if env.budget_left <= 0 or self._spent(env, initial_budget) >= target_nfe:
                return
            anchor = ranked[min(round_index, len(ranked) - 1)]
            child_count = int(schedule["children"])
            if round_index > 0:
                child_count = max(1, child_count - 1)
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=float(schedule["branch_time"]) if beta < 0.8 else float(schedule["finish_time"]),
                    noise_policy="mixed_noise" if beta >= 0.55 else "fresh_noise",
                    num_children=child_count,
                    strength=max(0.25, float(schedule["noise_strength"]) * (0.85 if round_index == 0 else 0.70)),
                )
            except (BudgetExceededError, InvalidActionError):
                break
            for child_id in child_ids:
                if env.budget_left <= 0 or self._spent(env, initial_budget) >= target_nfe:
                    return
                try:
                    self._advance_to(env, child_id, float(schedule["finish_time"]))
                    env.preview(child_id, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return

    def _advance_to(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> None:
        state = env.get_state()
        current_time = float(state.particles[particle_id].time)
        for grid_time in env.time_grid:
            if grid_time > current_time + 1e-9 and grid_time <= target_time + 1e-9:
                env.forward(particle_id, target_time=float(grid_time), solver="euler")
                current_time = float(grid_time)

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        previews: dict[int, PreviewRecord] = {}
        for particle in state.particles.values():
            if particle.status == "pruned" or particle.last_preview_id is None:
                continue
            preview = state.previews.get(particle.last_preview_id)
            if preview is None or preview.score is None:
                continue
            previews[particle.id] = preview
        return sorted(
            previews.values(),
            key=lambda preview: (
                float(preview.score),
                -float(preview.uncertainty or 0.0),
                float(preview.time),
            ),
            reverse=True,
        )

    def _score_gap(self, ranked: list[PreviewRecord]) -> float:
        if len(ranked) < 2:
            return float("inf")
        return float(ranked[0].score or 0.0) - float(ranked[1].score or 0.0)

    def _target_nfe(self, initial_budget: int, beta: float) -> int:
        if initial_budget <= 0:
            return 0
        if beta <= 0.0:
            return min(initial_budget, 10)
        fraction = 0.09 + 0.34 * beta + 0.28 * beta * beta
        return min(initial_budget, max(10, int(round(initial_budget * fraction))))

    def _fallback_answer(self, env: FlowTTSEnv) -> AnswerRecord:
        state = env.get_state()
        if state.previews:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="latest_active")

    def _grid_index(self, last_idx: int, fraction: float) -> int:
        idx = int(round(last_idx * float(fraction)))
        return min(max(1, idx), last_idx)

    def _grid_time(self, env: FlowTTSEnv, index: int) -> float:
        index = min(max(0, int(index)), len(env.time_grid) - 1)
        return float(env.time_grid[index])

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))
