"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Beta-scheduled scout, confirm, branch, and prune controller."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        initial_budget = max(0, int(env.budget_left))

        if beta <= 0.0:
            root_id = env.spawn(1)[0]
            try:
                self._finish(env, root_id)
            except (BudgetExceededError, InvalidActionError):
                return env.answer(rule="latest_active")
            return env.answer(rule="latest_active")

        schedule = self._schedule(env, beta)

        try:
            root_ids = env.spawn(int(schedule["roots"]))
            self._advance_to(env, root_ids, float(schedule["scout_time"]), allow_stop=True)
            self._preview_batch(env, root_ids)

            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer_fallback(env, beta)

            keep_ids = self._select_keep_ids(ranked, schedule, beta)
            self._prune_roots(env, root_ids, keep_ids)

            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer_fallback(env, beta)

            confirm_time = float(schedule["confirm_time"])
            if confirm_time > float(schedule["scout_time"]) and beta >= 0.25:
                self._confirm_survivors(env, keep_ids, confirm_time, schedule, beta)
                ranked = self._ranked_previews(env)
                if not ranked:
                    return self._answer_fallback(env, beta)

            branch_budget = self._branch_budget(beta)
            if branch_budget > 0:
                self._branch_and_refine(env, ranked, schedule, beta, initial_budget, branch_budget)

            if beta >= 0.6:
                self._late_confirmation(env, ranked, schedule, initial_budget, beta)

        except (BudgetExceededError, InvalidActionError):
            return env.answer(rule="latest_active")

        return env.answer(rule="best_preview_score")

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int]:
        grid_last = max(1, len(env.time_grid) - 1)
        scout_idx = self._grid_index(grid_last, 0.12 + 0.10 * beta)
        confirm_idx = self._grid_index(grid_last, 0.42 + 0.20 * beta)
        branch_idx = self._grid_index(grid_last, 0.68 + 0.12 * beta)
        finish_idx = self._grid_index(grid_last, 0.88 + 0.08 * beta)

        roots = 1 + int(beta >= 0.20) + int(beta >= 0.55) + int(beta >= 0.85)
        scout_preview_cap = 1 + int(beta >= 0.20) + int(beta >= 0.50) + int(beta >= 0.80)
        keep_base = 1 + int(beta >= 0.30) + int(beta >= 0.65)
        branch_children = 1 + int(beta >= 0.55) + int(beta >= 0.85)
        max_branch_anchors = 0
        if beta >= 0.25:
            max_branch_anchors += 1
        if beta >= 0.60:
            max_branch_anchors += 1
        if beta >= 0.90:
            max_branch_anchors += 1

        return {
            "roots": roots,
            "scout_time": self._grid_time(env, scout_idx),
            "confirm_time": self._grid_time(env, confirm_idx),
            "branch_time": self._grid_time(env, branch_idx),
            "finish_time": self._grid_time(env, finish_idx),
            "scout_preview_cap": scout_preview_cap,
            "keep_base": keep_base,
            "branch_children": branch_children,
            "max_branch_anchors": max_branch_anchors,
            "score_gap": 0.008 + 0.012 * beta,
            "uncertainty": 0.30 + 0.22 * beta,
            "noise_policy": "inferred_noise" if beta < 0.50 else "mixed_noise",
            "noise_strength": 0.88 - 0.38 * beta,
            "topoff_passes": 0 if beta < 0.45 else 1 + int(beta >= 0.80),
        }

    def _answer_fallback(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        if beta <= 0.0:
            return env.answer(rule="latest_active")
        ranked = self._ranked_previews(env)
        if ranked:
            return env.answer(rule="best_preview_score")
        return env.answer(rule="latest_active")

    def _finish(self, env: FlowTTSEnv, particle_id: int) -> None:
        state = env.get_state()
        current_time = float(state.particles[particle_id].time)
        for grid_time in env.time_grid:
            if grid_time > current_time + 1e-9:
                env.forward(particle_id, target_time=float(grid_time), solver="euler")
                current_time = float(grid_time)

    def _advance_to(self, env: FlowTTSEnv, particle_ids: list[int], target_time: float, allow_stop: bool = False) -> None:
        for particle_id in particle_ids:
            self._advance_particle(env, particle_id, target_time, allow_stop=allow_stop)

    def _advance_particle(self, env: FlowTTSEnv, particle_id: int, target_time: float, allow_stop: bool = False) -> None:
        state = env.get_state()
        current_time = float(state.particles[particle_id].time)
        for grid_time in env.time_grid:
            if grid_time > current_time + 1e-9 and grid_time <= target_time + 1e-9:
                if allow_stop and self._unsafe_to_continue(env, particle_id):
                    return
                env.forward(particle_id, target_time=float(grid_time), solver="euler")
                current_time = float(grid_time)

    def _preview_batch(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            if env.budget_left <= 0:
                return
            env.preview(particle_id, mode="clean_anchor", scorer="default")

    def _select_keep_ids(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        beta: float,
    ) -> set[int]:
        keep = int(schedule["keep_base"])
        gap = self._score_gap(ranked)
        top_uncertainty = float(ranked[0].uncertainty or 0.0)
        if gap <= float(schedule["score_gap"]):
            keep += 1
        if top_uncertainty >= float(schedule["uncertainty"]):
            keep += 1
        if beta >= 0.85 and len(ranked) > 2:
            keep += 1
        keep = max(1, min(len(ranked), keep))
        return {preview.particle_id for preview in ranked[:keep]}

    def _prune_roots(self, env: FlowTTSEnv, root_ids: list[int], keep_ids: set[int]) -> None:
        losers = [particle_id for particle_id in root_ids if particle_id not in keep_ids]
        if losers:
            env.prune(losers)

    def _confirm_survivors(
        self,
        env: FlowTTSEnv,
        keep_ids: set[int],
        target_time: float,
        schedule: dict[str, float | int],
        beta: float,
    ) -> None:
        for particle_id in list(keep_ids):
            if env.budget_left <= 0:
                return
            if beta < 0.65 and len(keep_ids) > 1 and self._unsafe_to_continue(env, particle_id):
                continue
            try:
                self._advance_particle(env, particle_id, target_time)
                if env.budget_left > 0:
                    env.preview(particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                continue

    def _branch_budget(self, beta: float) -> int:
        if beta < 0.25:
            return 0
        if beta < 0.55:
            return 1
        if beta < 0.85:
            return 2
        return 3

    def _branch_and_refine(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        beta: float,
        initial_budget: int,
        branch_budget: int,
    ) -> None:
        branch_count = min(int(schedule["max_branch_anchors"]), len(ranked), branch_budget)
        if branch_count <= 0:
            return

        target_time = float(schedule["branch_time"])
        finish_time = float(schedule["finish_time"])
        chosen = self._choose_branch_anchors(ranked, branch_count, schedule, beta)

        for index, anchor in enumerate(chosen):
            if env.budget_left <= 0 or self._spent(env, initial_budget) >= initial_budget:
                return
            num_children = int(schedule["branch_children"])
            if index == 0 and beta >= 0.8 and self._score_gap(ranked) <= float(schedule["score_gap"]):
                num_children += 1
            if index > 0 and beta < 0.75:
                num_children = 1
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=target_time,
                    noise_policy=str(schedule["noise_policy"]),
                    num_children=num_children,
                    strength=float(schedule["noise_strength"]),
                )
            except (BudgetExceededError, InvalidActionError):
                return

            self._prune_if_clear_loser(env, ranked, anchor)
            for child_id in child_ids:
                if env.budget_left <= 0:
                    return
                try:
                    self._advance_particle(env, child_id, finish_time, allow_stop=True)
                    if env.budget_left > 0:
                        env.preview(child_id, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return

        if beta >= 0.8 and env.budget_left > 0:
            self._top_off(env, ranked, schedule, initial_budget)

    def _choose_branch_anchors(
        self,
        ranked: list[PreviewRecord],
        branch_count: int,
        schedule: dict[str, float | int],
        beta: float,
    ) -> list[PreviewRecord]:
        chosen: list[PreviewRecord] = []
        gap = self._score_gap(ranked)
        top_uncertainty = float(ranked[0].uncertainty or 0.0)
        if gap <= float(schedule["score_gap"]) or top_uncertainty >= float(schedule["uncertainty"]):
            chosen.append(ranked[0])
        for preview in ranked[1:]:
            if len(chosen) >= branch_count:
                break
            if beta < 0.5 and preview.time < ranked[0].time:
                continue
            chosen.append(preview)
        if not chosen:
            chosen.append(ranked[0])
        return chosen[:branch_count]

    def _prune_if_clear_loser(self, env: FlowTTSEnv, ranked: list[PreviewRecord], anchor: PreviewRecord) -> None:
        if len(ranked) < 2:
            return
        gap = self._score_gap(ranked)
        if gap > 0.03 and float(anchor.score or 0.0) < float(ranked[0].score or 0.0):
            try:
                env.prune([anchor.particle_id])
            except (BudgetExceededError, InvalidActionError):
                return

    def _top_off(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        initial_budget: int,
    ) -> None:
        if not ranked or env.budget_left <= 0:
            return
        target_time = float(schedule["finish_time"])
        anchor = ranked[0]
        passes = int(schedule["topoff_passes"])
        for round_index in range(passes):
            if env.budget_left <= 0:
                return
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=target_time,
                    noise_policy="mixed_noise" if round_index > 0 else str(schedule["noise_policy"]),
                    num_children=1,
                    strength=max(0.25, float(schedule["noise_strength"]) * (0.90 if round_index == 0 else 0.75)),
                )
            except (BudgetExceededError, InvalidActionError):
                return
            for child_id in child_ids:
                if env.budget_left <= 0:
                    return
                try:
                    self._advance_particle(env, child_id, target_time)
                    env.preview(child_id, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return

    def _late_confirmation(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        initial_budget: int,
        beta: float,
    ) -> None:
        if not ranked or env.budget_left <= 0:
            return
        gap = self._score_gap(ranked)
        if gap > float(schedule["score_gap"]) and float(ranked[0].uncertainty or 0.0) < float(schedule["uncertainty"]):
            return
        passes = 1 + int(beta >= 0.85)
        confirm_time = float(schedule["finish_time"]) if beta >= 0.85 else float(schedule["branch_time"])
        for round_index in range(passes):
            if env.budget_left <= 0 or self._spent(env, initial_budget) >= initial_budget:
                return
            anchor = ranked[min(round_index, len(ranked) - 1)]
            try:
                child_ids = env.backward(
                    anchor.id,
                    target_time=confirm_time,
                    noise_policy="mixed_noise" if beta >= 0.75 else str(schedule["noise_policy"]),
                    num_children=1,
                    strength=max(0.2, float(schedule["noise_strength"]) * (0.80 if round_index == 0 else 0.65)),
                )
            except (BudgetExceededError, InvalidActionError):
                return
            for child_id in child_ids:
                if env.budget_left <= 0:
                    return
                try:
                    self._advance_particle(env, child_id, float(schedule["finish_time"]))
                    env.preview(child_id, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return

    def _unsafe_to_continue(self, env: FlowTTSEnv, particle_id: int) -> bool:
        state = env.get_state()
        particle = state.particles[particle_id]
        return particle.status == "pruned" or particle.time >= 1.0

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        previews = [
            preview
            for preview in state.previews.values()
            if preview.score is not None
            and preview.particle_id in state.particles
            and state.particles[preview.particle_id].status != "pruned"
        ]
        return sorted(
            previews,
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

    def _grid_index(self, grid_last: int, fraction: float) -> int:
        fraction = min(max(float(fraction), 0.0), 1.0)
        return min(grid_last, max(0, int(round(grid_last * fraction))))

    def _grid_time(self, env: FlowTTSEnv, index: int) -> float:
        return float(env.time_grid[min(max(int(index), 0), len(env.time_grid) - 1)])

    def _spent(self, env: FlowTTSEnv, initial_budget: int) -> int:
        return max(0, int(initial_budget - env.budget_left))
