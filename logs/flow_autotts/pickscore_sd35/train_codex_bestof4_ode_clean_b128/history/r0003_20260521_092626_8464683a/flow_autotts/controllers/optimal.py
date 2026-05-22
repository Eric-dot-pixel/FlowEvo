"""Candidate controller for SD3.5 PickScore discovery."""

from __future__ import annotations

from flow_autotts.core.env import FlowTTSEnv
from flow_autotts.core.errors import BudgetExceededError, InvalidActionError
from flow_autotts.core.state import AnswerRecord, PreviewRecord


class OptimalController:
    """Scout width first, then confirm, then locally refine uncertain winners."""

    def solve(self, env: FlowTTSEnv, beta: float) -> AnswerRecord:
        beta = min(max(float(beta), 0.0), 1.0)
        schedule = self._schedule(env, beta)

        if beta <= 0.0:
            return self._deterministic_answer(env)

        try:
            root_ids = env.spawn(int(schedule["roots"]))
            self._advance_batch(env, root_ids, float(schedule["scout_time"]))
            self._preview_batch(env, root_ids)

            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            keep_ids = self._select_survivors(ranked, schedule)
            self._prune_except(env, root_ids, keep_ids)

            self._confirm_survivors(env, keep_ids, schedule)
            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            self._finish_selected_roots(env, ranked, schedule)
            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            self._refine_with_backward(env, ranked, schedule)
            ranked = self._ranked_previews(env)
            if not ranked:
                return self._answer(env)

            self._late_rechecks(env, ranked, schedule)
        except (BudgetExceededError, InvalidActionError):
            return self._answer(env)

        return self._answer(env)

    def _schedule(self, env: FlowTTSEnv, beta: float) -> dict[str, float | int]:
        grid_last = max(1, len(env.time_grid) - 1)

        if beta <= 0.0:
            return {
                "roots": 1,
                "scout_time": 1.0,
                "confirm_time": 1.0,
                "finish_time": 1.0,
                "branch_time": 0.85,
                "keep": 1,
                "finish_roots": 1,
                "branch_anchors": 0,
                "children": 0,
                "late_passes": 0,
                "score_gap": 0.02,
                "finish_gap": 0.02,
                "branch_gap": 0.01,
                "uncertainty": 0.45,
                "noise_policy": "inferred_noise",
                "noise_strength": 0.55,
            }

        scout_fraction = 0.20 + 0.10 * beta
        confirm_fraction = 0.52 + 0.18 * beta
        finish_fraction = 0.88 + 0.10 * beta

        roots = 2 + int(beta >= 0.35) + int(beta >= 0.65) + int(beta >= 0.90)
        keep = 1 + int(beta >= 0.22) + int(beta >= 0.60) + int(beta >= 0.88)
        finish_roots = 1 + int(beta >= 0.50) + int(beta >= 0.82)
        branch_anchors = int(beta >= 0.28) + int(beta >= 0.58) + int(beta >= 0.88)
        children = int(beta >= 0.28) + int(beta >= 0.72) + int(beta >= 0.92)
        late_passes = int(beta >= 0.58) + int(beta >= 0.86)

        branch_fraction = 0.82 - 0.18 * beta
        branch_fraction = max(0.58, min(branch_fraction, 0.86))
        noise_policy = "inferred_noise" if beta < 0.42 else "mixed_noise"
        noise_strength = 0.58 - 0.22 * beta

        return {
            "roots": roots,
            "scout_time": self._grid_time(env, self._grid_index(grid_last, scout_fraction)),
            "confirm_time": self._grid_time(env, self._grid_index(grid_last, confirm_fraction)),
            "finish_time": self._grid_time(env, self._grid_index(grid_last, finish_fraction)),
            "branch_time": self._grid_time(env, self._grid_index(grid_last, branch_fraction)),
            "keep": keep,
            "finish_roots": finish_roots,
            "branch_anchors": branch_anchors,
            "children": children,
            "late_passes": late_passes,
            "score_gap": 0.010 + 0.010 * beta,
            "finish_gap": 0.018 + 0.010 * beta,
            "branch_gap": 0.006 + 0.010 * beta,
            "uncertainty": 0.18 + 0.16 * beta,
            "noise_policy": noise_policy,
            "noise_strength": max(0.20, noise_strength),
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

    def _advance_particle(self, env: FlowTTSEnv, particle_id: int, target_time: float) -> None:
        state = env.get_state()
        particle = state.particles.get(particle_id)
        if particle is None or particle.status == "pruned":
            return

        current_time = float(particle.time)
        for grid_time in env.time_grid:
            grid_time = float(grid_time)
            if grid_time > current_time + 1e-9 and grid_time <= target_time + 1e-9:
                env.forward(particle_id, target_time=grid_time, solver="euler")
                current_time = grid_time

    def _preview_batch(self, env: FlowTTSEnv, particle_ids: list[int]) -> None:
        for particle_id in particle_ids:
            if env.budget_left <= 0:
                return
            try:
                env.preview(particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                return

    def _select_survivors(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> set[int]:
        if not ranked:
            return set()

        keep = int(schedule["keep"])
        gap = self._score_gap(ranked)
        top_uncertainty = float(ranked[0].uncertainty or 0.0)
        if gap <= float(schedule["score_gap"]):
            keep += 1
        if top_uncertainty >= float(schedule["uncertainty"]):
            keep += 1
        keep = min(len(ranked), max(1, keep))
        return {preview.particle_id for preview in ranked[:keep]}

    def _prune_except(self, env: FlowTTSEnv, particle_ids: list[int], keep_ids: set[int]) -> None:
        losers = [particle_id for particle_id in particle_ids if particle_id not in keep_ids]
        if not losers:
            return
        try:
            env.prune(losers)
        except InvalidActionError:
            return

    def _confirm_survivors(
        self,
        env: FlowTTSEnv,
        keep_ids: set[int],
        schedule: dict[str, float | int],
    ) -> None:
        target_time = float(schedule["confirm_time"])
        for particle_id in list(keep_ids):
            if env.budget_left <= 0:
                return
            state = env.get_state()
            particle = state.particles.get(particle_id)
            if particle is None or particle.status == "pruned":
                continue
            try:
                self._advance_particle(env, particle_id, target_time)
                if env.budget_left > 0:
                    env.preview(particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                continue

    def _finish_selected_roots(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> None:
        if not ranked:
            return

        finish_count = min(len(ranked), max(1, int(schedule["finish_roots"])))
        gap = self._score_gap(ranked)
        top_uncertainty = float(ranked[0].uncertainty or 0.0)
        if gap <= float(schedule["finish_gap"]):
            finish_count = min(len(ranked), finish_count + 1)
        if top_uncertainty >= float(schedule["uncertainty"]):
            finish_count = min(len(ranked), finish_count + 1)

        for preview in ranked[:finish_count]:
            if env.budget_left <= 0:
                return
            try:
                self._advance_particle(env, preview.particle_id, 1.0)
                if env.budget_left > 0:
                    env.preview(preview.particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                continue

    def _refine_with_backward(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> None:
        anchor_budget = int(schedule["branch_anchors"])
        child_budget = int(schedule["children"])
        if anchor_budget <= 0 or child_budget <= 0 or not ranked:
            return

        chosen = self._choose_branch_anchors(ranked, schedule, anchor_budget)
        target_time = float(schedule["branch_time"])
        finish_time = float(schedule["finish_time"])

        for index, anchor in enumerate(chosen):
            if env.budget_left <= 0:
                return

            num_children = child_budget
            if index > 0:
                num_children = max(1, num_children - 1)
            if (
                index == 0
                and len(ranked) > 1
                and self._score_gap(ranked) <= float(schedule["branch_gap"])
            ):
                num_children += 1

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

            for child_id in child_ids:
                if env.budget_left <= 0:
                    return
                try:
                    self._advance_particle(env, child_id, finish_time)
                    if env.budget_left > 0:
                        env.preview(child_id, mode="clean_anchor", scorer="default")
                except (BudgetExceededError, InvalidActionError):
                    return

            self._prune_clear_losers(env)

    def _late_rechecks(
        self,
        env: FlowTTSEnv,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
    ) -> None:
        passes = int(schedule["late_passes"])
        if passes <= 0 or not ranked:
            return

        for round_index in range(passes):
            current_ranked = self._ranked_previews(env)
            if not current_ranked or env.budget_left <= 0:
                return
            gap = self._score_gap(current_ranked)
            top_uncertainty = float(current_ranked[0].uncertainty or 0.0)
            if gap > float(schedule["finish_gap"]) and top_uncertainty < float(schedule["uncertainty"]):
                return

            preview = current_ranked[min(round_index, len(current_ranked) - 1)]
            try:
                if env.budget_left > 0:
                    env.preview(preview.particle_id, mode="clean_anchor", scorer="default")
            except (BudgetExceededError, InvalidActionError):
                return

    def _choose_branch_anchors(
        self,
        ranked: list[PreviewRecord],
        schedule: dict[str, float | int],
        anchor_budget: int,
    ) -> list[PreviewRecord]:
        if not ranked or anchor_budget <= 0:
            return []

        chosen = [ranked[0]]
        gap = self._score_gap(ranked)
        top_uncertainty = float(ranked[0].uncertainty or 0.0)

        if gap <= float(schedule["branch_gap"]) or top_uncertainty >= float(schedule["uncertainty"]):
            for preview in ranked[1:]:
                if len(chosen) >= anchor_budget:
                    break
                chosen.append(preview)

        return chosen[:anchor_budget]

    def _prune_clear_losers(self, env: FlowTTSEnv) -> None:
        ranked = self._ranked_previews(env)
        if len(ranked) < 3:
            return

        best_score = float(ranked[0].score or 0.0)
        losers = [
            preview.particle_id
            for preview in ranked[2:]
            if best_score - float(preview.score or 0.0) > 0.030
        ]
        if not losers:
            return
        try:
            env.prune(losers)
        except InvalidActionError:
            return

    def _ranked_previews(self, env: FlowTTSEnv) -> list[PreviewRecord]:
        state = env.get_state()
        previews_by_particle: dict[int, PreviewRecord] = {}
        for particle in state.particles.values():
            if particle.status == "pruned" or particle.last_preview_id is None:
                continue
            preview = state.previews.get(particle.last_preview_id)
            if preview is None or preview.score is None:
                continue
            previews_by_particle[particle.id] = preview
        return sorted(
            previews_by_particle.values(),
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
        return min(grid_last, max(1, int(round(grid_last * fraction))))

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
