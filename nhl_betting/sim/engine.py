from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .state import GameState, TeamState, PlayerState, Event
from .models import RateModels, TeamRates, PlayerRates


@dataclass
class SimConfig:
    periods: int = 3
    seconds_per_period: int = 20 * 60
    overtime_seconds: int = 5 * 60
    ot_enabled: bool = True
    shootout_enabled: bool = True
    seed: Optional[int] = None
    # Overdispersion controls: increase variance while preserving means
    # shots: multiplicative lognormal noise per segment
    dispersion_shots: float = 0.0
    # goals: logit-normal noise on per-shot conversion probability
    dispersion_goals: float = 0.0
    # Special teams multipliers for shots (base factors)
    pp_shots_mult: float = 1.4
    pk_shots_mult: float = 0.7
    # Additional goal conversion multipliers applied after PP/PK adjustments
    pp_goals_mult: float = 1.0
    pk_goals_mult: float = 1.0
    # Score-state effects mode for play-level simulation.
    # - dynamic: time-remaining + score-diff dependent multipliers (default)
    # - legacy: fixed +/-10% based on start-of-period score diff
    # - off: disable score-state multipliers (also disables empty-net heuristic)
    score_effects: str = "dynamic"
    # Goal event emission model.
    # - from_shots: decide goals as outcomes of individual shot events (so every goal is also a shot)
    # - independent: legacy mode (goals are sampled from team conversion, then attributed separately)
    goal_model: str = "from_shots"

    # Assist attribution model.
    # - onice: emit assist events at goal time, sampled from on-ice teammates (recommended)
    # - legacy: do not emit assist events in the period sim; rely on legacy fallback attribution
    # - off: disable assists entirely
    assist_model: str = "onice"

    # Player usage / TOI allocation model (for on-ice unit rotation).
    # - deterministic: proportional allocation from projected TOI weights (stable; legacy-ish)
    # - stochastic: sample segment allocations via multinomial from TOI weights (adds realistic variance)
    # - noisy: perturb TOI weights slightly per sim, then allocate deterministically (lower variance)
    usage_model: str = "deterministic"

    # Strength of perturbation used when usage_model == 'noisy'.
    # Interpreted as Gaussian noise std-dev applied to log-probabilities.
    usage_noisy_sigma: float = 0.18


class PossessionSimulator:
    """Placeholder for future possession-level simulation.

    Currently unused; will be extended to model discrete events (passes, shots, turnovers)
    conditioned on on-ice line combos and score effects.
    """
    def __init__(self, rates: TeamRates, rng: random.Random):
        self.rates = rates
        self.rng = rng


class PeriodSimulator:
    def __init__(self, cfg: SimConfig, rng: random.Random, np_rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.np_rng = np_rng

    def _poisson(self, lam: float) -> int:
        # Use a local numpy Generator (do not reseed global RNG)
        return int(self.np_rng.poisson(lam=max(lam, 1e-9)))

    def simulate_period(self, gs: GameState, rates: RateModels, period_idx: int) -> Tuple[int, int, List[Event]]:
        T = self.cfg.seconds_per_period
        # Convert per-60 rates to period totals (EV approximation)
        home_shots = self._poisson(rates.home.shots_per_60 * T / 3600.0)
        away_shots = self._poisson(rates.away.shots_per_60 * T / 3600.0)
        home_goals = self._poisson(rates.home.goals_per_60 * T / 3600.0)
        away_goals = self._poisson(rates.away.goals_per_60 * T / 3600.0)
        events: List[Event] = []
        # Distribute events uniformly in the period for now
        def _uniform_times(n: int) -> List[float]:
            return sorted([self.rng.random() * T for _ in range(n)])
        for t in _uniform_times(home_shots):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.home.name, kind="shot"))
        for t in _uniform_times(away_shots):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.away.name, kind="shot"))
        for t in _uniform_times(home_goals):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.home.name, kind="goal"))
        for t in _uniform_times(away_goals):
            events.append(Event(t=t + period_idx * T, period=period_idx + 1, team=gs.away.name, kind="goal"))
        return home_goals, away_goals, events

    def simulate_period_with_lines(
        self,
        gs: GameState,
        rates: RateModels,
        period_idx: int,
        lineup_home: List[Dict],
        lineup_away: List[Dict],
        st_home: Optional[Dict[str, float]] = None,
        st_away: Optional[Dict[str, float]] = None,
        special_teams_cal: Optional[Dict[str, float]] = None,
        period_seconds: Optional[int] = None,
    ) -> Tuple[int, int, List[Event]]:
        """Segment the period by EV lines and simulate events with simple score effects.

        Assumptions:
        - Rotate EV lines roughly equally across the period.
                - Score effects: trailing team shoots more and leading team shoots less, with strength
                    increasing late in games and with larger score differentials (capped to [0.6, 1.4]).
        - Goals drawn proportionally to shots.
        """
        T = int(period_seconds or self.cfg.seconds_per_period)

        usage_model = str(getattr(self.cfg, "usage_model", "deterministic") or "deterministic").strip().lower()
        if usage_model in ("legacy", "stable", "fixed"):
            usage_model = "deterministic"
        if usage_model in ("rand", "random", "stoch"):
            usage_model = "stochastic"
        if usage_model in ("noisy", "noise", "lite", "light", "stochastic_lite", "stochastic-lite", "stochastic_light", "stochastic-light"):
            usage_model = "noisy"
        if usage_model not in ("deterministic", "stochastic", "noisy"):
            usage_model = "deterministic"

        assist_model = str(getattr(self.cfg, "assist_model", "onice") or "onice").strip().lower()
        if assist_model in ("on_ice", "on-ice"):
            assist_model = "onice"
        if assist_model in ("none", "false", "0"):
            assist_model = "off"
        if assist_model not in ("onice", "legacy", "off"):
            assist_model = "onice"

        def _f(val: object, default: float) -> float:
            try:
                if val is None:
                    return float(default)
                return float(val)
            except Exception:
                return float(default)
        # Use more realistic shift-like segments to improve TOI realism.
        # Regulation: ~45s target; OT: slightly shorter segments.
        target_seg = 40.0 if T < self.cfg.seconds_per_period else 45.0
        segments = int(max(6, round(T / max(1.0, target_seg))))
        seg_len = T / max(1, segments)
        events: List[Event] = []
        home_goals = 0
        away_goals = 0
        # Small deterministic per-period pace bias to reduce uniformity across periods
        # Slightly boosts or reduces base pace for this period (applies to both sides)
        period_bias_table = [0.98, 1.03, 0.99, 1.00]
        try:
            period_pace_bias = float(period_bias_table[min(max(0, period_idx), len(period_bias_table)-1)])
        except Exception:
            period_pace_bias = 1.0
        # Build ordered line groups (L1->L4, D1->D3); fallback to roster
        def _line_order(lineup: List[Dict], prefix: str) -> List[List[int]]:
            df = lineup
            slots = [f"{prefix}{i}" for i in ([1,2,3,4] if prefix == "L" else [1,2,3])]
            out = []
            for s in slots:
                pids = [int(r.get("player_id")) for r in df if str(r.get("line_slot")) == s]
                if pids:
                    out.append(pids)
            return out

        def _dedupe_preserve_order(pids: List[int]) -> List[int]:
            seen = set()
            out: List[int] = []
            for pid in pids:
                try:
                    pid_i = int(pid)
                except Exception:
                    continue
                if pid_i in seen:
                    continue
                seen.add(pid_i)
                out.append(pid_i)
            return out
        def _fill_unit(
            pids: List[int],
            team: TeamState,
            target_n: int,
            prefer_f: int,
            prefer_d: int,
        ) -> List[int]:
            # Deduplicate while preserving order
            seen = set()
            uniq: List[int] = []
            for pid in (pids or []):
                try:
                    pid_i = int(pid)
                except Exception:
                    continue
                if pid_i in seen:
                    continue
                if pid_i not in team.players:
                    continue
                if str(team.players[pid_i].position) not in ("F", "D"):
                    continue
                seen.add(pid_i)
                uniq.append(pid_i)

            def _toi(pid: int) -> float:
                try:
                    return float(team.players[int(pid)].toi_proj or 0.0)
                except Exception:
                    return 0.0

            # Prefer a typical structure (e.g., PP: 3F+2D; PK: 2F+2D)
            f = [pid for pid in uniq if str(team.players[pid].position) == "F"]
            d = [pid for pid in uniq if str(team.players[pid].position) == "D"]
            f_sel = sorted(f, key=_toi, reverse=True)[: max(0, int(prefer_f))]
            d_sel = sorted(d, key=_toi, reverse=True)[: max(0, int(prefer_d))]
            unit = []
            for pid in (f_sel + d_sel):
                if pid not in unit:
                    unit.append(pid)
            # Fill remaining from the provided list by TOI
            if len(unit) < target_n:
                rest = [pid for pid in uniq if pid not in unit]
                rest = sorted(rest, key=_toi, reverse=True)
                for pid in rest:
                    unit.append(pid)
                    if len(unit) >= target_n:
                        break
            # If still short, fill from roster best-available by TOI, favoring missing positions
            if len(unit) < target_n:
                # Determine missing counts
                need_f = max(0, int(prefer_f) - sum(1 for pid in unit if str(team.players[pid].position) == "F"))
                need_d = max(0, int(prefer_d) - sum(1 for pid in unit if str(team.players[pid].position) == "D"))
                pool = [pid for pid, ps in team.players.items() if str(ps.position) in ("F", "D") and int(pid) not in unit]
                pool = sorted(pool, key=_toi, reverse=True)
                # First fill missing D then F (PK often needs D; PP needs D too)
                if need_d > 0:
                    for pid in pool:
                        if str(team.players[pid].position) == "D":
                            unit.append(pid)
                            need_d -= 1
                            if len(unit) >= target_n:
                                break
                if len(unit) < target_n and need_f > 0:
                    for pid in pool:
                        if pid in unit:
                            continue
                        if str(team.players[pid].position) == "F":
                            unit.append(pid)
                            need_f -= 1
                            if len(unit) >= target_n:
                                break
                # Finally fill anything
                if len(unit) < target_n:
                    for pid in pool:
                        if pid in unit:
                            continue
                        unit.append(pid)
                        if len(unit) >= target_n:
                            break
            return unit[:target_n]

        def _pp_units(lineup: List[Dict], team: TeamState) -> List[List[int]]:
            u1 = [int(r.get("player_id")) for r in lineup if r.get("pp_unit") == 1]
            u2 = [int(r.get("player_id")) for r in lineup if r.get("pp_unit") == 2]
            out: List[List[int]] = []
            if u1:
                out.append(_fill_unit(u1, team, target_n=5, prefer_f=3, prefer_d=2))
            if u2:
                out.append(_fill_unit(u2, team, target_n=5, prefer_f=3, prefer_d=2))
            return [u for u in out if u]

        def _pk_units(lineup: List[Dict], team: TeamState) -> List[List[int]]:
            u1 = [int(r.get("player_id")) for r in lineup if r.get("pk_unit") == 1]
            u2 = [int(r.get("player_id")) for r in lineup if r.get("pk_unit") == 2]
            out: List[List[int]] = []
            if u1:
                out.append(_fill_unit(u1, team, target_n=4, prefer_f=2, prefer_d=2))
            if u2:
                out.append(_fill_unit(u2, team, target_n=4, prefer_f=2, prefer_d=2))
            return [u for u in out if u]
        def _chunk(lst: List[int], size: int) -> List[List[int]]:
            return [lst[i:i+size] for i in range(0, len(lst), size)] or [[]]

        def _normalize_groups(groups: List[List[int]], size: int, max_groups: int) -> List[List[int]]:
            # Split any oversized groups into fixed-size chunks (e.g., if upstream data assigns
            # too many skaters to the same line slot). Preserve ordering and dedupe.
            out: List[List[int]] = []
            for grp in (groups or []):
                g = _dedupe_preserve_order([int(x) for x in (grp or [])])
                if not g:
                    continue
                out.extend(_chunk(g, size))
                if len(out) >= max_groups:
                    break
            out = [g for g in out if g]
            return out[:max_groups] if out else [[]]
        # Weighted selection helpers based on player on-ice stats/weights
        def _weighted_choice(pid_group: List[int], team: TeamState, kind: str) -> Optional[int]:
            if not pid_group:
                pid_group = []
            cands = []
            weights = []
            for pid in pid_group:
                ps = team.players.get(int(pid))
                if not ps:
                    continue
                cands.append(int(pid))
                if kind == "shot":
                    w = max(0.01, float(ps.shot_weight or 0.0))
                elif kind == "goal":
                    w = max(0.01, float(ps.goal_weight or (ps.shot_weight or 0.0) * 0.30))
                elif kind == "assist":
                    # Assist propensities are not modeled directly; approximate using a playmaking
                    # proxy from shot involvement, with a mild forward bias.
                    w = max(0.01, float(ps.shot_weight or 0.0))
                    try:
                        if str(ps.position) == "F":
                            w *= 1.10
                        elif str(ps.position) == "D":
                            w *= 0.90
                    except Exception:
                        pass
                elif kind == "block":
                    w = max(0.01, float(ps.block_weight or 0.0))
                else:
                    w = 1.0

                # IMPORTANT: on-ice selection already incorporates TOI via line rotation.
                # Using TOI-scaled weights here double-counts TOI and creates unrealistic
                # shot concentration. Convert to per-minute propensities.
                try:
                    toi = float(ps.toi_proj or 0.0)
                    if toi > 1e-6:
                        w = float(w) / toi
                except Exception:
                    pass
                weights.append(w)

            # Last-resort fallback: if the on-ice group contains no valid roster IDs (e.g.,
            # lineup data references unknown pids), fall back to the roster pool so we do not
            # emit unattributed events (player_id=None). Unattributed shots inflate goalie
            # saves (team shots) without appearing in skater SOG, biasing both.
            if not cands:
                if kind in {"shot", "goal"}:
                    pool = [int(pid) for pid, ps in team.players.items() if str(ps.position) in ("F", "D")]
                elif kind == "assist":
                    pool = [int(pid) for pid, ps in team.players.items() if str(ps.position) in ("F", "D")]
                elif kind == "block":
                    pool = [int(pid) for pid, ps in team.players.items() if str(ps.position) == "D"]
                else:
                    pool = [int(pid) for pid in team.players.keys()]
                if not pool:
                    return None
                cands = pool
                weights = []
                for pid in cands:
                    ps = team.players.get(int(pid))
                    if not ps:
                        continue
                    if kind == "shot":
                        w = max(0.01, float(ps.shot_weight or 0.0))
                    elif kind == "goal":
                        w = max(0.01, float(ps.goal_weight or (ps.shot_weight or 0.0) * 0.30))
                    elif kind == "assist":
                        w = max(0.01, float(ps.shot_weight or 0.0))
                        try:
                            if str(ps.position) == "F":
                                w *= 1.10
                            elif str(ps.position) == "D":
                                w *= 0.90
                        except Exception:
                            pass
                    elif kind == "block":
                        w = max(0.01, float(ps.block_weight or 0.0))
                    else:
                        w = 1.0
                    # Keep the same per-minute propensity adjustment as the on-ice path
                    try:
                        toi = float(ps.toi_proj or 0.0)
                        if toi > 1e-6:
                            w = float(w) / toi
                    except Exception:
                        pass
                    weights.append(w)

            # Normalize, smooth, and cap probabilities to prevent a single skater from soaking up
            # an implausible share of team shots in expectation.
            try:
                def _wf(x: object) -> float:
                    try:
                        if x is None:
                            return 1e-6
                        v = float(x)
                        if not np.isfinite(v):
                            return 1e-6
                        return max(1e-6, v)
                    except Exception:
                        return 1e-6

                w = np.array([_wf(x) for x in weights], dtype=float)
                # Flatten extremes (temperature-like) while preserving ordering
                w = np.power(w, 0.85)
                p = w / max(1e-12, float(w.sum()))
                # Mix in uniform mass to keep distribution realistic across a line
                alpha = 0.12
                p = (1.0 - alpha) * p + alpha * (1.0 / max(1, len(p)))
                # Cap the maximum share per event (line-level) then renormalize
                cap = 0.35
                if cap > 0:
                    p = np.minimum(p, cap)
                    p = p / max(1e-12, float(p.sum()))
                idx = int(self.np_rng.choice(np.arange(len(cands)), size=1, replace=True, p=p)[0])
                return cands[idx]
            except Exception:
                # Conservative fallback
                return cands[0]
        # Fallback lines from full roster when lineup slots are missing
        def _sorted_top(lst: List[Tuple[int, float]], k: int) -> List[int]:
            return [pid for pid, _ in sorted(lst, key=lambda x: x[1], reverse=True)[:k]]
        # Build fallback pools from roster with TOI ranking (12 F, 6 D)
        f_home = [(pid, float(p.toi_proj or 0.0)) for pid, p in gs.home.players.items() if str(p.position) in ("F", "C", "LW", "RW")]
        d_home_all = [(pid, float(p.toi_proj or 0.0)) for pid, p in gs.home.players.items() if str(p.position) == "D"]
        f_away = [(pid, float(p.toi_proj or 0.0)) for pid, p in gs.away.players.items() if str(p.position) in ("F", "C", "LW", "RW")]
        d_away_all = [(pid, float(p.toi_proj or 0.0)) for pid, p in gs.away.players.items() if str(p.position) == "D"]
        f_home_top = _sorted_top(f_home, 12); d_home_top = _sorted_top(d_home_all, 6)
        f_away_top = _sorted_top(f_away, 12); d_away_top = _sorted_top(d_away_all, 6)
        l_home = _line_order(lineup_home, "L") or _chunk(f_home_top, 3)
        d_home = _line_order(lineup_home, "D") or _chunk(d_home_top, 2)
        l_away = _line_order(lineup_away, "L") or _chunk(f_away_top, 3)
        d_away = _line_order(lineup_away, "D") or _chunk(d_away_top, 2)
        # Ensure we have true 3F lines and 2D pairs even when lineup slot data is messy.
        l_home = _normalize_groups(l_home, size=3, max_groups=4)
        d_home = _normalize_groups(d_home, size=2, max_groups=3)
        l_away = _normalize_groups(l_away, size=3, max_groups=4)
        d_away = _normalize_groups(d_away, size=2, max_groups=3)

        # Sanitize EV groups against the roster so on-ice selection never references unknown pids.
        # Forward lines should only contain forwards; D pairs should only contain defensemen.
        def _toi(team: TeamState, pid: int) -> float:
            try:
                return float(team.players[int(pid)].toi_proj or 0.0)
            except Exception:
                return 0.0

        def _fill_forwards(group: List[int], team: TeamState, n: int) -> List[int]:
            uniq = []
            seen = set()
            for pid in (group or []):
                try:
                    pid_i = int(pid)
                except Exception:
                    continue
                if pid_i in seen:
                    continue
                ps = team.players.get(pid_i)
                if not ps:
                    continue
                if str(ps.position) != "F":
                    continue
                seen.add(pid_i)
                uniq.append(pid_i)
            uniq = sorted(uniq, key=lambda p: _toi(team, p), reverse=True)
            unit = uniq[:n]
            if len(unit) < n:
                pool = [int(pid) for pid, ps in team.players.items() if str(ps.position) == "F" and int(pid) not in unit]
                pool = sorted(pool, key=lambda p: _toi(team, p), reverse=True)
                for pid in pool:
                    unit.append(pid)
                    if len(unit) >= n:
                        break
            return unit[:n]

        def _fill_defense(group: List[int], team: TeamState, n: int) -> List[int]:
            uniq = []
            seen = set()
            for pid in (group or []):
                try:
                    pid_i = int(pid)
                except Exception:
                    continue
                if pid_i in seen:
                    continue
                ps = team.players.get(pid_i)
                if not ps:
                    continue
                if str(ps.position) != "D":
                    continue
                seen.add(pid_i)
                uniq.append(pid_i)
            uniq = sorted(uniq, key=lambda p: _toi(team, p), reverse=True)
            unit = uniq[:n]
            if len(unit) < n:
                pool = [int(pid) for pid, ps in team.players.items() if str(ps.position) == "D" and int(pid) not in unit]
                pool = sorted(pool, key=lambda p: _toi(team, p), reverse=True)
                for pid in pool:
                    unit.append(pid)
                    if len(unit) >= n:
                        break
            return unit[:n]

        l_home = [_fill_forwards(g, gs.home, 3) for g in (l_home or [])]
        l_away = [_fill_forwards(g, gs.away, 3) for g in (l_away or [])]
        d_home = [_fill_defense(g, gs.home, 2) for g in (d_home or [])]
        d_away = [_fill_defense(g, gs.away, 2) for g in (d_away or [])]
        # Rotation sequences weighted by projected TOI
        def _line_weights(pid_groups: List[List[int]], team: TeamState) -> List[float]:
            vals = []
            for grp in pid_groups:
                if grp:
                    avg_toi = sum([max(0.0, float(team.players.get(int(pid)).toi_proj or 0.0)) for pid in grp if team.players.get(int(pid))]) / max(1, len(grp))
                else:
                    avg_toi = 15.0
                vals.append(max(0.1, avg_toi))
            s = sum(vals) or 1.0
            return [v/s for v in vals]
        def _alloc_indices(weights: List[float], n: int) -> List[int]:
            # Proportional allocation of n slots to indices by weights
            if not weights:
                return [0]*n
            raw = [w * n for w in weights]
            base = [int(x) for x in raw]
            rem = n - sum(base)
            frac = sorted([(i, (raw[i] - base[i])) for i in range(len(weights))], key=lambda x: x[1], reverse=True)
            for k in range(rem):
                base[frac[k % len(frac)][0]] += 1
            seq = []
            for i, count in enumerate(base):
                seq.extend([i]*int(count))
            # Interleave to avoid long runs of the same line
            out = []
            for i in range(n):
                out.append(seq[i % len(seq)])
            return out

        def _alloc_indices_stochastic(weights: List[float], n: int) -> List[int]:
            # Sample allocation of n slots to indices by weights (adds variance but preserves means)
            if n <= 0:
                return []
            if not weights:
                return [0] * n
            try:
                w = np.array([max(0.0, float(x)) for x in weights], dtype=float)
                s = float(w.sum())
                if s <= 0:
                    return [0] * n
                p = w / s
                counts = self.np_rng.multinomial(n=n, pvals=p)
                seq: List[int] = []
                for i, c in enumerate(counts.tolist()):
                    seq.extend([int(i)] * int(c))
                if len(seq) < n:
                    seq.extend([0] * int(n - len(seq)))
                if len(seq) > n:
                    seq = seq[:n]
                # Shuffle to avoid deterministic ordering
                perm = self.np_rng.permutation(int(n)).tolist()
                return [seq[int(j)] for j in perm]
            except Exception:
                return _alloc_indices(weights, n)

        def _alloc_indices_noisy(weights: List[float], n: int) -> List[int]:
            # Lower-variance stochasticity: perturb weights slightly, then allocate deterministically.
            # This keeps TOI shares close to expected while introducing realistic variation.
            if n <= 0:
                return []
            if not weights:
                return [0] * n
            try:
                w = np.array([max(0.0, float(x)) for x in weights], dtype=float)
                s = float(w.sum())
                if s <= 0.0:
                    return [0] * n
                p = w / s
                # Perturb log-probabilities with small Gaussian noise.
                try:
                    sigma = float(getattr(self.cfg, "usage_noisy_sigma", 0.18) or 0.18)
                except Exception:
                    sigma = 0.18
                sigma = float(max(0.0, min(2.0, sigma)))
                noise = self.np_rng.normal(loc=0.0, scale=float(sigma), size=int(len(p)))
                p2 = np.exp(np.log(p + 1e-12) + noise)
                p2 = p2 / max(1e-12, float(p2.sum()))
                return _alloc_indices([float(x) for x in p2.tolist()], n)
            except Exception:
                return _alloc_indices(weights, n)

        if usage_model == "stochastic":
            alloc_fn = _alloc_indices_stochastic
        elif usage_model == "noisy":
            alloc_fn = _alloc_indices_noisy
        else:
            alloc_fn = _alloc_indices
        w_lh = _line_weights(l_home, gs.home); w_dh = _line_weights(d_home, gs.home)
        w_la = _line_weights(l_away, gs.away); w_da = _line_weights(d_away, gs.away)
        rot_lh = alloc_fn(w_lh, segments) if l_home else []
        rot_dh = alloc_fn(w_dh, segments) if d_home else []
        rot_la = alloc_fn(w_la, segments) if l_away else []
        rot_da = alloc_fn(w_da, segments) if d_away else []
        idx_lh = 0; idx_da = 0; idx_la = 0; idx_dh = 0
        # Special teams units
        pp_home = _pp_units(lineup_home, gs.home)
        pk_home = _pk_units(lineup_home, gs.home)
        pp_away = _pp_units(lineup_away, gs.away)
        pk_away = _pk_units(lineup_away, gs.away)
        idx_pp_h = 0; idx_pk_h = 0; idx_pp_a = 0; idx_pk_a = 0
        # Special teams parameters
        st_home = st_home or {"pp_pct": 0.2, "pk_pct": 0.8, "drawn_per_game": 3.0, "committed_per_game": 3.0}
        st_away = st_away or {"pp_pct": 0.2, "pk_pct": 0.8, "drawn_per_game": 3.0, "committed_per_game": 3.0}
        # Calibration multipliers (default to neutral 1.0)
        cal_pp_sh_mult = _f((special_teams_cal or {}).get("pp_shot_multiplier", 1.0), 1.0)
        cal_pk_sh_mult = _f((special_teams_cal or {}).get("pk_shot_multiplier", 1.0), 1.0)
        cal_pp_gl_mult = _f((special_teams_cal or {}).get("pp_goal_multiplier", 1.0), 1.0)
        cal_pk_gl_mult = _f((special_teams_cal or {}).get("pk_goal_multiplier", 1.0), 1.0)
        # Combined PP intensity from penalty rates.
        # Use committed rates to avoid double-counting (drawn and committed are the same events).
        # Approximate total PP time as: minors_per_game * 120s, then convert to fraction of game time.
        h_comm = _f(st_home.get("committed_per_game", 3.0), 3.0)
        a_comm = _f(st_away.get("committed_per_game", 3.0), 3.0)
        try:
            # Each committed minor yields ~2 minutes of PP time for the opponent.
            # `h_comm`/`a_comm` are per-game rates, so convert to a fraction of *game* time.
            # This function runs once per period; sampling PP segments using per-period seconds
            # would overstate PP/PK time by ~3x.
            pp_seconds_total = max(0.0, float(h_comm + a_comm)) * 120.0
            # Regulation game seconds used for per-period sampling.
            reg_game_seconds = float(max(1.0, float(self.cfg.periods) * float(self.cfg.seconds_per_period)))
            denom_seconds = reg_game_seconds
            # In OT, treat the OT window as its own clock.
            if int(T) != int(self.cfg.seconds_per_period):
                denom_seconds = float(max(1.0, float(T)))
            pp_frac_total = max(0.0, min(0.45, float(pp_seconds_total) / float(denom_seconds)))
        except Exception:
            # Fallback: convert minutes of PP per game into a fraction of regulation time.
            try:
                reg_game_seconds = float(max(1.0, float(self.cfg.periods) * float(self.cfg.seconds_per_period)))
                pp_frac_total = max(0.0, min(0.45, (float(h_comm + a_comm) * 120.0) / reg_game_seconds))
            except Exception:
                pp_frac_total = 0.18

        # Side weighting: home PP occurs when away commits; away PP occurs when home commits.
        denom = max(1e-6, float(h_comm + a_comm))
        home_pp_prob = float(a_comm) / denom

        # Pre-sample segment strengths so EV rotation can be allocated over EV time only.
        seg_home_pp_flags: List[bool] = []
        seg_away_pp_flags: List[bool] = []
        for _k in range(segments):
            r_seg = self.rng.random()
            if r_seg < pp_frac_total:
                is_home = (self.rng.random() < home_pp_prob)
                seg_home_pp_flags.append(bool(is_home))
                seg_away_pp_flags.append(bool(not is_home))
            else:
                seg_home_pp_flags.append(False)
                seg_away_pp_flags.append(False)
        ev_seg_count = sum(1 for _k in range(segments) if (not seg_home_pp_flags[_k] and not seg_away_pp_flags[_k]))
        rot_lh_ev = alloc_fn(w_lh, ev_seg_count) if (l_home and ev_seg_count > 0) else []
        rot_dh_ev = alloc_fn(w_dh, ev_seg_count) if (d_home and ev_seg_count > 0) else []
        rot_la_ev = alloc_fn(w_la, ev_seg_count) if (l_away and ev_seg_count > 0) else []
        rot_da_ev = alloc_fn(w_da, ev_seg_count) if (d_away and ev_seg_count > 0) else []
        ev_ptr = 0

        def _sample_index(weights: List[float]) -> int:
            if not weights:
                return 0
            try:
                def _wf0(x: object) -> float:
                    try:
                        if x is None:
                            return 0.0
                        v = float(x)
                        if not np.isfinite(v):
                            return 0.0
                        return max(0.0, v)
                    except Exception:
                        return 0.0

                w = np.array([_wf0(x) for x in weights], dtype=float)
                s = float(w.sum())
                if s <= 0.0:
                    return 0
                p = w / s
                return int(self.np_rng.choice(np.arange(len(p)), size=1, replace=True, p=p)[0])
            except Exception:
                return 0

        def _pick_unit(units: List[List[int]], p0: float) -> List[int]:
            if not units:
                return []
            if len(units) == 1:
                return units[0]
            try:
                p0 = float(max(0.0, min(1.0, p0)))
            except Exception:
                p0 = 0.65
            rest = max(1, int(len(units) - 1))
            p_rest = (1.0 - p0) / float(rest)
            probs = [p0] + [p_rest] * rest
            try:
                idx = int(self.np_rng.choice(np.arange(len(units)), size=1, replace=True, p=np.array(probs, dtype=float) / float(sum(probs)))[0])
            except Exception:
                idx = 0 if (self.rng.random() < p0) else 1
            return units[int(idx) % len(units)]

        def _score_effect_delta(diff: int, game_seconds_remaining: float) -> float:
            """Return a signed shot-rate delta for the HOME team.

            Positive means home shoots more (home trailing), negative means home shoots less (home leading).
            """
            if diff == 0:
                return 0.0
            # Urgency increases as the game winds down.
            # 0.0 when >30 minutes left, 1.0 when <=5 minutes left.
            urg = (1800.0 - float(game_seconds_remaining)) / 1500.0
            urg = max(0.0, min(1.0, float(urg)))
            abs_diff = abs(int(diff))
            if abs_diff <= 1:
                low, high = 0.04, 0.12
            else:
                low, high = 0.07, 0.18
            mag = float(low + (high - low) * urg)
            # Home leading => negative delta for home. Home trailing => positive delta for home.
            return -mag if diff > 0 else mag

        score_mode = str(getattr(self.cfg, "score_effects", "dynamic") or "dynamic").strip().lower()
        if score_mode in {"none", "false", "0"}:
            score_mode = "off"

        goal_model = str(getattr(self.cfg, "goal_model", "from_shots") or "from_shots").strip().lower()
        if goal_model in {"legacy", "old", "independent", "ind"}:
            goal_model = "independent"
        elif goal_model in {"fromshots", "from_shots", "shots", "shot", "linked"}:
            goal_model = "from_shots"
        else:
            goal_model = "from_shots"

        def _finishing_mult(team: TeamState, ice: List[int], shooter_pid: Optional[int]) -> float:
            """Per-shot finishing multiplier based on goal_weight/shot_weight.

            Normalized to mean ~1.0 for the current on-ice group so team-level conversion stays anchored.
            """
            if shooter_pid is None:
                return 1.0
            try:
                shooter = team.players.get(int(shooter_pid))
            except Exception:
                shooter = None
            if not shooter:
                return 1.0

            def _ratio(pid: int) -> float:
                ps = team.players.get(int(pid))
                if not ps:
                    return 1.0
                try:
                    gw = float(ps.goal_weight or 0.0)
                    sw = float(ps.shot_weight or 0.0)
                    r = gw / max(1e-6, sw)
                except Exception:
                    r = 1.0
                return float(max(0.05, min(3.0, r)))

            try:
                shooter_r = _ratio(int(shooter_pid))
                ice_rs = [_ratio(int(pid)) for pid in (ice or []) if int(pid) in team.players]
                mean_r = float(sum(ice_rs) / max(1, len(ice_rs))) if ice_rs else shooter_r
                mult = shooter_r / max(1e-6, mean_r)
                return float(max(0.25, min(3.0, mult)))
            except Exception:
                return 1.0

        def _emit_assists(
            team: TeamState,
            ice: List[int],
            scorer_pid: Optional[int],
            t_goal: float,
            period_num: int,
            team_name: str,
            strength: str,
        ) -> None:
            # Typical NHL: most goals have at least one assist, and many have two.
            # Use a simple model: primary assist with prob p_primary; conditional secondary
            # assist with prob p_secondary_given_primary.
            p_primary = 0.85
            p_secondary_given_primary = 0.90
            if scorer_pid is None:
                return
            teammates = [int(pid) for pid in (ice or []) if pid is not None and int(pid) != int(scorer_pid)]
            if not teammates:
                return
            primary_pid: Optional[int] = None
            if self.rng.random() < float(p_primary):
                primary_pid = _weighted_choice(teammates, team, "assist")
                if primary_pid is not None:
                    events.append(
                        Event(
                            t=t_goal,
                            period=int(period_num),
                            team=str(team_name),
                            kind="assist",
                            player_id=int(primary_pid),
                            meta={"strength": strength, "type": "primary"},
                        )
                    )
            if primary_pid is not None and self.rng.random() < float(p_secondary_given_primary):
                pool = [pid for pid in teammates if int(pid) != int(primary_pid)]
                if pool:
                    secondary_pid = _weighted_choice(pool, team, "assist")
                    if secondary_pid is not None:
                        events.append(
                            Event(
                                t=t_goal,
                                period=int(period_num),
                                team=str(team_name),
                                kind="assist",
                                player_id=int(secondary_pid),
                                meta={"strength": strength, "type": "secondary"},
                            )
                        )

        # Simulate each segment
        for k in range(segments):
            time_elapsed = k * seg_len
            time_remaining = T - time_elapsed
            # score effects factor
            if score_mode == "dynamic":
                sim_home_score = int(gs.home.score) + int(home_goals)
                sim_away_score = int(gs.away.score) + int(away_goals)
                diff = int(sim_home_score - sim_away_score)
            elif score_mode == "legacy":
                diff = int(gs.home.score) - int(gs.away.score)
            else:
                diff = 0
            # Approximate game time remaining for regulation periods.
            if int(period_seconds or self.cfg.seconds_per_period) == int(self.cfg.seconds_per_period):
                game_seconds_remaining = float((max(0, int(self.cfg.periods) - 1 - int(period_idx)) * int(self.cfg.seconds_per_period)) + time_remaining)
            else:
                game_seconds_remaining = float(time_remaining)

            if score_mode == "dynamic":
                score_delta_home = _score_effect_delta(diff, game_seconds_remaining)
            elif score_mode == "legacy":
                if diff > 0:
                    score_delta_home = -0.10
                elif diff < 0:
                    score_delta_home = 0.10
                else:
                    score_delta_home = 0.0
            else:
                score_delta_home = 0.0

            home_factor = max(0.6, min(1.4, 1.0 + float(score_delta_home)))
            away_factor = max(0.6, min(1.4, 1.0 - float(score_delta_home)))
            # PP/PK effects: use pre-sampled segment strengths
            seg_is_home_pp = bool(seg_home_pp_flags[k])
            seg_is_away_pp = bool(seg_away_pp_flags[k])
            # Base PP/PK shot factors further scaled by calibration
            pp_mult_shots = float(self.cfg.pp_shots_mult) * cal_pp_sh_mult
            pk_mult_shots = float(self.cfg.pk_shots_mult) * cal_pk_sh_mult
            if seg_is_home_pp:
                home_factor *= pp_mult_shots
                away_factor *= pk_mult_shots
            elif seg_is_away_pp:
                away_factor *= pp_mult_shots
                home_factor *= pk_mult_shots
            # Empty-net end-game effects: trailing team increases shot rate; leading team more likely to score into empty net
            # Simple heuristic: apply in final 3 minutes of P3 when down 1, and final 2 minutes when down >=2
            if period_idx == 2:  # third period (0-indexed)
                empty_net_p = 0.18
                two_goal_scale = 0.30
                if time_remaining <= 180:  # last 3 minutes
                    if diff < 0:  # home trailing
                        home_factor *= 1.20
                        # away empty-net chance
                        p_empty_mult = 1.0 + empty_net_p + (two_goal_scale if diff <= -2 and time_remaining <= 120 else 0.0)
                    elif diff > 0:  # away trailing
                        away_factor *= 1.20
                        p_empty_mult = 1.0 + empty_net_p + (two_goal_scale if diff >= 2 and time_remaining <= 120 else 0.0)
                    else:
                        p_empty_mult = 1.0
                else:
                    p_empty_mult = 1.0
            else:
                p_empty_mult = 1.0
            # Expected shots in segment
            lam_h = max(1e-6, rates.home.shots_per_60 * seg_len / 3600.0 * home_factor * period_pace_bias)
            lam_a = max(1e-6, rates.away.shots_per_60 * seg_len / 3600.0 * away_factor * period_pace_bias)
            # Apply overdispersion via lognormal multiplicative noise
            if float(self.cfg.dispersion_shots or 0.0) > 0.0:
                try:
                    sigma = float(self.cfg.dispersion_shots)
                    lam_h = max(1e-6, float(lam_h) * float(self.np_rng.lognormal(mean=0.0, sigma=sigma)))
                    lam_a = max(1e-6, float(lam_a) * float(self.np_rng.lognormal(mean=0.0, sigma=sigma)))
                except Exception:
                    pass
            sh_h = int(self.np_rng.poisson(lam_h))
            sh_a = int(self.np_rng.poisson(lam_a))
            # Goals proportional to shots vs base conversion
            p_goal_home = (rates.home.goals_per_60 / max(rates.home.shots_per_60, 1e-3))
            p_goal_away = (rates.away.goals_per_60 / max(rates.away.shots_per_60, 1e-3))
            if seg_is_home_pp:
                p_goal_home = p_goal_home * (1.0 + 1.5 * _f(st_home.get("pp_pct", 0.2), 0.2)) * cal_pp_gl_mult
                p_goal_home = min(0.45, p_goal_home)
                p_goal_away = p_goal_away * (0.9 * _f(st_away.get("pk_pct", 0.8), 0.8)) * cal_pk_gl_mult
                p_goal_away = max(0.01, p_goal_away)
            elif seg_is_away_pp:
                p_goal_away = p_goal_away * (1.0 + 1.5 * _f(st_away.get("pp_pct", 0.2), 0.2)) * cal_pp_gl_mult
                p_goal_away = min(0.45, p_goal_away)
                p_goal_home = p_goal_home * (0.9 * _f(st_home.get("pk_pct", 0.8), 0.8)) * cal_pk_gl_mult
                p_goal_home = max(0.01, p_goal_home)
            # Apply additional PP/PK goal conversion multipliers from config
            if seg_is_home_pp:
                p_goal_home *= float(self.cfg.pp_goals_mult)
                p_goal_away *= float(self.cfg.pk_goals_mult)
            elif seg_is_away_pp:
                p_goal_away *= float(self.cfg.pp_goals_mult)
                p_goal_home *= float(self.cfg.pk_goals_mult)
            # Apply empty-net multiplier to the leading team's conversion, if any
            if period_idx == 2 and p_empty_mult != 1.0:
                if diff < 0:  # away leading
                    p_goal_away = min(0.65, p_goal_away * p_empty_mult)
                elif diff > 0:  # home leading
                    p_goal_home = min(0.65, p_goal_home * p_empty_mult)
            # Apply goals overdispersion via logit-normal noise
            if float(self.cfg.dispersion_goals or 0.0) > 0.0:
                try:
                    sigma_g = float(self.cfg.dispersion_goals)
                    def _perturb_prob(p: float) -> float:
                        p = max(1e-4, min(0.99, float(p)))
                        logit = math.log(p / (1.0 - p))
                        noisy = logit + float(self.np_rng.normal(loc=0.0, scale=sigma_g))
                        val = 1.0 / (1.0 + math.exp(-noisy))
                        return max(0.005, min(0.95, float(val)))
                    p_goal_home = _perturb_prob(p_goal_home)
                    p_goal_away = _perturb_prob(p_goal_away)
                except Exception:
                    pass
            if goal_model == "independent":
                g_h = sum(1 for _ in range(sh_h) if self.rng.random() < p_goal_home)
                g_a = sum(1 for _ in range(sh_a) if self.rng.random() < p_goal_away)
            else:
                g_h = 0
                g_a = 0
            # Times in segment window
            t0 = k * seg_len + period_idx * T
            # Determine on-ice groups for attribution
            strength_h = "PP" if seg_is_home_pp else ("PK" if seg_is_away_pp else "EV")
            strength_a = "PP" if seg_is_away_pp else ("PK" if seg_is_home_pp else "EV")
            if seg_is_home_pp:
                # PP usage: prefer unit 1, but allow stochastic mixing
                if pp_home:
                    if usage_model == "stochastic":
                        ice_h = _pick_unit(pp_home, p0=0.72)
                    else:
                        if len(pp_home) > 1 and self.rng.random() < 0.65:
                            ice_h = pp_home[0]
                        else:
                            ice_h = pp_home[idx_pp_h % len(pp_home)]
                            idx_pp_h = (idx_pp_h + 1) % max(1, len(pp_home))
                else:
                    # Fallback to EV group if PP units unknown
                    ih = _sample_index(w_lh) if l_home else 0
                    idh = _sample_index(w_dh) if d_home else 0
                    ice_h = ((l_home[ih] if l_home else []) + (d_home[idh] if d_home else []))
                # Away team on PK if units available
                if pk_away:
                    if usage_model == "stochastic":
                        ice_a = _pick_unit(pk_away, p0=0.66)
                    else:
                        if len(pk_away) > 1 and self.rng.random() < 0.60:
                            ice_a = pk_away[0]
                        else:
                            ice_a = pk_away[idx_pk_a % len(pk_away)]
                            idx_pk_a = (idx_pk_a + 1) % max(1, len(pk_away))
                else:
                    ia = _sample_index(w_la) if l_away else 0
                    ida = _sample_index(w_da) if d_away else 0
                    ice_a = ((l_away[ia] if l_away else []) + (d_away[ida] if d_away else []))
                # Enforce typical skater counts: PP ~5, PK ~4
                ice_h = (ice_h or [])[:5]
                ice_a = (ice_a or [])[:4]
            elif seg_is_away_pp and pp_away:
                # PP usage: prefer unit 1, but allow stochastic mixing
                if pp_away:
                    if usage_model == "stochastic":
                        ice_a = _pick_unit(pp_away, p0=0.72)
                    else:
                        if len(pp_away) > 1 and self.rng.random() < 0.65:
                            ice_a = pp_away[0]
                        else:
                            ice_a = pp_away[idx_pp_a % len(pp_away)]
                            idx_pp_a = (idx_pp_a + 1) % max(1, len(pp_away))
                else:
                    ia = _sample_index(w_la) if l_away else 0
                    ida = _sample_index(w_da) if d_away else 0
                    ice_a = ((l_away[ia] if l_away else []) + (d_away[ida] if d_away else []))
                # Home team on PK if units available
                if pk_home:
                    if usage_model == "stochastic":
                        ice_h = _pick_unit(pk_home, p0=0.66)
                    else:
                        if len(pk_home) > 1 and self.rng.random() < 0.60:
                            ice_h = pk_home[0]
                        else:
                            ice_h = pk_home[idx_pk_h % len(pk_home)]
                            idx_pk_h = (idx_pk_h + 1) % max(1, len(pk_home))
                else:
                    ih = _sample_index(w_lh) if l_home else 0
                    idh = _sample_index(w_dh) if d_home else 0
                    ice_h = ((l_home[ih] if l_home else []) + (d_home[idh] if d_home else []))
                # Enforce typical skater counts: PP ~5, PK ~4
                ice_a = (ice_a or [])[:5]
                ice_h = (ice_h or [])[:4]
            else:
                # EV: 5 skaters or 3v3 in OT
                is_ot = (T != self.cfg.seconds_per_period)
                if not is_ot:
                    # Allocate EV rotation over EV segments only (avoids PP-skipping bias).
                    ih = rot_lh_ev[ev_ptr % max(1, len(rot_lh_ev) or 1)] if rot_lh_ev else 0
                    idh = rot_dh_ev[ev_ptr % max(1, len(rot_dh_ev) or 1)] if rot_dh_ev else 0
                    ia = rot_la_ev[ev_ptr % max(1, len(rot_la_ev) or 1)] if rot_la_ev else 0
                    ida = rot_da_ev[ev_ptr % max(1, len(rot_da_ev) or 1)] if rot_da_ev else 0
                    ice_h = ((l_home[ih] if l_home else []) + (d_home[idh] if d_home else []))
                    ice_a = ((l_away[ia] if l_away else []) + (d_away[ida] if d_away else []))
                    # cap to 5 skaters
                    ice_h = (ice_h or [])[:5]
                    ice_a = (ice_a or [])[:5]
                    ev_ptr += 1
                else:
                    # OT 3v3: select 2F + 1D from top pools
                    def _pick_3v3(lines_f: List[List[int]], lines_d: List[List[int]], idx_f: int, idx_d: int) -> Tuple[List[int], int, int]:
                        f_group = lines_f[idx_f % max(1, len(lines_f))] if lines_f else []
                        d_group = lines_d[idx_d % max(1, len(lines_d))] if lines_d else []
                        idx_f = (idx_f + 1) % max(1, len(lines_f))
                        idx_d = (idx_d + 1) % max(1, len(lines_d))
                        f_sel = (f_group or [])[:2]
                        d_sel = (d_group or [])[:1]
                        return (f_sel + d_sel), idx_f, idx_d
                    ice_h, idx_lh, idx_dh = _pick_3v3(l_home, d_home, idx_lh, idx_dh)
                    ice_a, idx_la, idx_da = _pick_3v3(l_away, d_away, idx_la, idx_da)
                # EV rotation pointers are handled via ev_ptr; do not advance idx_* here.
            # Emit shift events for TOI attribution for all players on ice.
            # NOTE: `segments` is a coarse rotation count per period; the selected units are assumed to
            # be on ice for the full segment duration. Using a fractional multiplier here severely
            # undercounts TOI and breaks TOI/stat consistency.
            dur_h = float(seg_len)
            dur_a = float(seg_len)
            for pid in (ice_h or []):
                events.append(Event(t=t0, period=period_idx + 1, team=gs.home.name, kind="shift", player_id=pid, meta={"dur": dur_h, "strength": strength_h}))
            for pid in (ice_a or []):
                events.append(Event(t=t0, period=period_idx + 1, team=gs.away.name, kind="shift", player_id=pid, meta={"dur": dur_a, "strength": strength_a}))
            # Goalies are on ice entire segment; ensure TOI is accrued
            def _starter_goalie(team: TeamState) -> Optional[int]:
                goalies = [p for p in team.players.values() if str(p.position) == "G"]
                if not goalies:
                    return None
                return int(max(goalies, key=lambda p: float(p.toi_proj or 0.0)).player_id)
            g_home_id = _starter_goalie(gs.home)
            g_away_id = _starter_goalie(gs.away)
            if g_home_id is not None:
                events.append(Event(t=t0, period=period_idx + 1, team=gs.home.name, kind="shift", player_id=g_home_id, meta={"dur": seg_len, "strength": strength_h}))
            if g_away_id is not None:
                events.append(Event(t=t0, period=period_idx + 1, team=gs.away.name, kind="shift", player_id=g_away_id, meta={"dur": seg_len, "strength": strength_a}))
            # Attribute shots (and optionally goals) to players on ice
            for _ in range(sh_h):
                pid = _weighted_choice(ice_h, gs.home, "shot") if ice_h else None
                t_sh = t0 + self.rng.random() * seg_len
                events.append(Event(t=t_sh, period=period_idx + 1, team=gs.home.name, kind="shot", player_id=pid, meta={"strength": strength_h}))
                if goal_model != "independent":
                    p = float(p_goal_home)
                    if pid is not None:
                        p *= float(_finishing_mult(gs.home, ice_h or [], int(pid)))
                    p = float(max(0.0005, min(0.95, p)))
                    if self.rng.random() < p:
                        g_h += 1
                        events.append(Event(t=t_sh, period=period_idx + 1, team=gs.home.name, kind="goal", player_id=pid, meta={"strength": strength_h}))
                        if assist_model == "onice":
                            _emit_assists(gs.home, ice_h or [], pid, t_sh, period_idx + 1, gs.home.name, strength_h)

            for _ in range(sh_a):
                pid = _weighted_choice(ice_a, gs.away, "shot") if ice_a else None
                t_sh = t0 + self.rng.random() * seg_len
                events.append(Event(t=t_sh, period=period_idx + 1, team=gs.away.name, kind="shot", player_id=pid, meta={"strength": strength_a}))
                if goal_model != "independent":
                    p = float(p_goal_away)
                    if pid is not None:
                        p *= float(_finishing_mult(gs.away, ice_a or [], int(pid)))
                    p = float(max(0.0005, min(0.95, p)))
                    if self.rng.random() < p:
                        g_a += 1
                        events.append(Event(t=t_sh, period=period_idx + 1, team=gs.away.name, kind="goal", player_id=pid, meta={"strength": strength_a}))
                        if assist_model == "onice":
                            _emit_assists(gs.away, ice_a or [], pid, t_sh, period_idx + 1, gs.away.name, strength_a)

            # Legacy independent goal attribution
            if goal_model == "independent":
                for _ in range(g_h):
                    pid = _weighted_choice(ice_h, gs.home, "goal") if ice_h else None
                    events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.home.name, kind="goal", player_id=pid, meta={"strength": strength_h}))
                for _ in range(g_a):
                    pid = _weighted_choice(ice_a, gs.away, "goal") if ice_a else None
                    events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.away.name, kind="goal", player_id=pid, meta={"strength": strength_a}))

            home_goals += int(g_h)
            away_goals += int(g_a)
            # Attribute blocks to defending players on ice (approximate fraction of opponent SOG)
            # Higher block rate on PK segments vs EV; lower on PP defending side
            # NOTE: A blocked shot is recorded on a shot attempt, not a shot on goal.
            # Our sim's "shot" event is closer to SOG (used for props), so a higher default
            # block probability is needed to match observed per-game block totals when
            # no external calibration is supplied.
            p_block_ev = _f((special_teams_cal or {}).get("blocks_ev_rate", 0.45), 0.45)
            p_block_pk = _f((special_teams_cal or {}).get("blocks_pk_rate", 0.55), 0.55)
            p_block_pp_def = _f((special_teams_cal or {}).get("blocks_pp_def_rate", 0.35), 0.35)
            # Away defending blocks vs home shots
            if seg_is_home_pp:
                p_blk_away = p_block_pk
            elif seg_is_away_pp:
                p_blk_away = p_block_pp_def
            else:
                p_blk_away = p_block_ev
            # Home defending blocks vs away shots
            if seg_is_away_pp:
                p_blk_home = p_block_pk
            elif seg_is_home_pp:
                p_blk_home = p_block_pp_def
            else:
                p_blk_home = p_block_ev
            b_away = sum(1 for _ in range(sh_h) if self.rng.random() < p_blk_away)
            b_home = sum(1 for _ in range(sh_a) if self.rng.random() < p_blk_home)
            # Attribute blocks to the defending skaters actually on the ice.
            # Previously we only attributed EV blocks to the defense pair, which
            # systematically under-credited forwards and over-credited defensemen.
            def _skaters_on_ice(pids: List[int], team: TeamState) -> List[int]:
                out: List[int] = []
                for pid in (pids or []):
                    try:
                        pid_i = int(pid)
                    except Exception:
                        continue
                    ps = team.players.get(pid_i)
                    if not ps:
                        continue
                    if str(ps.position) not in ("F", "D"):
                        continue
                    out.append(pid_i)
                return out

            # b_home = home blocks vs away shots, so use home skaters currently on ice.
            # b_away = away blocks vs home shots, so use away skaters currently on ice.
            def_h = _skaters_on_ice(ice_h or [], gs.home)
            def_a = _skaters_on_ice(ice_a or [], gs.away)
            for _ in range(b_home):
                pid = _weighted_choice(def_h, gs.home, "block") if def_h else None
                events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.home.name, kind="block", player_id=pid, meta={"strength": strength_h}))
            for _ in range(b_away):
                pid = _weighted_choice(def_a, gs.away, "block") if def_a else None
                events.append(Event(t=t0 + self.rng.random() * seg_len, period=period_idx + 1, team=gs.away.name, kind="block", player_id=pid, meta={"strength": strength_a}))
            # NOTE: Do not rotate idx_l*/idx_d* here.
            # They are advanced in-branch using the appropriate rotation arrays (rot_*),
            # and rotating again here (especially with the wrong modulo) over-allocates TOI
            # to early rotation entries and creates unrealistic TOI/stat extremes.
        return home_goals, away_goals, events


class GameSimulator:
    def __init__(self, cfg: SimConfig, rates: RateModels):
        self.cfg = cfg
        self.rates = rates
        # Use local RNG instances; do not reseed global numpy.
        self.np_rng = np.random.default_rng(cfg.seed)
        self.rng = random.Random(cfg.seed)
        self.period_sim = PeriodSimulator(cfg, self.rng, self.np_rng)

    def _init_game_state(self, home_name: str, away_name: str, roster_home: List[Dict], roster_away: List[Dict]) -> GameState:
        def _norm_pos(raw: object) -> str:
            s = str(raw or "").strip().upper()
            if not s:
                return ""
            if s in {"G", "GOL", "GOALIE", "GOALTENDER"} or s.startswith("G"):
                return "G"
            # Normalize defense positions commonly seen in lineup data.
            # Examples: D, LD, RD, LHD, RHD.
            if s in {"D", "DEF", "DEFENSE", "DEFENCE", "LD", "RD", "LHD", "RHD"}:
                return "D"
            if s.startswith("D"):
                return "D"
            if s in {"F", "C", "LW", "RW", "W"}:
                return "F"
            return s

        home = TeamState(name=home_name, players={})
        away = TeamState(name=away_name, players={})
        for row in roster_home:
            pid = int(row.get("player_id"))
            toi = float(row.get("proj_toi", 0.0))
            pos = _norm_pos(row.get("position"))
            # Use provided weights if present; else derive heuristics
            sw = row.get("shot_weight")
            gw = row.get("goal_weight")
            bw = row.get("block_weight")
            if sw is None or gw is None or bw is None:
                # Baseline heuristic
                # NOTE: weights are interpreted as per-game totals and later divided by proj_toi
                # to produce per-minute propensities. Keep these defaults in realistic units
                # (shots/min, blocks/min), otherwise missing-projection players will dominate.
                sw_h = toi * (0.105 if pos == "F" else (0.060 if pos == "D" else 0.010))
                bw_h = toi * (0.085 if pos == "D" else (0.040 if pos == "F" else 0.010))
                gw_h = max(0.01, sw_h * 0.30)
                sw = float(sw if sw is not None else sw_h)
                bw = float(bw if bw is not None else bw_h)
                gw = float(gw if gw is not None else gw_h)
            p = PlayerState(player_id=pid, full_name=row.get("full_name"), position=pos, team=home_name, toi_proj=toi, shot_weight=float(sw), goal_weight=float(gw), block_weight=float(bw))
            home.players[pid] = p
        for row in roster_away:
            pid = int(row.get("player_id"))
            toi = float(row.get("proj_toi", 0.0))
            pos = _norm_pos(row.get("position"))
            sw = row.get("shot_weight")
            gw = row.get("goal_weight")
            bw = row.get("block_weight")
            if sw is None or gw is None or bw is None:
                sw_h = toi * (0.105 if pos == "F" else (0.060 if pos == "D" else 0.010))
                bw_h = toi * (0.085 if pos == "D" else (0.040 if pos == "F" else 0.010))
                gw_h = max(0.01, sw_h * 0.30)
                sw = float(sw if sw is not None else sw_h)
                bw = float(bw if bw is not None else bw_h)
                gw = float(gw if gw is not None else gw_h)
            p = PlayerState(player_id=pid, full_name=row.get("full_name"), position=pos, team=away_name, toi_proj=toi, shot_weight=float(sw), goal_weight=float(gw), block_weight=float(bw))
            away.players[pid] = p
        return GameState(home=home, away=away, period=0, clock=self.cfg.seconds_per_period)

    def _allocate_player_stats(self, team: TeamState, kind: str, count: int) -> None:
        # Allocate team events to players proportional to projected TOI (simple baseline)
        skaters = [p for p in team.players.values() if p.position in ("F", "D")]
        goalies = [p for p in team.players.values() if p.position == "G"]
        pool = skaters if kind in ("shots", "goals", "blocks") else goalies
        total = sum(max(p.toi_proj, 1e-6) for p in pool) or 1.0
        weights = [(p, max(p.toi_proj, 1e-6) / total) for p in pool]
        # Multinomial draw
        probs = [w for (_, w) in weights]
        draws = self.np_rng.multinomial(n=count, pvals=probs) if probs else np.zeros(len(weights), dtype=int)
        for (p, _), n in zip(weights, draws):
            p.stats[kind] = p.stats.get(kind, 0.0) + float(n)

    def simulate(self, home_name: str, away_name: str, roster_home: List[Dict], roster_away: List[Dict]) -> Tuple[GameState, List[Event]]:
        gs = self._init_game_state(home_name, away_name, roster_home, roster_away)
        events_all: List[Event] = []
        for pd in range(self.cfg.periods):
            hg, ag, ev = self.period_sim.simulate_period(gs, self.rates, pd)
            gs.home.score += int(hg)
            gs.away.score += int(ag)
            # Track shots separately: approximate shots = goals + misses; here use rates
            # Allocate player stats
            self._allocate_player_stats(gs.home, "shots", max(0, int(round(self.rates.home.shots_per_60 * self.cfg.seconds_per_period / 3600.0))))
            self._allocate_player_stats(gs.away, "shots", max(0, int(round(self.rates.away.shots_per_60 * self.cfg.seconds_per_period / 3600.0))))
            self._allocate_player_stats(gs.home, "goals", int(hg))
            self._allocate_player_stats(gs.away, "goals", int(ag))
            events_all.extend(ev)
        # Simple OT if tied
        if self.cfg.ot_enabled and gs.home.score == gs.away.score:
            ot_hg, ot_ag, _ = self.period_sim.simulate_period(gs, self.rates, self.cfg.periods)
            gs.home.score += int(ot_hg)
            gs.away.score += int(ot_ag)
        # Saves: opponent shots - opponent goals
        home_saves = max(0, int(round(self.rates.away.shots_per_60 * self.cfg.seconds_per_period / 3600.0 * self.cfg.periods)) - gs.away.score)
        away_saves = max(0, int(round(self.rates.home.shots_per_60 * self.cfg.seconds_per_period / 3600.0 * self.cfg.periods)) - gs.home.score)
        self._allocate_player_stats(gs.home, "saves", home_saves)
        self._allocate_player_stats(gs.away, "saves", away_saves)
        return gs, events_all

    def simulate_with_lineups(
        self,
        home_name: str,
        away_name: str,
        roster_home: List[Dict],
        roster_away: List[Dict],
        lineup_home: List[Dict],
        lineup_away: List[Dict],
        st_home: Optional[Dict[str, float]] = None,
        st_away: Optional[Dict[str, float]] = None,
        special_teams_cal: Optional[Dict[str, float]] = None,
    ) -> Tuple[GameState, List[Event]]:
        gs = self._init_game_state(home_name, away_name, roster_home, roster_away)
        events_all: List[Event] = []
        assist_model = str(getattr(self.cfg, "assist_model", "onice") or "onice").strip().lower()
        if assist_model in ("on_ice", "on-ice"):
            assist_model = "onice"
        if assist_model in ("none", "false", "0"):
            assist_model = "off"
        if assist_model not in ("onice", "legacy", "off"):
            assist_model = "onice"
        for pd in range(self.cfg.periods):
            hg, ag, ev = self.period_sim.simulate_period_with_lines(gs, self.rates, pd, lineup_home, lineup_away, st_home=st_home, st_away=st_away, special_teams_cal=special_teams_cal, period_seconds=self.cfg.seconds_per_period)
            gs.home.score += int(hg)
            gs.away.score += int(ag)
            has_assist_events = any(getattr(x, "kind", None) == "assist" for x in (ev or []))
            # Attribute player events directly from simulated events (shots/goals)
            for e in ev:
                if e.kind in ("shot", "goal", "block") and e.player_id is not None:
                    # Find player state and increment
                    team = gs.home if e.team == gs.home.name else gs.away
                    pstate = team.players.get(int(e.player_id))
                    if pstate:
                        key = "shots" if e.kind == "shot" else ("goals" if e.kind == "goal" else "blocks")
                        pstate.stats[key] = pstate.stats.get(key, 0.0) + 1.0
                        # Legacy fallback: if the period sim did not emit assist events,
                        # assign 1-2 assists to random teammates for points markets.
                        if e.kind == "goal" and (not has_assist_events) and assist_model == "legacy":
                            try:
                                # Choose assisting teammates excluding the scorer
                                skaters = [p for p in team.players.values() if p.position in ("F","D") and int(p.player_id) != int(e.player_id)]
                                if skaters:
                                    # Probability of one vs two assists
                                    n_assists = 1 + (1 if self.rng.random() < 0.35 else 0)
                                    # Weight by projected TOI
                                    weights = [max(p.toi_proj, 1e-3) for p in skaters]
                                    total_w = sum(weights) or 1.0
                                    probs = [w/total_w for w in weights]
                                    # Draw without replacement
                                    idxs = list(range(len(skaters)))
                                    chosen = []
                                    # sample n_assists unique indices
                                    if len(skaters) >= n_assists:
                                        chosen = list(self.np_rng.choice(idxs, size=n_assists, replace=False, p=np.array(probs)))
                                    else:
                                        chosen = idxs
                                    for ci in chosen:
                                        ap = skaters[int(ci)]
                                        ap.stats["assists"] = ap.stats.get("assists", 0.0) + 1.0
                                        # Emit assist event for period-level aggregation
                                        events_all.append(Event(t=e.t, period=e.period, team=team.name, kind="assist", player_id=int(ap.player_id), meta={"strength": e.meta.get("strength", "EV")}))
                            except Exception:
                                pass
            events_all.extend(ev)
        # Overtime if tied after regulation
        if self.cfg.ot_enabled and gs.home.score == gs.away.score:
            ot_idx = self.cfg.periods
            hg, ag, ev = self.period_sim.simulate_period_with_lines(gs, self.rates, ot_idx, lineup_home, lineup_away, st_home=st_home, st_away=st_away, special_teams_cal=special_teams_cal, period_seconds=self.cfg.overtime_seconds)
            gs.home.score += int(hg)
            gs.away.score += int(ag)
            events_all.extend(ev)
        # Shootout resolution if still tied; does not affect boxscore stats
        if self.cfg.shootout_enabled and gs.home.score == gs.away.score:
            # Randomly select winner; minimal bias via rates
            p_home_win = 0.5
            try:
                # small edge from goals per 60
                r_h = float(self.rates.home.goals_per_60 or 0.0)
                r_a = float(self.rates.away.goals_per_60 or 0.0)
                total = max(1e-6, r_h + r_a)
                p_home_win = max(0.25, min(0.75, r_h / total))
            except Exception:
                pass
            if self.rng.random() < p_home_win:
                gs.home.score += 1
                events_all.append(Event(t=self.cfg.periods * self.cfg.seconds_per_period + self.cfg.overtime_seconds, period=self.cfg.periods + 1, team=gs.home.name, kind="shootout"))
            else:
                gs.away.score += 1
                events_all.append(Event(t=self.cfg.periods * self.cfg.seconds_per_period + self.cfg.overtime_seconds, period=self.cfg.periods + 1, team=gs.away.name, kind="shootout"))
        # Saves from events: opponent shots minus opponent goals
        shots_home = sum(1 for e in events_all if e.kind == "shot" and e.team == gs.home.name)
        shots_away = sum(1 for e in events_all if e.kind == "shot" and e.team == gs.away.name)
        goals_home = sum(1 for e in events_all if e.kind == "goal" and e.team == gs.home.name)
        goals_away = sum(1 for e in events_all if e.kind == "goal" and e.team == gs.away.name)
        home_saves = max(0, shots_away - goals_away)
        away_saves = max(0, shots_home - goals_home)
        # Assign saves to starting goalie if present; fallback to TOI allocation
        def _assign_saves(team: TeamState, count: int) -> None:
            goalies = [p for p in team.players.values() if p.position == "G"]
            if goalies:
                starter = max(goalies, key=lambda p: (p.toi_proj or 0.0))
                starter.stats["saves"] = starter.stats.get("saves", 0.0) + float(count)
            else:
                self._allocate_player_stats(team, "saves", count)
        _assign_saves(gs.home, home_saves)
        _assign_saves(gs.away, away_saves)
        return gs, events_all
