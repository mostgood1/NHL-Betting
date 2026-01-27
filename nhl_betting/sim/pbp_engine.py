from __future__ import annotations
import dataclasses
import random
from typing import List, Dict, Optional
import pandas as pd
from pathlib import Path
from nhl_betting.web.teams import get_team_assets

# Play-by-play sim with lines, shifts, and per-player aggregation

@dataclasses.dataclass
class Player:
    name: str
    position: str  # 'F', 'D', 'G'
    shot_rate: float = 1.0  # shots per 60 at 5v5 baseline
    pass_rate: float = 3.0   # passes per 60 (drives possession)
    block_rate: float = 1.0  # blocks per 60
    xg_per_shot: float = 0.08
    assist_chance: float = 0.45
    toi_target: float = 16.0  # minutes target

@dataclasses.dataclass
class PlayerStats:
    shots: float = 0.0
    goals: float = 0.0
    assists: float = 0.0
    points: float = 0.0
    blocked: float = 0.0
    saves: float = 0.0
    toi_sec: float = 0.0

@dataclasses.dataclass
class Team:
    abbr: str
    roster: List[Player]
    goalie: Optional[Player] = None
    pace_rate: float = 1.0
    penalty_rate: float = 0.15  # per team per game
    pk_save_boost: float = 0.05
    pp_units: Optional[List[List[Player]]] = None
    pk_units: Optional[List[List[Player]]] = None
    co_toi_pairs: Optional[Dict[str, Dict[str, float]]] = None

@dataclasses.dataclass
class Line:
    forwards: List[Player]
    defenders: List[Player]

@dataclasses.dataclass
class GameState:
    time_sec: int = 0
    period: int = 1
    home_score: int = 0
    away_score: int = 0
    manpower: str = "5v5"  # '5v5', 'PP', 'PK'
    possession: str = "HOME"  # 'HOME' or 'AWAY'
    home_line_idx: int = 0
    away_line_idx: int = 0
    penalty_timer_sec: int = 0
    # Track last stoppage to trigger faceoffs
    need_faceoff: bool = False

class Simulator:
    def __init__(self, home: Team, away: Team, lines_home: List[Line], lines_away: List[Line], seed: Optional[int] = None):
        self.home = home
        self.away = away
        self.lines_home = lines_home
        self.lines_away = lines_away
        if seed is not None:
            random.seed(seed)
        self.state = GameState(period=1, time_sec=0, possession=random.choice(["HOME","AWAY"]))
        self.on_ice_home = self.lines_home[0]
        self.on_ice_away = self.lines_away[0]
        self.player_stats: Dict[str, Dict[int, PlayerStats]] = {}

    def _players_on_ice(self, side: str) -> List[Player]:
        if self.state.manpower == 'PP':
            units = self.home.pp_units if side == 'HOME' else self.away.pp_units
            if units and len(units) > 0:
                # simple: alternate PP unit by minute buckets
                idx = 0 if (self.state.time_sec // 60) % 2 == 0 else 1
                idx = idx if idx < len(units) else 0
                return units[idx]
        if self.state.manpower == 'PK':
            units = self.home.pk_units if side == 'HOME' else self.away.pk_units
            if units and len(units) > 0:
                idx = 0 if (self.state.time_sec // 60) % 2 == 0 else 1
                idx = idx if idx < len(units) else 0
                return units[idx]
        if side == 'HOME':
            return self.on_ice_home.forwards + self.on_ice_home.defenders
        return self.on_ice_away.forwards + self.on_ice_away.defenders

    def _rotate_lines_if_needed(self, dt: int):
        # Rotate roughly every 45 seconds of elapsed time
        if (self.state.time_sec // 45) % 1 == 0 and dt >= 45:
            self.state.home_line_idx = (self.state.home_line_idx + 1) % len(self.lines_home)
            self.state.away_line_idx = (self.state.away_line_idx + 1) % len(self.lines_away)
            self.on_ice_home = self.lines_home[self.state.home_line_idx]
            self.on_ice_away = self.lines_away[self.state.away_line_idx]

    def _advance_time(self, mean_sec: float = 6.0) -> int:
        # Exponential time advance for next micro-event
        inc = max(1, int(random.expovariate(1.0/max(mean_sec,1e-3))))
        prev = self.state.time_sec
        self.state.time_sec += inc
        if self.state.time_sec >= 20*60:
            self.state.time_sec = 0
            self.state.period += 1
            self.state.need_faceoff = True
        # Tick down penalty and revert to 5v5
        if self.state.penalty_timer_sec > 0:
            self.state.penalty_timer_sec = max(0, self.state.penalty_timer_sec - inc)
            if self.state.penalty_timer_sec == 0:
                self.state.manpower = '5v5'
        return min(inc, 20*60 - prev)

    def _tick_toi(self, dt: int):
        for p in self._players_on_ice('HOME'):
            self._stats(p.name, self.state.period).toi_sec += dt
        for p in self._players_on_ice('AWAY'):
            self._stats(p.name, self.state.period).toi_sec += dt

    def _stats(self, player_name: str, period: int) -> PlayerStats:
        if player_name not in self.player_stats:
            self.player_stats[player_name] = {}
        if period not in self.player_stats[player_name]:
            self.player_stats[player_name][period] = PlayerStats()
        return self.player_stats[player_name][period]

    def _sample_event(self) -> str:
        # Base weights influenced by pace
        pace = 1.0
        if self.state.possession == 'HOME':
            pace = self.home.pace_rate
        else:
            pace = self.away.pace_rate
        w_pass = 3.0 * pace
        w_shot = 1.2 * pace
        w_pen = 0.02
        if self.state.manpower == 'PP':
            w_shot *= 1.5
        elif self.state.manpower == 'PK':
            w_shot *= 0.8
        total = w_pass + w_shot + w_pen
        r = random.random() * total
        if r < w_pass:
            return 'pass'
        elif r < w_pass + w_shot:
            return 'shot'
        else:
            return 'penalty'

    def _choose_shooter(self, side: str) -> Player:
        players = self._players_on_ice(side)
        weights = [max(0.01, p.shot_rate) for p in players]
        r = random.random() * sum(weights)
        s = 0.0
        for p, w in zip(players, weights):
            s += w
            if r <= s:
                return p
        return players[0]

    def _choose_blocker(self, side: str) -> Optional[Player]:
        players = self._players_on_ice(side)
        weights = [max(0.01, p.block_rate) for p in players]
        r = random.random() * sum(weights)
        s = 0.0
        for p, w in zip(players, weights):
            s += w
            if r <= s:
                return p
        return None

    def _handle_shot(self, team_key: str):
        opp_key = 'AWAY' if team_key == 'HOME' else 'HOME'
        shooter = self._choose_shooter(team_key)
        self._stats(shooter.name, self.state.period).shots += 1
        # Chained outcome probabilities
        p_block = 0.22
        p_save = 0.70
        if self.state.manpower == 'PP':
            p_block *= 0.8
            p_save *= 0.9
        elif self.state.manpower == 'PK':
            p_block *= 1.1
            p_save *= 1.05
        # Goalie calibration: boost saves based on team setting
        opp_team = self.home if opp_key == 'HOME' else self.away
        p_save *= (1.0 + max(0.0, float(opp_team.pk_save_boost or 0.0)))
        r = random.random()
        if r < p_block:
            blocker = self._choose_blocker(opp_key)
            if blocker:
                self._stats(blocker.name, self.state.period).blocked += 1
            return 'block'
        r2 = random.random()
        if r2 < p_save:
            goalie = self.home.goalie if opp_key == 'HOME' else self.away.goalie
            if goalie:
                self._stats(goalie.name, self.state.period).saves += 1
            # possession may flip
            if random.random() < 0.5:
                self.state.possession = opp_key
            return 'save'
        # Goal
        self._stats(shooter.name, self.state.period).goals += 1
        # assist assignment from teammates on ice
        teammates = [p for p in self._players_on_ice(team_key) if p.name != shooter.name]
        if teammates and random.random() < shooter.assist_chance:
            # Weight assister by co-TOI pair minutes and pass_rate
            team = self.home if team_key == 'HOME' else self.away
            weights = []
            for cand in teammates:
                base = cand.pass_rate
                co = 0.0
                if team.co_toi_pairs and shooter.name in team.co_toi_pairs:
                    co = float(team.co_toi_pairs[shooter.name].get(cand.name, 0.0))
                weights.append(max(0.01, base + 0.05 * co))
            r = random.random() * sum(weights)
            s = 0.0
            assister = teammates[0]
            for cand, w in zip(teammates, weights):
                s += w
                if r <= s:
                    assister = cand
                    break
            self._stats(assister.name, self.state.period).assists += 1
        # Update points
        self._stats(shooter.name, self.state.period).points = (
            self._stats(shooter.name, self.state.period).goals + self._stats(shooter.name, self.state.period).assists
        )
        # Score and reset possession after faceoff
        if team_key == 'HOME':
            self.state.home_score += 1
        else:
            self.state.away_score += 1
        # Trigger neutral faceoff for next possession
        self.state.need_faceoff = True
        return 'goal'

    def _faceoff_possession(self) -> str:
        # Weight possession by presence of centers on ice and slight home edge
        def side_weight(side: str) -> float:
            players = self._players_on_ice(side)
            # heuristic: centers get bonus; otherwise use pass_rate as proxy for puck control
            w = 1.0 + 0.05*(1 if side == 'HOME' else 0)
            for p in players:
                name = p.name.upper()
                # simple check: center name may include position 'C' from lineup context; fallback to pass_rate
                w += 0.02 * max(0.0, p.pass_rate)
            return w
        w_home = side_weight('HOME')
        w_away = side_weight('AWAY')
        r = random.random() * (w_home + w_away)
        return 'HOME' if r <= w_home else 'AWAY'

    def step(self):
        if self.state.period > 3:
            return False
        dt = self._advance_time()
        self._tick_toi(dt)
        self._rotate_lines_if_needed(dt)
        if self.state.need_faceoff:
            self.state.possession = self._faceoff_possession()
            self.state.need_faceoff = False
        ev = self._sample_event()
        team_key = self.state.possession
        if ev == 'pass':
            # small turnover chance switches possession
            if random.random() < 0.1:
                self.state.possession = 'AWAY' if self.state.possession == 'HOME' else 'HOME'
            return True
        elif ev == 'shot':
            self._handle_shot(team_key)
            return True
        elif ev == 'penalty':
            # toggle PP/PK for about 120 seconds
            self.state.manpower = 'PP' if team_key == 'HOME' else 'PK'
            self.state.penalty_timer_sec = 120
            return True
        return True

    def run(self):
        while self.state.period <= 3:
            self.step()
        return self.player_stats


# Utilities to build rosters and lines from artifacts

def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        return None
    return None

def _read_lineups(date: str) -> Optional[pd.DataFrame]:
    proc = Path('data/processed')
    p = proc / f'lineups_{date}.csv'
    return _safe_read_csv(p)

def _read_starting_goalies(date: str) -> Optional[pd.DataFrame]:
    proc = Path('data/processed')
    p = proc / f'starting_goalies_{date}.csv'
    return _safe_read_csv(p)

def _read_shifts(date: str) -> Optional[pd.DataFrame]:
    def _read_co_toi(date: str) -> Optional[pd.DataFrame]:
        proc = Path('data/processed')
        p = proc / f'lineups_co_toi_{date}.csv'
        return _safe_read_csv(p)

    def _read_predictions(date: str) -> Optional[pd.DataFrame]:
        proc = Path('data/processed')
        p = proc / f'predictions_{date}.csv'
        return _safe_read_csv(p)

    def abbr_to_name(abbr: str) -> Optional[str]:
        a = get_team_assets(abbr)
        nm = a.get('name') if a else None
        return str(nm) if nm else None
    proc = Path('data/processed')
    p = proc / f'shifts_{date}.csv'
    return _safe_read_csv(p)

def build_team_from_roster(date: str, team_abbr: str, projections: Optional[pd.DataFrame] = None) -> Team:
    proc = Path('data/processed')
    roster_path = proc / f'roster_snapshot_{date}.csv'
    lineup_df = _read_lineups(date)
    goalies_df = _read_starting_goalies(date)
    df = _safe_read_csv(roster_path)
    players: List[Player] = []
    goalie: Optional[Player] = None
    # Prefer lineup snapshot when available to get line slots and PP/PK units
    shifts_df = _read_shifts(date)
    toi_map: Dict[str, float] = {}
    if shifts_df is not None and not shifts_df.empty and {'team','player_id','start_s','end_s'}.issubset(shifts_df.columns):
        dftoi = shifts_df[shifts_df['team'] == team_abbr].copy()
        dftoi['dur'] = pd.to_numeric(dftoi['end_s'], errors='coerce') - pd.to_numeric(dftoi['start_s'], errors='coerce')
        dftoi = dftoi.groupby('player_id', as_index=False)['dur'].sum()
        for _, rr in dftoi.iterrows():
            try:
                toi_map[int(rr['player_id'])] = float(rr['dur'])/60.0
            except Exception:
                pass
    if lineup_df is not None and not lineup_df.empty and {'team','full_name','position','line_slot','pp_unit','pk_unit','proj_toi','player_id'}.issubset(lineup_df.columns):
        dft = lineup_df[lineup_df['team'] == team_abbr]
        for _, r in dft.iterrows():
            name = str(r.get('full_name') or '')
            pos = str(r.get('position') or '').upper()[:1]
            shot = 1.0
            block = 1.0
            assist = 0.4
            toi = float(r.get('proj_toi') or 16.0)
            # Override TOI from shiftcharts if available
            try:
                pid = int(r.get('player_id'))
                if pid in toi_map:
                    toi = float(toi_map[pid])
            except Exception:
                pass
            if projections is not None and {'player','team_abbr','market','value'}.issubset(projections.columns):
                pp = projections[(projections['player'] == name) & (projections['team_abbr'] == team_abbr)]
                sog = pp[pp['market']=='SOG']['value'].mean() if not pp.empty else None
                blk = pp[pp['market']=='BLOCKS']['value'].mean() if not pp.empty else None
                pts = pp[pp['market']=='POINTS']['value'].mean() if not pp.empty else None
                svs = pp[pp['market']=='SAVES']['value'].mean() if not pp.empty else None
                if pd.notnull(sog):
                    shot = float(sog)
                if pd.notnull(blk):
                    block = float(blk)
                if pd.notnull(pts):
                    assist = 0.35 + min(0.5, float(pts) * 0.05)
                # Goalies: use SAVES projection to calibrate save boost
                if pos == 'G' and pd.notnull(svs):
                    # map higher saves to higher save baseline via pk_save_boost
                    team.pk_save_boost = 0.03 + min(0.12, float(svs) * 0.002)
            p = Player(name=name, position=pos, shot_rate=shot, block_rate=block, assist_chance=assist, toi_target=toi)
            if pos == 'G':
                goalie = p
            else:
                players.append(p)
        # Assign starting goalie from snapshot if available
        if goalie is None and goalies_df is not None and not goalies_df.empty and 'team' in goalies_df.columns:
            gg = goalies_df[goalies_df['team'] == team_abbr]
            if not gg.empty:
                gname = str(gg.iloc[0].get('goalie') or '')
                goalie = Player(name=gname, position='G')
    elif df is not None and {'team_abbr','player','position'}.issubset(df.columns):
        dft = df[df['team_abbr'] == team_abbr]
        for _, r in dft.iterrows():
            name = str(r['player'])
            pos = str(r['position']).upper()[:1]
            shot = 1.0
            block = 1.0
            assist = 0.4
            toi = 16.0
            if projections is not None and {'player','team_abbr','market','value'}.issubset(projections.columns):
                pp = projections[(projections['player'] == name) & (projections['team_abbr'] == team_abbr)]
                sog = pp[pp['market']=='SOG']['value'].mean() if not pp.empty else None
                blk = pp[pp['market']=='BLOCKS']['value'].mean() if not pp.empty else None
                pts = pp[pp['market']=='POINTS']['value'].mean() if not pp.empty else None
                if pd.notnull(sog):
                    shot = float(sog)
                if pd.notnull(blk):
                    block = float(blk)
                if pd.notnull(pts):
                    assist = 0.35 + min(0.5, float(pts) * 0.05)
                # crude TOI scaling: more shots -> more ice time
                toi = 14.0 + min(8.0, shot * 0.8)
            p = Player(name=name, position=pos, shot_rate=shot, block_rate=block, assist_chance=assist, toi_target=toi)
            if pos == 'G':
                goalie = p
            else:
                players.append(p)
    else:
        # Fallback generic roster: 12F, 6D, 1G
        for i in range(12):
            players.append(Player(name=f"{team_abbr}_F{i+1}", position='F', shot_rate=1.1 + 0.1*i))
        for i in range(6):
            players.append(Player(name=f"{team_abbr}_D{i+1}", position='D', block_rate=1.2 + 0.1*i, shot_rate=0.6))
        goalie = Player(name=f"{team_abbr}_G1", position='G')
    return Team(abbr=team_abbr, roster=players, goalie=goalie)

def build_lines(team: Team) -> List[Line]:
    fwds = [p for p in team.roster if p.position == 'F']
    defs = [p for p in team.roster if p.position == 'D']
    # Sort forwards by shot_rate desc; defenders by block_rate desc
    fwds.sort(key=lambda x: x.shot_rate, reverse=True)
    defs.sort(key=lambda x: x.block_rate, reverse=True)
    lines: List[Line] = []
    for i in range(0, min(len(fwds), 12), 3):
        fgrp = fwds[i:i+3]
        didx = (i // 3) * 2
        dgrp = defs[didx:didx+2] if didx+1 < len(defs) else defs[-2:]
        if len(fgrp) == 3 and len(dgrp) == 2:
            lines.append(Line(forwards=fgrp, defenders=dgrp))
    # Ensure at least 3 lines
    while len(lines) < 3 and len(fwds) >= 3 and len(defs) >= 2:
        lines.append(Line(forwards=fwds[:3], defenders=defs[:2]))
    # Robust fallback: if still empty, build a generic line from top skaters
    if not lines:
        skaters = [p for p in team.roster if p.position in ('F','D')]
        skaters.sort(key=lambda x: (x.toi_target, x.shot_rate), reverse=True)
        if len(skaters) >= 5:
            lines.append(Line(forwards=skaters[:3], defenders=skaters[3:5]))
        elif len(skaters) >= 3:
            lines.append(Line(forwards=skaters[:3], defenders=[]))
    return lines
    return lines

def build_special_units(date: str, team: Team) -> None:
    df = _read_lineups(date)
    co_df = _read_co_toi(date)
    units_pp: List[List[Player]] = []
    units_pk: List[List[Player]] = []
    co_pairs: Dict[str, Dict[str, float]] = {}
    if df is not None and not df.empty and {'team','full_name','position','pp_unit','pk_unit'}.issubset(df.columns):
        dft = df[df['team'] == team.abbr]
        # Build PP units
        for unit_no in [1,2]:
            grp = dft[dft['pp_unit'] == unit_no]
            names = [str(x) for x in grp['full_name'].tolist()]
            players = []
            for nm in names:
                p = next((q for q in team.roster if q.name == nm), None)
                if p:
                    players.append(p)
            if len(players) >= 4:
                units_pp.append(players[:5])
        # Build PK units
        for unit_no in [1,2]:
            grp = dft[dft['pk_unit'] == unit_no]
            names = [str(x) for x in grp['full_name'].tolist()]
            players = []
            for nm in names:
                p = next((q for q in team.roster if q.name == nm), None)
                if p:
                    players.append(p)
            if len(players) >= 3:
                units_pk.append(players[:4])
    # Fallback generic units if missing
    if not units_pp:
        top_fwds = sorted([p for p in team.roster if p.position == 'F'], key=lambda x: x.toi_target, reverse=True)[:4]
        top_defs = sorted([p for p in team.roster if p.position == 'D'], key=lambda x: x.toi_target, reverse=True)[:1]
        units_pp = [top_fwds + top_defs]
    if not units_pk:
        top_fwds = sorted([p for p in team.roster if p.position == 'F'], key=lambda x: x.block_rate, reverse=True)[:3]
        top_defs = sorted([p for p in team.roster if p.position == 'D'], key=lambda x: x.block_rate, reverse=True)[:1]
        units_pk = [top_fwds + top_defs]
    team.pp_units = units_pp
    team.pk_units = units_pk
    # Build co-TOI pairs map (name-based) for assist weighting
    if co_df is not None and not co_df.empty and {'team','player_id_a','player_id_b','co_toi_ev'}.issubset(co_df.columns):
        # Map player_id to full_name using lineup df
        id_to_name: Dict[int, str] = {}
        if df is not None and not df.empty and {'team','player_id','full_name'}.issubset(df.columns):
            dfn = df[df['team'] == team.abbr]
            for _, r in dfn.iterrows():
                try:
                    id_to_name[int(r.get('player_id'))] = str(r.get('full_name'))
                except Exception:
                    pass
        dct: Dict[str, Dict[str, float]] = {}
        co_team = co_df[co_df['team'] == team.abbr]
        for _, r in co_team.iterrows():
            try:
                a = id_to_name.get(int(r.get('player_id_a')))
                b = id_to_name.get(int(r.get('player_id_b')))
                if a and b:
                    dct.setdefault(a, {})[b] = float(r.get('co_toi_ev') or 0.0)
                    dct.setdefault(b, {})[a] = float(r.get('co_toi_ev') or 0.0)
            except Exception:
                pass
        team.co_toi_pairs = dct

def simulate_game(home_abbr: str, away_abbr: str, date: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Dict[int, PlayerStats]]:
    projections = None
    if date:
        p = Path('data/processed') / f'props_projections_all_{date}.csv'
        projections = _safe_read_csv(p)
    home = build_team_from_roster(date or '', home_abbr, projections=projections)
    away = build_team_from_roster(date or '', away_abbr, projections=projections)
    # Pace calibration from predictions totals
    pred_df = _read_predictions(date or '')
    if pred_df is not None and not pred_df.empty and {'home','away'}.issubset(pred_df.columns):
        home_name = abbr_to_name(home_abbr) or home_abbr
        away_name = abbr_to_name(away_abbr) or away_abbr
        row = pred_df[(pred_df['home'] == home_name) & (pred_df['away'] == away_name)]
        if not row.empty:
            r0 = row.iloc[0]
            try:
                ph = float(r0.get('proj_home_goals') or r0.get('period1_home_proj') or 3.0)
                pa = float(r0.get('proj_away_goals') or r0.get('period1_away_proj') or 3.0)
                tot = ph + pa
                base_tot = 6.2
                home.pace_rate = max(0.6, min(1.6, 0.9 + 0.2 * (ph / 3.1)))
                away.pace_rate = max(0.6, min(1.6, 0.9 + 0.2 * (pa / 3.1)))
            except Exception:
                pass
    lines_home = build_lines(home)
    lines_away = build_lines(away)
    if date:
        build_special_units(date, home)
        build_special_units(date, away)
    sim = Simulator(home, away, lines_home, lines_away, seed=seed)
    stats = sim.run()
    return stats

def to_dataframe(stats: Dict[str, Dict[int, PlayerStats]], team_map: Dict[str, str]) -> pd.DataFrame:
    rows = []
    for player, per in stats.items():
        tm = team_map.get(player, None)
        for period, s in per.items():
            rows.append({
                'team_abbr': tm,
                'player': player,
                'period': period,
                'shots': round(s.shots, 3),
                'goals': round(s.goals, 3),
                'assists': round(s.assists, 3),
                'points': round(s.points, 3),
                'blocked': round(s.blocked, 3),
                'saves': round(s.saves, 3),
                'toi_min': round(s.toi_sec/60.0, 3),
            })
    return pd.DataFrame(rows)

