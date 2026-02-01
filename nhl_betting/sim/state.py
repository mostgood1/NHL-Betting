from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class PlayerState:
    player_id: int
    full_name: str
    position: str  # F/D/G
    team: str
    toi_proj: float = 0.0
    # Weights to bias event attribution while on ice
    shot_weight: float = 0.0
    goal_weight: float = 0.0
    block_weight: float = 0.0
    stats: Dict[str, float] = field(default_factory=dict)  # shots, goals, assists, blocks, saves


@dataclass
class TeamState:
    name: str
    abbrev: Optional[str] = None
    score: int = 0
    shots: int = 0
    penalties: int = 0
    players: Dict[int, PlayerState] = field(default_factory=dict)


@dataclass
class GameState:
    home: TeamState
    away: TeamState
    period: int = 0
    clock: int = 20 * 60  # seconds remaining in current period
    events: List["Event"] = field(default_factory=list)


@dataclass
class Event:
    t: float  # absolute seconds since start
    period: int
    team: str  # team name
    kind: str  # faceoff|shot|goal|block|penalty|save|turnover|shift
    player_id: Optional[int] = None
    meta: Dict[str, float] = field(default_factory=dict)
