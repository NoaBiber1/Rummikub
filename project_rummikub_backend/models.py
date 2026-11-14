from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class TileColor(str, Enum):
    RED = "RED"
    BLUE = "BLUE"
    BLACK = "BLACK"
    YELLOW = "YELLOW"
    JOKER = "JOKER"


class SetType(str, Enum):
    RUN = "RUN"
    GROUP = "GROUP"
    INVALID = "INVALID"


class Tile(BaseModel):
    id: str = Field(..., description="UUID")
    color: TileColor
    value: int = Field(..., ge=0, le=13, description="1-13, or 0 for JOKER")


class Set(BaseModel):
    id: str = Field(..., description="UUID")
    type: SetType
    tiles: list[Tile]


class GameState(BaseModel):
    board: list[Set]
    rack: list[Tile]


class TurnRequest(BaseModel):
    """A turn to record. Without ``session_id`` a new active session is opened for ``user_id``."""

    user_id: str
    session_id: Optional[str] = None
    board: list[Set]
    rack: list[Tile]
    move_applied: Optional[dict] = None


class TurnResponse(BaseModel):
    session_id: str
    turn_index: int


class EndGameRequest(BaseModel):
    session_id: str


class EndGameResponse(BaseModel):
    session_id: str
    status: str


class ChallengeSummary(BaseModel):
    id: str
    board: list[Set]
    rack: list[Tile]
    ai_score: int


class ChallengeSubmission(BaseModel):
    """A player's answer: the full board after their turn (rack tiles placed are counted from it)."""

    user_id: str
    board: list[Set]


class ChallengeResult(BaseModel):
    challenge_id: str
    human_placed: int
    ai_placed: int
    score: int


class LeaderboardEntry(BaseModel):
    rank: int
    user_id: str
    total_points: int
    challenges_won: int


class StepType(str, Enum):
    PLACE_FROM_RACK = "PLACE_FROM_RACK"
    MOVE_BETWEEN_SETS = "MOVE_BETWEEN_SETS"
    FORM_NEW_SET = "FORM_NEW_SET"
    DISBAND_SET = "DISBAND_SET"


class SolutionStep(BaseModel):
    stepIndex: int = Field(..., ge=0, description="0-based position in the sequence")
    type: StepType
    sourceSetId: Optional[str] = Field(None, description="Set the tiles come from; None for the rack or a new set")
    targetSetId: Optional[str] = Field(None, description="Set the tiles go to; None for a disband")
    tiles: list[Tile]
    description: str


# Frozen contract with the Flutter client. Do not rename these fields.
# Demo: python backend on :8000; Flutter uses 127.0.0.1 (or 10.0.2.2 on Android emulator).
# ``solution_steps`` is an additive field (defaults to empty) for the step-by-step viewer.
class OptimalState(BaseModel):
    board: list[Set]
    tiles_used_from_rack: int
    remaining_rack: list[Tile]
    solution_steps: list[SolutionStep] = Field(default_factory=list)


class SolveResponse(BaseModel):
    original_state: GameState
    optimal_state: OptimalState
    is_valid: bool
    execution_time_ms: int
