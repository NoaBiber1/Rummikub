from enum import Enum

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


# Frozen contract with the Flutter client. Do not rename these fields.
# Demo: python backend on :8000; Flutter uses 127.0.0.1 (or 10.0.2.2 on Android emulator).
class OptimalState(BaseModel):
    board: list[Set]
    tiles_used_from_rack: int
    remaining_rack: list[Tile]


class SolveResponse(BaseModel):
    original_state: GameState
    optimal_state: OptimalState
    is_valid: bool
    execution_time_ms: int
