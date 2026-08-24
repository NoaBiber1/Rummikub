from ortools.sat.python import cp_model

from models import GameState, OptimalState


def solve_rummikub(game_state: GameState) -> OptimalState:
    """Maximize tiles dropped from the rack onto a valid Rummikub board.

    Hard timeout: 5 seconds (project constraint).
    """
    model = cp_model.CpModel()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0

    # TODO: Decision variables
    # - Assign each tile (existing board tiles + rack tiles) to a set, or leave unused
    # - Binary indicator: whether each rack tile is placed onto the board
    # - Per-set type: RUN vs GROUP

    # TODO: Validity constraints (standard Rummikub rules)
    # - Every set on the board must contain at least 3 tiles
    # - GROUP: same value, all different colors, 3 or 4 tiles
    # - RUN: same color, consecutive values, at least 3 tiles
    # - Jokers may substitute any missing tile in a run or group
    # - Tiles already on the board MUST remain on the board (cannot return to the rack)

    # TODO: Objective
    # - Maximize tiles_used_from_rack

    # status = solver.Solve(model)

    # Mock until the CP-SAT constraints are implemented: echo the input state.
    return OptimalState(
        board=game_state.board,
        tiles_used_from_rack=0,
        remaining_rack=game_state.rack,
    )
