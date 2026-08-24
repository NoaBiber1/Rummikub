import 'game_state.dart';

/// Solver API contract. Frozen with the Python backend:
/// `optimal_state.board`, `tiles_used_from_rack`, `remaining_rack`.
class SolveResult {
  const SolveResult({
    required this.originalState,
    required this.optimalState,
    required this.tilesUsedFromRack,
    required this.isValid,
    required this.executionTimeMs,
  });

  final GameState originalState;
  final GameState optimalState;
  final int tilesUsedFromRack;
  final bool isValid;
  final int executionTimeMs;
}
