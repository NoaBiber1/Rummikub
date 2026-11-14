import 'game_state.dart';
import 'solution_step.dart';

/// Solver API contract. Frozen with the Python backend:
/// `optimal_state.board`, `tiles_used_from_rack`, `remaining_rack`.
/// `optimal_state.solution_steps` is additive and may be missing.
class SolveResult {
  const SolveResult({
    required this.originalState,
    required this.optimalState,
    required this.tilesUsedFromRack,
    required this.isValid,
    required this.executionTimeMs,
    this.solutionSteps = const [],
  });

  final GameState originalState;
  final GameState optimalState;
  final int tilesUsedFromRack;
  final bool isValid;
  final int executionTimeMs;
  final List<SolutionStep> solutionSteps;
}
