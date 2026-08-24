import 'game_state.dart';
import 'json_map.dart';

class ArenaChallenge {
  const ArenaChallenge({
    required this.id,
    required this.title,
    required this.initialState,
    required this.algorithmScore,
    this.algorithmAfterState,
    this.source = 'solver',
  });

  final String id;
  final String title;
  final GameState initialState;
  final int algorithmScore;

  /// Documented solver output for this puzzle, when published from Solve.
  final GameState? algorithmAfterState;

  /// `seed` = bundled puzzle, `solver` = published from a system solution.
  final String source;

  factory ArenaChallenge.fromFirestore(
    String id,
    Map<String, dynamic> data,
  ) {
    final initial = asStringKeyedMap(data['initial_state']);
    final afterRaw = data['algorithm_after_state'];
    return ArenaChallenge(
      id: id,
      title: (data['title'] as String?) ?? 'Challenge',
      initialState: GameState.fromJson(initial),
      algorithmScore: (data['algorithm_score'] as num?)?.toInt() ?? 0,
      algorithmAfterState: afterRaw is Map
          ? GameState.fromJson(asStringKeyedMap(afterRaw))
          : null,
      source: (data['source'] as String?) ?? 'solver',
    );
  }

  Map<String, dynamic> toFirestore() {
    return {
      'title': title,
      'initial_state': initialState.toJson(),
      'algorithm_score': algorithmScore,
      'source': source,
      if (algorithmAfterState != null)
        'algorithm_after_state': algorithmAfterState!.toJson(),
    };
  }
}
