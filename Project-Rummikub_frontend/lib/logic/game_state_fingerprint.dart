import '../models/models.dart';

/// Stable identity for a board+rack position, ignoring tile/set UUIDs.
///
/// Used so the same puzzle is one `global_arena` document instead of a new
/// challenge on every Solve.
class GameStateFingerprint {
  static String canonical(GameState state) {
    final sets = state.board.map(_setKey).toList()..sort();
    final rack = state.rack.map(_tileKey).toList()..sort();
    return '${sets.join('|')}||${rack.join(',')}';
  }

  /// Firestore document id for a solver-published challenge.
  static String challengeId(GameState state) {
    final raw = canonical(state);
    final safe = raw.replaceAll('/', '_');
    if (safe.length <= 700) return 'sol_$safe';
    return 'sol_${_fnv1aHex(safe)}';
  }

  static String _tileKey(Tile tile) => '${tile.color}:${tile.value}';

  static String _setKey(TileSet set) => set.tiles.map(_tileKey).join(',');

  static String _fnv1aHex(String input) {
    var hash = 0x811c9dc5;
    for (final unit in input.codeUnits) {
      hash ^= unit;
      hash = (hash * 0x01000193) & 0x7fffffff;
    }
    return hash.toRadixString(16).padLeft(8, '0');
  }
}
