import '../models/models.dart';

/// Local Rummikub set validation. Port of `validator.py` — do not invent extra rules.
class SetValidator {
  static const minSetSize = 3;
  static const maxGroupSize = 4;
  static const minTileValue = 1;
  static const maxTileValue = 13;

  static bool isJoker(Tile tile) => tile.color == 'JOKER' || tile.value == 0;

  /// Returns `(true, 'Board is valid')` or `(false, error)`.
  static (bool, String) validateBoard(List<TileSet> board) {
    for (var i = 0; i < board.length; i++) {
      final (ok, message) = validateSet(board[i], i);
      if (!ok) return (false, message);
    }
    return (true, 'Board is valid');
  }

  static bool isBoardValid(List<TileSet> board) => validateBoard(board).$1;

  static (bool, String) validateSet(TileSet rummikubSet, int index) {
    final label = 'Set ${index + 1} (id=${rummikubSet.id})';
    final tiles = rummikubSet.tiles;

    if (tiles.length < minSetSize) {
      return (
        false,
        '$label is invalid: a set must contain at least $minSetSize tiles '
            '(found ${tiles.length}).',
      );
    }

    if (rummikubSet.type == 'RUN') {
      return _validateRun(tiles, label);
    }
    if (rummikubSet.type == 'GROUP') {
      return _validateGroup(tiles, label);
    }

    final (runOk, runError) = _validateRun(tiles, label);
    if (runOk) return (true, '');
    final (groupOk, groupError) = _validateGroup(tiles, label);
    if (groupOk) return (true, '');
    return (
      false,
      '$label is invalid: tiles do not form a valid RUN or GROUP. '
          'RUN check: $runError GROUP check: $groupError',
    );
  }

  /// Labels a set RUN, GROUP, or INVALID from its tiles (run preferred if both).
  static TileSet classifySet(TileSet rummikubSet) {
    final tiles = rummikubSet.tiles;
    if (tiles.length < minSetSize) {
      return rummikubSet.copyWith(type: 'INVALID');
    }
    if (_validateRun(tiles, '').$1) {
      return rummikubSet.copyWith(type: 'RUN');
    }
    if (_validateGroup(tiles, '').$1) {
      return rummikubSet.copyWith(type: 'GROUP');
    }
    return rummikubSet.copyWith(type: 'INVALID');
  }

  static GameState classifyGameState(GameState state) {
    return state.copyWith(
      board: state.board.map(classifySet).toList(),
    );
  }

  static bool setTilesAreValid(List<Tile> tiles) {
    if (tiles.length < minSetSize) return false;
    return _validateRun(tiles, '').$1 || _validateGroup(tiles, '').$1;
  }

  static (bool, String) _validateRun(List<Tile> tiles, String label) {
    if (tiles.length > maxTileValue) {
      return (
        false,
        '$label is not a valid RUN: a run cannot contain more than '
            '$maxTileValue tiles (found ${tiles.length}).',
      );
    }

    final naturalIndices = <int>[];
    for (var i = 0; i < tiles.length; i++) {
      if (!isJoker(tiles[i])) naturalIndices.add(i);
    }
    if (naturalIndices.isEmpty) return (true, '');

    final colors = {for (final i in naturalIndices) tiles[i].color};
    if (colors.length > 1) {
      final colorList = (colors.toList()..sort()).join(', ');
      return (
        false,
        '$label is not a valid RUN: all non-Joker tiles must be the same color '
            '(found $colorList).',
      );
    }

    final firstIndex = naturalIndices.first;
    final startValue = tiles[firstIndex].value - firstIndex;
    final endValue = startValue + tiles.length - 1;

    if (startValue < minTileValue || endValue > maxTileValue) {
      final outOfRange =
          startValue < minTileValue ? startValue : endValue;
      return (
        false,
        '$label is not a valid RUN: Joker placement would require a tile with '
            'value $outOfRange, which is outside the valid range '
            '$minTileValue-$maxTileValue.',
      );
    }

    for (final index in naturalIndices) {
      final expected = startValue + index;
      final actual = tiles[index].value;
      if (actual != expected) {
        return (
          false,
          '$label is not a valid RUN: values must be strictly consecutive '
              'in board order (expected $expected at position ${index + 1}, '
              'found $actual).',
        );
      }
    }

    return (true, '');
  }

  static (bool, String) _validateGroup(List<Tile> tiles, String label) {
    if (tiles.length > maxGroupSize) {
      return (
        false,
        '$label is not a valid GROUP: a group cannot contain more than '
            '$maxGroupSize tiles, one per color (found ${tiles.length}).',
      );
    }

    final naturals = tiles.where((tile) => !isJoker(tile)).toList();
    if (naturals.isEmpty) return (true, '');

    final values = {for (final tile in naturals) tile.value};
    if (values.length > 1) {
      final valueList = (values.toList()..sort()).join(', ');
      return (
        false,
        '$label is not a valid GROUP: all non-Joker tiles must have the same '
            'value (found $valueList).',
      );
    }

    final seenColors = <String>{};
    for (final tile in naturals) {
      if (seenColors.contains(tile.color)) {
        return (
          false,
          '$label is not a valid GROUP: colors must be strictly unique '
              '(duplicate ${tile.color}).',
        );
      }
      seenColors.add(tile.color);
    }

    return (true, '');
  }
}
