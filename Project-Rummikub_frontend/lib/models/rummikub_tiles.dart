import 'tile.dart';

const _colorOrder = ['BLACK', 'BLUE', 'RED', 'YELLOW'];

int _colorRank(Tile tile) {
  final rank = _colorOrder.indexOf(tile.color);
  return rank < 0 ? _colorOrder.length : rank;
}

/// Orders tiles by color (Black, Blue, Red, Yellow), then by value (1-13).
/// Jokers go last. Returns a new list.
List<Tile> sortTiles(Iterable<Tile> tiles) {
  final sorted = tiles.toList();
  sorted.sort((a, b) {
    final aJoker = a.color == 'JOKER';
    final bJoker = b.color == 'JOKER';
    if (aJoker != bJoker) return aJoker ? 1 : -1;
    final byColor = _colorRank(a).compareTo(_colorRank(b));
    return byColor != 0 ? byColor : a.value.compareTo(b.value);
  });
  return sorted;
}

/// Tiles a rack shelf takes before new tiles start the next one.
const kRackRowCapacity = 14;

/// Gives every rack tile a shelf. A tile keeps its own [Tile.rackRow], else
/// the one it had in [previous] (tile id -> row, so a solved rack keeps the
/// photo's layout); the rest fill the top shelf up to [kRackRowCapacity], then
/// the bottom one. Returns a new list in the same order.
List<Tile> withRackRows(
  List<Tile> rack, {
  Map<String, int> previous = const {},
}) {
  int? known(Tile t) => t.rackRow ?? previous[t.id];
  var top = rack.where((t) => known(t) == 0).length;
  final placed = <Tile>[];
  for (final tile in rack) {
    var row = known(tile);
    if (row == null) {
      row = top < kRackRowCapacity ? 0 : 1;
      if (row == 0) top++;
    }
    placed.add(tile.rackRow == row ? tile : tile.copyWith(rackRow: row));
  }
  return placed;
}

/// The rack's two shelves, top first, each in rack order. Tiles without a
/// shelf are placed as [withRackRows] would.
List<List<Tile>> splitRackRows(List<Tile> rack) {
  final placed = withRackRows(rack);
  return [
    for (final row in [0, 1]) placed.where((t) => t.rackRow == row).toList(),
  ];
}

/// Sorts a set's tiles, unless it holds a joker: a joker's position in a run
/// is meaningful, so those sets keep the order the user gave them.
List<Tile> sortSetTiles(List<Tile> tiles) =>
    tiles.any((t) => t.color == 'JOKER') ? tiles : sortTiles(tiles);
