import 'json_map.dart';
import 'set.dart';
import 'tile.dart';

/// Represents the complete state of a Rummikub game.
/// 
/// Contains:
/// - board: Array of TileSet objects representing sets on the board
/// - rack: Array of Tile objects representing tiles in the player's rack
class GameState {
  final List<TileSet> board;
  final List<Tile> rack;

  GameState({
    required this.board,
    required this.rack,
  });

  /// Creates a GameState from JSON map
  factory GameState.fromJson(Map<String, dynamic> json) {
    if (json['board'] == null || json['rack'] == null) {
      throw ArgumentError('GameState JSON must contain board and rack fields');
    }

    final board = asMapList(json['board']).map(TileSet.fromJson).toList();
    final rack = asMapList(json['rack']).map(Tile.fromJson).toList();

    return GameState(
      board: board,
      rack: rack,
    );
  }

  /// Converts GameState to JSON map
  Map<String, dynamic> toJson() {
    return {
      'board': board.map((set) => set.toJson()).toList(),
      'rack': rack.map((tile) => tile.toJson()).toList(),
    };
  }

  /// Creates an empty GameState
  factory GameState.empty() {
    return GameState(
      board: [],
      rack: [],
    );
  }

  /// Creates a copy of this GameState with optional field updates
  GameState copyWith({
    List<TileSet>? board,
    List<Tile>? rack,
  }) {
    return GameState(
      board: board ?? List.from(this.board),
      rack: rack ?? List.from(this.rack),
    );
  }

  /// Returns the total number of tiles on the board
  int get totalTilesOnBoard {
    return board.fold(0, (sum, set) => sum + set.tiles.length);
  }

  /// Returns the number of tiles in the rack
  int get rackSize => rack.length;

  /// Returns the number of sets on the board
  int get numberOfSets => board.length;

  /// Checks if the board is empty
  bool get isBoardEmpty => board.isEmpty;

  /// Checks if the rack is empty
  bool get isRackEmpty => rack.isEmpty;

  /// Returns all invalid sets on the board
  List<TileSet> get invalidSets {
    return board.where((set) => set.isInvalid).toList();
  }

  /// Returns all valid sets on the board
  List<TileSet> get validSets {
    return board.where((set) => set.isValid).toList();
  }

  /// Checks if all sets on the board are valid
  bool get isAllSetsValid {
    return board.every((set) => set.isValid);
  }

  /// Adds a tile to the rack
  GameState addTileToRack(Tile tile) {
    return copyWith(rack: [...rack, tile]);
  }

  /// Removes a tile from the rack by ID
  GameState removeTileFromRack(String tileId) {
    return copyWith(
      rack: rack.where((tile) => tile.id != tileId).toList(),
    );
  }

  /// Adds a set to the board
  GameState addSetToBoard(TileSet set) {
    return copyWith(board: [...board, set]);
  }

  /// Removes a set from the board by ID
  GameState removeSetFromBoard(String setId) {
    return copyWith(
      board: board.where((set) => set.id != setId).toList(),
    );
  }

  /// Updates a set on the board
  GameState updateSetOnBoard(String setId, TileSet updatedSet) {
    return copyWith(
      board: board.map((set) => set.id == setId ? updatedSet : set).toList(),
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is GameState &&
        _listEquals(other.board, board) &&
        _listEquals(other.rack, rack);
  }

  @override
  int get hashCode => Object.hash(Object.hashAll(board), Object.hashAll(rack));

  @override
  String toString() {
    return 'GameState(board: ${board.length} sets, rack: ${rack.length} tiles)';
  }

  /// Helper method to compare two lists
  static bool _listEquals(List a, List b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
