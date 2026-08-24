import 'package:flutter/foundation.dart';
import '../logic/set_validator.dart';
import '../models/models.dart';

/// Provider for managing the game state throughout the app.
class GameProvider extends ChangeNotifier {
  GameState _gameState = GameState.empty();
  GameState? _visionOutput;

  GameState get gameState => _gameState;

  /// Snapshot from camera/gallery recognition, used for `vision_corrections`.
  GameState? get visionOutput => _visionOutput;

  void _commit(GameState next) {
    _gameState = SetValidator.classifyGameState(next);
    notifyListeners();
  }

  void updateBoard(List<TileSet> newBoard) {
    _commit(_gameState.copyWith(board: newBoard));
  }

  void updateRack(List<Tile> newRack) {
    _commit(_gameState.copyWith(rack: newRack));
  }

  void clearState() {
    _visionOutput = null;
    _gameState = GameState.empty();
    notifyListeners();
  }

  void updateGameState(GameState newState, {bool fromVision = false}) {
    if (fromVision) {
      _visionOutput = SetValidator.classifyGameState(newState);
    }
    _commit(newState);
  }

  void beginManualEntry() {
    _visionOutput = null;
    _commit(GameState.empty());
  }

  void beginArena(GameState initialState) {
    _visionOutput = null;
    _commit(initialState);
  }

  void clearVisionOutput() {
    _visionOutput = null;
  }

  void addTileToRack(Tile tile) {
    _commit(_gameState.addTileToRack(tile));
  }

  void removeTileFromRack(String tileId) {
    _commit(_gameState.removeTileFromRack(tileId));
  }

  void addSetToBoard(TileSet set) {
    _commit(_gameState.addSetToBoard(set));
  }

  void removeSetFromBoard(String setId) {
    _commit(_gameState.removeSetFromBoard(setId));
  }

  void updateSetOnBoard(String setId, TileSet updatedSet) {
    _commit(_gameState.updateSetOnBoard(setId, updatedSet));
  }

  /// Replaces [original] with [updated] on the rack or in [setId].
  void replaceTile(Tile original, Tile updated, {String? setId}) {
    if (setId == null) {
      final newRack = _gameState.rack
          .map((tile) => tile.id == original.id ? updated : tile)
          .toList();
      _commit(_gameState.copyWith(rack: newRack));
      return;
    }
    final set = _gameState.board.firstWhere((s) => s.id == setId);
    final newTiles = set.tiles
        .map((tile) => tile.id == original.id ? updated : tile)
        .toList();
    _commit(_gameState.updateSetOnBoard(setId, set.copyWith(tiles: newTiles)));
  }

  void deleteTile(Tile tile, {String? setId}) {
    if (setId == null) {
      _commit(_gameState.removeTileFromRack(tile.id));
      return;
    }
    final set = _gameState.board.firstWhere((s) => s.id == setId);
    final newTiles = set.tiles.where((t) => t.id != tile.id).toList();
    if (newTiles.isEmpty) {
      _commit(_gameState.removeSetFromBoard(setId));
    } else {
      _commit(
        _gameState.updateSetOnBoard(setId, set.copyWith(tiles: newTiles)),
      );
    }
  }

  /// Moves [tileId] from rack (`fromSetId == null`) or a set onto [destination].
  void moveTile({
    required String tileId,
    String? fromSetId,
    required TileDestination destination,
  }) {
    final located = _takeTile(tileId, fromSetId);
    if (located == null) return;
    var board = located.board;
    var rack = located.rack;
    final tile = located.tile;

    switch (destination.kind) {
      case TileDestKind.rack:
        rack = [...rack, tile];
      case TileDestKind.newSet:
        board = [...board, TileSet(type: 'INVALID', tiles: [tile])];
      case TileDestKind.insertIntoSet:
        board = _insertIntoSet(
          board: board,
          setId: destination.setId!,
          tile: tile,
          index: _adjustedInsertIndex(
            fromSetId: fromSetId,
            setId: destination.setId!,
            requestedIndex: destination.index ?? 0,
            originalIndex: located.originalIndex,
          ),
        );
    }

    _commit(GameState(board: board, rack: rack));
  }

  _TakenTile? _takeTile(String tileId, String? fromSetId) {
    if (fromSetId == null) {
      Tile? found;
      for (final tile in _gameState.rack) {
        if (tile.id == tileId) found = tile;
      }
      if (found == null) return null;
      return _TakenTile(
        tile: found,
        board: _gameState.board,
        rack: _gameState.rack.where((t) => t.id != tileId).toList(),
      );
    }

    Tile? found;
    int? originalIndex;
    final board = <TileSet>[];
    for (final set in _gameState.board) {
      if (set.id != fromSetId) {
        board.add(set);
        continue;
      }
      for (var i = 0; i < set.tiles.length; i++) {
        if (set.tiles[i].id == tileId) {
          found = set.tiles[i];
          originalIndex = i;
        }
      }
      final remaining = set.tiles.where((t) => t.id != tileId).toList();
      if (remaining.isNotEmpty) {
        board.add(set.copyWith(tiles: remaining));
      }
    }
    if (found == null) return null;
    return _TakenTile(
      tile: found,
      board: board,
      rack: _gameState.rack,
      originalIndex: originalIndex,
    );
  }

  static int _adjustedInsertIndex({
    required String? fromSetId,
    required String setId,
    required int requestedIndex,
    required int? originalIndex,
  }) {
    var index = requestedIndex;
    if (fromSetId == setId && originalIndex != null && index > originalIndex) {
      index -= 1;
    }
    return index < 0 ? 0 : index;
  }

  static List<TileSet> _insertIntoSet({
    required List<TileSet> board,
    required String setId,
    required Tile tile,
    required int index,
  }) {
    var found = false;
    final next = board.map((set) {
      if (set.id != setId) return set;
      found = true;
      final tiles = [...set.tiles];
      final clamped = index.clamp(0, tiles.length);
      tiles.insert(clamped, tile);
      return set.copyWith(tiles: tiles);
    }).toList();
    if (found) return next;
    return [
      ...board,
      TileSet(id: setId, type: 'INVALID', tiles: [tile]),
    ];
  }
}

class TileDestination {
  const TileDestination._(this.kind, this.setId, [this.index]);

  final TileDestKind kind;
  final String? setId;
  final int? index;

  static const rack = TileDestination._(TileDestKind.rack, null);
  static const newSet = TileDestination._(TileDestKind.newSet, null);
  static TileDestination insertIntoSet(String setId, int index) =>
      TileDestination._(TileDestKind.insertIntoSet, setId, index);
}

enum TileDestKind { rack, newSet, insertIntoSet }

class _TakenTile {
  _TakenTile({
    required this.tile,
    required this.board,
    required this.rack,
    this.originalIndex,
  });

  final Tile tile;
  final List<TileSet> board;
  final List<Tile> rack;
  final int? originalIndex;
}
