import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import '../logic/set_validator.dart';
import '../models/models.dart';
import '../models/rummikub_tiles.dart';
import '../services/api_service.dart';
import '../services/roboflow_debug_dump.dart';

/// Provider for managing the game state throughout the app.
class GameProvider extends ChangeNotifier {
  GameProvider({ApiService? api}) : _api = api ?? ApiService();

  final ApiService _api;

  GameState _gameState = GameState.empty();
  GameState? _visionOutput;
  String? _sessionId;
  int _endedGames = 0;

  String? lastVisionDebug;
  RoboflowDebugDump? lastRoboflowDump;

  GameState get gameState => _gameState;

  ApiService get api => _api;

  /// Counts finished games, so a screen showing a game's leftovers (photos)
  /// can tell it has to start fresh.
  int get endedGames => _endedGames;

  /// The backend game this play-through belongs to; null until its first turn.
  String? get sessionId => _sessionId;

  /// Snapshot from camera/gallery recognition, used for `vision_corrections`.
  GameState? get visionOutput => _visionOutput;

  void _commit(GameState next) {
    final boardCleared = _gameState.board.isNotEmpty && next.board.isEmpty;
    // The rack keeps the order the tiles were detected, added or moved in,
    // and each tile its shelf: from the photo, else where it was before.
    final rows = {
      for (final tile in _gameState.rack)
        if (tile.rackRow != null) tile.id: tile.rackRow!,
    };
    _gameState = SetValidator.classifyGameState(
      next.copyWith(rack: withRackRows(next.rack, previous: rows)),
    );
    if (boardCleared) _closeSession();
    notifyListeners();
  }

  /// Records the current turn (`POST /game/turn`), opening the backend game on
  /// the first one. Returns the turn, or null when the backend is unreachable:
  /// the game keeps working without its history.
  Future<GameTurn?> recordTurn(String userId, {GameState? state}) async {
    try {
      final turn = await _api.recordTurn(
        userId: userId,
        state: state ?? _gameState,
        sessionId: _sessionId,
      );
      _sessionId = turn.sessionId;
      return turn;
    } on SolverException {
      return null;
    }
  }

  /// Tells the backend the game is over. Never throws.
  void _closeSession() {
    final id = _sessionId;
    _sessionId = null;
    if (id == null) return;
    _api.endGame(id).catchError((Object _) {});
  }

  /// Ends the game: the backend marks it packed and the app goes back to a
  /// blank state, ready for a fresh recognition.
  void endGame() {
    _endedGames++;
    clearState();
  }

  void updateBoard(List<TileSet> newBoard) {
    _commit(_gameState.copyWith(board: newBoard));
  }

  void updateRack(List<Tile> newRack) {
    _commit(_gameState.copyWith(rack: newRack));
  }

  void clearState() {
    _closeSession();
    _visionOutput = null;
    lastVisionDebug = null;
    lastRoboflowDump = null;
    _gameState = GameState.empty();
    notifyListeners();
  }

  void updateGameState(
    GameState newState, {
    bool fromVision = false,
    String? visionDebug,
    RoboflowDebugDump? roboflowDump,
  }) {
    if (fromVision) {
      _visionOutput = SetValidator.classifyGameState(newState);
      lastVisionDebug = visionDebug;
      lastRoboflowDump = roboflowDump;
    }
    _commit(newState);
  }

  void beginManualEntry() {
    _visionOutput = null;
    lastVisionDebug = null;
    lastRoboflowDump = null;
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
    _commit(_gameState.updateSetOnBoard(
      setId,
      updatedSet.copyWith(tiles: sortSetTiles(updatedSet.tiles)),
    ));
  }

  /// Replaces [replaces] consecutive sets starting at [setId] with
  /// [replacement], in place on the board.
  void _spliceSet(
    String setId,
    List<TileSet> replacement, {
    int replaces = 1,
  }) {
    final board = _gameState.board;
    final at = board.indexWhere((s) => s.id == setId);
    if (at < 0) return;
    _commit(_gameState.copyWith(board: [
      ...board.sublist(0, at),
      ...replacement,
      ...board.sublist(math.min(at + replaces, board.length)),
    ]));
  }

  /// Joins set [setId] with the set below it. The result keeps [setId]'s slot
  /// and id; its tiles are sorted unless one side is empty, which has nothing
  /// to merge and so keeps the other side's order. Does nothing for the last set.
  void mergeWithNextSet(String setId) {
    final board = _gameState.board;
    final at = board.indexWhere((s) => s.id == setId);
    if (at < 0 || at == board.length - 1) return;
    final first = board[at];
    final tiles = [...first.tiles, ...board[at + 1].tiles];
    final oneSideEmpty = first.isEmpty || board[at + 1].isEmpty;
    _spliceSet(
      setId,
      [first.copyWith(tiles: oneSideEmpty ? tiles : sortSetTiles(tiles))],
      replaces: 2,
    );
  }

  /// Cuts set [setId] before tile [index]; the tail becomes the next set.
  void splitSet(String setId, int index) {
    final set = _gameState.board.firstWhere((s) => s.id == setId);
    if (index <= 0 || index >= set.tiles.length) return;
    _spliceSet(setId, [
      set.copyWith(tiles: set.tiles.sublist(0, index)),
      TileSet(type: 'INVALID', tiles: set.tiles.sublist(index)),
    ]);
  }

  /// Replaces [setId] with the valid sets it cuts into; false if none exist.
  bool autoSplitSet(String setId) {
    final set = _gameState.board.firstWhere((s) => s.id == setId);
    final pieces = SetValidator.splitIntoValidSets(set.tiles);
    if (pieces == null || pieces.length < 2) return false;
    _spliceSet(setId, [
      set.copyWith(tiles: pieces.first),
      for (final piece in pieces.skip(1))
        TileSet(type: 'INVALID', tiles: piece),
    ]);
    return true;
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
    _commit(_gameState.updateSetOnBoard(
      setId,
      set.copyWith(tiles: sortSetTiles(newTiles)),
    ));
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
        _gameState.updateSetOnBoard(
          setId,
          set.copyWith(tiles: sortSetTiles(newTiles)),
        ),
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
        board = [
          ...board,
          TileSet(type: 'INVALID', tiles: [tile])
        ];
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
      // Like every other edit: a tile dropped at the end of a run still lands
      // in order. Sets with a joker keep the position the tile was given.
      return set.copyWith(tiles: sortSetTiles(tiles));
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
