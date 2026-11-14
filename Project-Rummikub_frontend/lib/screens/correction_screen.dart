import 'dart:convert';
import 'dart:math' as math;
import 'dart:ui' as ui;

import 'package:firebase_auth/firebase_auth.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';

import '../logic/set_validator.dart';
import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/file_download.dart';
import '../services/firebase_service.dart';
import '../services/roboflow_debug_dump.dart';
import '../services/vision_parser.dart';
import '../widgets/rack_widget.dart';
import '../widgets/rummikub_tiles.dart';
import '../widgets/tile_picker_popover.dart';
import '../widgets/tile_widget.dart';
import 'result_screen.dart';

/// Human-in-the-loop editor: tap a tile, then tap a set's move icon (or the
/// Rack slot) to move it.
///
/// Tiles are edited or deleted with their own buttons, not by tapping them.
/// Invalid sets get a red border.
class CorrectionScreen extends StatefulWidget {
  const CorrectionScreen({super.key, this.arenaChallenge});

  /// [ArenaChallenge.source] of puzzles from the backend's `/challenge/list`;
  /// their submit is scored by the backend and counts toward the leaderboard.
  static const communitySource = 'community';

  final ArenaChallenge? arenaChallenge;

  bool get isArena => arenaChallenge != null;

  @override
  State<CorrectionScreen> createState() => _CorrectionScreenState();
}

class _CorrectionScreenState extends State<CorrectionScreen> {
  bool _isBusy = false;
  String? _selectedTileId;
  String? _selectedSetId;

  bool get _hasSelection => _selectedTileId != null;

  void _clearSelection() {
    setState(() {
      _selectedTileId = null;
      _selectedSetId = null;
    });
  }

  Future<void> _onSolvePressed(GameState currentState) async {
    if (_isBusy) return;
    setState(() => _isBusy = true);
    try {
      final game = context.read<GameProvider>();
      final userId = _currentUserId();
      final turn = userId == null
          ? null
          : await game.recordTurn(userId, state: currentState);
      final result = await game.api.solveBoard(
        currentState,
        sessionId: turn?.sessionId,
        turnIndex: turn?.turnIndex,
      );
      if (!mounted) return;
      final vision = context.read<GameProvider>().visionOutput;
      if (vision != null) {
        try {
          await FirebaseService().saveVisionCorrection(
            visionOutput: vision,
            userCorrectedOutput: currentState,
          );
        } catch (_) {
          // Training log must not block the solve result.
        }
        if (!mounted) return;
        context.read<GameProvider>().clearVisionOutput();
      }
      context.read<GameProvider>().updateGameState(result.optimalState);
      if (!mounted) return;
      await Navigator.push<void>(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            beforeState: result.originalState,
            afterState: result.optimalState,
            tilesDropped: result.tilesUsedFromRack,
            executionTimeMs: result.executionTimeMs,
            solutionSteps: result.solutionSteps,
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_messageFromError(error)),
          backgroundColor: Theme.of(context).colorScheme.error,
        ),
      );
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  static String? _currentUserId() {
    try {
      return FirebaseAuth.instance.currentUser?.uid;
    } catch (_) {
      return null; // Firebase is not initialised (tests).
    }
  }

  /// Ends the game and goes back to the capture screen, ready for new photos.
  void _onEndGame() {
    context.read<GameProvider>().endGame();
    Navigator.of(context).popUntil((route) => route.isFirst);
  }

  Future<void> _onArenaSubmit(GameState currentState) async {
    if (_isBusy) return;
    final challenge = widget.arenaChallenge!;
    final (ok, message) = SetValidator.validateBoard(currentState.board);
    if (!ok) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text(message)),
      );
      return;
    }

    final score = rackTilesPlaced(challenge.initialState, currentState);

    setState(() => _isBusy = true);
    try {
      ChallengeResult? verdict;
      if (challenge.source == CorrectionScreen.communitySource) {
        final userId = _currentUserId();
        if (userId == null) {
          throw Exception('Sign in to submit a community challenge.');
        }
        verdict = await context.read<GameProvider>().api.submitChallenge(
              challengeId: challenge.id,
              userId: userId,
              board: currentState,
            );
      }
      if (!mounted) return;
      await Navigator.push<void>(
        context,
        MaterialPageRoute(
          builder: (_) => ResultScreen(
            beforeState: challenge.initialState,
            afterState: currentState,
            tilesDropped: score,
            algorithmScore: challenge.algorithmScore,
            challengeId: challenge.id,
            algorithmAfterState: challenge.algorithmAfterState,
            relativeScore: verdict?.score,
          ),
        ),
      );
    } catch (error) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(_messageFromError(error)),
          backgroundColor: Theme.of(context).colorScheme.error,
        ),
      );
    } finally {
      if (mounted) setState(() => _isBusy = false);
    }
  }

  String _messageFromError(Object error) {
    final text = error.toString();
    const prefix = 'Exception: ';
    return text.startsWith(prefix) ? text.substring(prefix.length) : text;
  }

  /// Selects [tile]; tapping the selected tile again deselects it. Editing
  /// goes through the tile's own Edit button, never a tap.
  void _onTileTap(Tile tile, {String? setId}) {
    if (_selectedTileId == tile.id && _selectedSetId == setId) {
      _clearSelection();
      return;
    }
    setState(() {
      _selectedTileId = tile.id;
      _selectedSetId = setId;
    });
  }

  void _onDrop(TileDestination destination) {
    if (_selectedTileId == null) return;
    context.read<GameProvider>().moveTile(
          tileId: _selectedTileId!,
          fromSetId: _selectedSetId,
          destination: destination,
        );
    _clearSelection();
  }

  Future<void> _exportRoboflowDump(RoboflowDebugDump dump) async {
    downloadBytes(dump.tableJpeg, 'rummikub-table-sent.jpg', 'image/jpeg');
    await Future<void>.delayed(const Duration(milliseconds: 350));
    downloadBytes(dump.rackJpeg, 'rummikub-rack-sent.jpg', 'image/jpeg');
    await Future<void>.delayed(const Duration(milliseconds: 350));
    downloadBytes(
      Uint8List.fromList(utf8.encode(dump.tableJson)),
      'rummikub-table-workflow.json',
      'application/json',
    );
    await Future<void>.delayed(const Duration(milliseconds: 350));
    downloadBytes(
      Uint8List.fromList(utf8.encode(dump.rackJson)),
      'rummikub-rack-workflow.json',
      'application/json',
    );
    await Clipboard.setData(
      ClipboardData(
        text:
            '${dump.unfilteredSummary}\n\n=== TABLE WORKFLOW JSON ===\n${dump.tableJson}\n\n=== RACK WORKFLOW JSON ===\n${dump.rackJson}',
      ),
    );
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      const SnackBar(
        content: Text(
          'Downloaded the exact JPEGs + JSON. JSON also copied. Send those files to the Roboflow agent.',
        ),
      ),
    );
  }

  Future<void> _addTile() async {
    final result = await showDialog<TileEditResult>(
      context: context,
      builder: (context) => const TileEditDialog(allowDelete: false),
    );
    if (!mounted || result == null || result.delete) return;
    context.read<GameProvider>().addTileToRack(
          Tile(color: result.color, value: result.value),
        );
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<GameProvider>(
      builder: (context, gameProvider, _) {
        final gameState = gameProvider.gameState;
        final groupingOff = VisionParser.disableSpatialGrouping &&
            gameProvider.visionOutput != null;
        final dump = gameProvider.lastRoboflowDump;
        final (boardValid, boardError) =
            SetValidator.validateBoard(gameState.board);
        final hasTiles = !gameState.isBoardEmpty || !gameState.isRackEmpty;
        final canSubmit = !groupingOff && boardValid && hasTiles && !_isBusy;
        final hint = groupingOff
            ? (_hasSelection
                ? 'Use the selected tile\'s edit or delete button, or tap it again to deselect. Grouping is off — these are raw detections.'
                : 'Raw detections with no series grouping. Tap a tile to select it, then use its edit or delete button.')
            : (_hasSelection
                ? 'Tap the move icon on a set (or the Rack slot) to place the tile, or tap it again to deselect.'
                : 'Tap a tile to select it, then tap the move icon on a set to place it. Hover a tile for edit, delete and ✂ (split the set after it); the merge icon under a set unites it with the next one.');

        return Stack(
          children: [
            Scaffold(
              appBar: AppBar(
                title: Text(widget.isArena ? 'Challenge' : 'Correct Tiles'),
                actions: [
                  if (!widget.isArena)
                    TextButton.icon(
                      key: const ValueKey('end-game'),
                      onPressed: _isBusy ? null : _onEndGame,
                      icon: const Icon(Icons.stop_circle_outlined),
                      label: const Text('End game'),
                    ),
                  IconButton(
                    tooltip: 'Add tile',
                    onPressed: _addTile,
                    icon: const Icon(Icons.add),
                  ),
                ],
              ),
              body: ListView(
                padding: const EdgeInsets.fromLTRB(16, 16, 16, 88),
                children: [
                  Text(
                    hint,
                    style: Theme.of(context).textTheme.bodyMedium,
                  ),
                  if (gameProvider.lastVisionDebug != null) ...[
                    const SizedBox(height: 8),
                    Text(
                      gameProvider.lastVisionDebug!,
                      style: Theme.of(context).textTheme.bodySmall,
                    ),
                  ],
                  if (gameProvider.lastRoboflowDump != null) ...[
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      onPressed: () => _exportRoboflowDump(
                        gameProvider.lastRoboflowDump!,
                      ),
                      icon: const Icon(Icons.download),
                      label: const Text('Download JPEG + JSON for Roboflow'),
                    ),
                  ],
                  if (!groupingOff && !boardValid && hasTiles) ...[
                    const SizedBox(height: 8),
                    Text(
                      boardError,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                      ),
                    ),
                  ],
                  const SizedBox(height: 16),
                  if (groupingOff) ...[
                    SectionHeader(
                      title: 'Rack',
                      subtitle: '${gameState.rackSize} raw tiles',
                    ),
                    const SizedBox(height: 8),
                    if (dump != null && dump.rackDetections.isNotEmpty)
                      _RawDetectionMap(
                        jpeg: dump.rackJpeg,
                        detections: dump.rackDetections,
                        onTapIndex: (index) {
                          if (index < gameState.rack.length) {
                            _onTileTap(gameState.rack[index]);
                          }
                        },
                      ),
                    const SizedBox(height: 8),
                    _ungroupedTileWrap(
                      tiles: [
                        for (final tile in gameState.rack) (tile, null),
                      ],
                      extra: EmptySlot(
                        label: _hasSelection ? 'Rack' : '+',
                        onTap: _hasSelection
                            ? () => _onDrop(TileDestination.rack)
                            : _addTile,
                      ),
                    ),
                    const SizedBox(height: 24),
                    SectionHeader(
                      title: 'Table',
                      subtitle:
                          '${gameState.totalTilesOnBoard} raw tiles (not grouped)',
                    ),
                    const SizedBox(height: 8),
                    if (dump != null && dump.tableDetections.isNotEmpty)
                      _RawDetectionMap(
                        jpeg: dump.tableJpeg,
                        detections: dump.tableDetections,
                        onTapIndex: (index) {
                          if (index < gameState.board.length &&
                              gameState.board[index].tiles.isNotEmpty) {
                            final set = gameState.board[index];
                            _onTileTap(set.tiles.first, setId: set.id);
                          }
                        },
                      ),
                    const SizedBox(height: 8),
                    if (gameState.isBoardEmpty)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Text('No table detections.'),
                      )
                    else
                      _ungroupedTileWrap(
                        tiles: [
                          for (final set in gameState.board)
                            for (final tile in set.tiles) (tile, set.id),
                        ],
                      ),
                  ] else ...[
                    Row(
                      children: [
                        Expanded(
                          child: SectionHeader(
                            title: 'Board',
                            subtitle: '${gameState.numberOfSets} sets',
                          ),
                        ),
                        IconButton(
                          key: const Key('add_set'),
                          tooltip: 'New set',
                          icon: const Icon(Icons.add_circle),
                          onPressed: () => gameProvider.addSetToBoard(
                            TileSet(type: 'INVALID', tiles: []),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 8),
                    if (gameState.isBoardEmpty)
                      const Padding(
                        padding: EdgeInsets.symmetric(vertical: 8),
                        child: Text(
                          'No sets on the board. Tap + above to add a new set.',
                        ),
                      )
                    else
                      ...gameState.board.asMap().entries.map((entry) {
                        final set = entry.value;
                        final isLast =
                            entry.key == gameState.board.length - 1;
                        return _SetWithToolbar(
                          onDelete: () =>
                              gameProvider.removeSetFromBoard(set.id),
                          onTilesAdded: (tiles) =>
                              gameProvider.updateSetOnBoard(
                            set.id,
                            set.copyWith(tiles: [...set.tiles, ...tiles]),
                          ),
                          onMoveHere: _hasSelection
                              ? () => _onDrop(
                                    TileDestination.insertIntoSet(
                                      set.id,
                                      set.tiles.length,
                                    ),
                                  )
                              : null,
                          child: BoardSetCard(
                            setIndex: entry.key,
                            tileSet: set,
                            selectedTileId: _selectedTileId,
                            onTileTap: (tile) =>
                                _onTileTap(tile, setId: set.id),
                            onTileEdited: (original, updated) =>
                                gameProvider.replaceTile(
                              original,
                              updated,
                              setId: set.id,
                            ),
                            onTileDeleted: (tile) =>
                                gameProvider.deleteTile(tile, setId: set.id),
                            onSplitAfter: (index) {
                              gameProvider.splitSet(set.id, index + 1);
                              _clearSelection();
                            },
                            onUniteWithNext: isLast
                                ? null
                                : () {
                                    gameProvider.mergeWithNextSet(set.id);
                                    _clearSelection();
                                  },
                            onAutoSplit: !_hasSelection &&
                                    set.isInvalid &&
                                    (SetValidator.splitIntoValidSets(set.tiles)
                                                ?.length ??
                                            0) >
                                        1
                                ? () => gameProvider.autoSplitSet(set.id)
                                : null,
                          ),
                        );
                      }),
                    const SizedBox(height: 24),
                    RackWidget(
                      tiles: gameState.rack,
                      selectedTileId:
                          _selectedSetId == null ? _selectedTileId : null,
                      onTileTap: _onTileTap,
                      onTileEdited: (original, updated) =>
                          gameProvider.replaceTile(original, updated),
                      onTileDeleted: (tile) => gameProvider.deleteTile(tile),
                      onClear: () => gameProvider.updateRack(const []),
                      onTilesAdded: (tiles) {
                        for (final tile in tiles) {
                          gameProvider.addTileToRack(tile);
                        }
                      },
                      extra: EmptySlot(
                        label: _hasSelection ? 'Rack' : '+',
                        onTap: _hasSelection
                            ? () => _onDrop(TileDestination.rack)
                            : _addTile,
                      ),
                    ),
                  ],
                ],
              ),
              floatingActionButton: groupingOff && !widget.isArena
                  ? null
                  : FloatingActionButton.extended(
                      onPressed: canSubmit
                          ? () => widget.isArena
                              ? _onArenaSubmit(gameState)
                              : _onSolvePressed(gameState)
                          : null,
                      icon: Icon(
                        widget.isArena ? Icons.check : Icons.play_arrow,
                      ),
                      label: Text(widget.isArena ? 'Submit' : 'Solve'),
                    ),
            ),
            if (_isBusy)
              const Positioned.fill(
                child: AbsorbPointer(
                  child: ColoredBox(
                    color: Color(0x80000000),
                    child: Center(child: CircularProgressIndicator()),
                  ),
                ),
              ),
          ],
        );
      },
    );
  }

  Widget _ungroupedTileWrap({
    required List<(Tile tile, String? setId)> tiles,
    Widget? extra,
  }) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        for (final entry in tiles)
          TileWidget(
            tile: entry.$1,
            selected:
                entry.$1.id == _selectedTileId && _selectedSetId == entry.$2,
            onTap: () => _onTileTap(entry.$1, setId: entry.$2),
            onEdit: (updated) {
              context
                  .read<GameProvider>()
                  .replaceTile(entry.$1, updated, setId: entry.$2);
              _clearSelection();
            },
            onDelete: () {
              context.read<GameProvider>().deleteTile(entry.$1, setId: entry.$2);
              _clearSelection();
            },
          ),
        if (extra != null) extra,
      ],
    );
  }
}

class _RawDetectionMap extends StatefulWidget {
  const _RawDetectionMap({
    required this.jpeg,
    required this.detections,
    this.onTapIndex,
  });

  final Uint8List jpeg;
  final List<TileDetection> detections;
  final ValueChanged<int>? onTapIndex;

  @override
  State<_RawDetectionMap> createState() => _RawDetectionMapState();
}

class _RawDetectionMapState extends State<_RawDetectionMap> {
  ui.Image? _image;
  var _ready = false;

  @override
  void initState() {
    super.initState();
    _decode();
  }

  @override
  void didUpdateWidget(_RawDetectionMap oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!identical(oldWidget.jpeg, widget.jpeg)) {
      _ready = false;
      _decode();
    }
  }

  Future<void> _decode() async {
    try {
      final image = await decodeImageFromList(widget.jpeg);
      if (!mounted) return;
      setState(() {
        _image = image;
        _ready = true;
      });
    } catch (_) {
      if (!mounted) return;
      setState(() {
        _image = null;
        _ready = true;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_ready) {
      return const SizedBox(
        height: 160,
        child: Center(child: CircularProgressIndicator()),
      );
    }

    final image = _image;
    if (image == null) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(12),
        child: Image.memory(
          widget.jpeg,
          height: 220,
          width: double.infinity,
          fit: BoxFit.contain,
        ),
      );
    }

    return LayoutBuilder(
      builder: (context, constraints) {
        const maxHeight = 360.0;
        final scale = math.min(
          constraints.maxWidth / image.width,
          maxHeight / image.height,
        );
        final width = image.width * scale;
        final height = image.height * scale;
        return Center(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(12),
            child: SizedBox(
              width: width,
              height: height,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: Image.memory(widget.jpeg, fit: BoxFit.fill),
                  ),
                  for (var i = 0; i < widget.detections.length; i++)
                    _positionedDetection(widget.detections[i], scale, i),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _positionedDetection(
    TileDetection detection,
    double scale,
    int index,
  ) {
    final boxWidth = math.max(28.0, detection.width * scale);
    final boxHeight = math.max(36.0, detection.height * scale);
    final left = (detection.x - detection.width / 2) * scale;
    final top = (detection.y - detection.height / 2) * scale;
    final foreground =
        detection.color == 'YELLOW' ? Colors.black87 : Colors.white;
    return Positioned(
      left: left,
      top: top,
      width: boxWidth,
      height: boxHeight,
      child: Material(
        color: colorForTile(detection.color).withValues(alpha: 0.86),
        borderRadius: BorderRadius.circular(4),
        child: InkWell(
          onTap: widget.onTapIndex == null
              ? null
              : () => widget.onTapIndex!(index),
          borderRadius: BorderRadius.circular(4),
          child: DecoratedBox(
            decoration: BoxDecoration(
              border: Border.all(color: Colors.white, width: 1.5),
              borderRadius: BorderRadius.circular(4),
            ),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  detection.shortLabel,
                  style: TextStyle(
                    color: foreground,
                    fontWeight: FontWeight.bold,
                    fontSize: boxHeight < 28 ? 10 : 16,
                  ),
                ),
                if (boxHeight >= 44)
                  Text(
                    detection.confidence.toStringAsFixed(2),
                    style: TextStyle(
                      color: foreground,
                      fontSize: 9,
                    ),
                  ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

/// A board set with a row above it holding the delete, add-tiles `+` and,
/// while a tile is selected, move-here buttons.
class _SetWithToolbar extends StatelessWidget {
  const _SetWithToolbar({
    required this.onTilesAdded,
    required this.onDelete,
    required this.child,
    this.onMoveHere,
  });

  final ValueChanged<List<Tile>> onTilesAdded;
  final VoidCallback onDelete;
  final Widget child;

  /// Set only while a tile is selected: the tap-to-move target for this set.
  final VoidCallback? onMoveHere;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            if (onMoveHere != null)
              IconButton(
                key: const Key('set_move_here'),
                tooltip: 'Move selected tile here',
                visualDensity: VisualDensity.compact,
                iconSize: 20,
                style: IconButton.styleFrom(
                  backgroundColor:
                      Theme.of(context).colorScheme.primaryContainer,
                ),
                icon: const Icon(Icons.move_down),
                onPressed: onMoveHere,
              ),
            IconButton(
              key: const Key('set_delete'),
              tooltip: 'Delete set',
              visualDensity: VisualDensity.compact,
              iconSize: 20,
              icon: const Icon(Icons.delete_outline),
              onPressed: onDelete,
            ),
            TilePickerButton(
              onTilesSelected: onTilesAdded,
              tooltip: 'Add tiles to set',
            ),
          ],
        ),
        child,
      ],
    );
  }
}
