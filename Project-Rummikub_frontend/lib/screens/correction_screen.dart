import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../logic/set_validator.dart';
import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/api_service.dart';
import '../services/firebase_service.dart';
import '../widgets/rummikub_tiles.dart';
import 'result_screen.dart';

/// Human-in-the-loop editor: tap a tile, then tap a + slot to move it.
///
/// Tap the same tile again to edit or delete. Invalid sets get a red border.
class CorrectionScreen extends StatefulWidget {
  const CorrectionScreen({super.key, this.arenaChallenge});

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
      final result = await ApiService().solveBoard(currentState);
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

    final initialRackIds =
        challenge.initialState.rack.map((tile) => tile.id).toSet();
    final boardIds = currentState.board
        .expand((set) => set.tiles)
        .map((tile) => tile.id)
        .toSet();
    final score = initialRackIds.where(boardIds.contains).length;

    setState(() => _isBusy = true);
    try {
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
          ),
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

  void _onTileTap(Tile tile, {String? setId}) {
    if (_selectedTileId == tile.id && _selectedSetId == setId) {
      _openEditDialog(tile, setId: setId);
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

  Future<void> _openEditDialog(Tile tile, {String? setId}) async {
    final result = await showDialog<TileEditResult>(
      context: context,
      builder: (context) => TileEditDialog(tile: tile),
    );
    if (!mounted || result == null) return;
    final provider = context.read<GameProvider>();
    if (result.delete) {
      provider.deleteTile(tile, setId: setId);
    } else {
      provider.replaceTile(
        tile,
        tile.copyWith(color: result.color, value: result.value),
        setId: setId,
      );
    }
    _clearSelection();
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
        final (boardValid, boardError) =
            SetValidator.validateBoard(gameState.board);
        final hasTiles = !gameState.isBoardEmpty || !gameState.isRackEmpty;
        final canSubmit = boardValid && hasTiles && !_isBusy;
        final hint = _hasSelection
            ? 'Tap + to the left or right of any tile to place it, or tap it again to edit.'
            : 'Tap a tile to select it, then tap + on either side of a tile to place it.';

        return Stack(
          children: [
            Scaffold(
              appBar: AppBar(
                title: Text(widget.isArena ? 'Challenge' : 'Correct Tiles'),
                actions: [
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
                  if (!boardValid && hasTiles) ...[
                    const SizedBox(height: 8),
                    Text(
                      boardError,
                      style: TextStyle(
                        color: Theme.of(context).colorScheme.error,
                      ),
                    ),
                  ],
                  const SizedBox(height: 16),
                  SectionHeader(
                    title: 'Board',
                    subtitle: '${gameState.numberOfSets} sets',
                  ),
                  const SizedBox(height: 8),
                  if (gameState.isBoardEmpty)
                    const Padding(
                      padding: EdgeInsets.symmetric(vertical: 8),
                      child: Text('No sets on the board. Add a new set below.'),
                    )
                  else
                    ...gameState.board.asMap().entries.map((entry) {
                      final set = entry.value;
                      return BoardSetCard(
                        setIndex: entry.key,
                        tileSet: set,
                        selectedTileId: _selectedTileId,
                        onTileTap: (tile) => _onTileTap(tile, setId: set.id),
                        onInsertAt: _hasSelection
                            ? (index) => _onDrop(
                                  TileDestination.insertIntoSet(set.id, index),
                                )
                            : null,
                      );
                    }),
                  const SizedBox(height: 8),
                  _DropZone(
                    label: 'New set',
                    enabled: _hasSelection,
                    onTap: () => _onDrop(TileDestination.newSet),
                  ),
                  const SizedBox(height: 24),
                  SectionHeader(
                    title: 'Rack',
                    subtitle: '${gameState.rackSize} tiles',
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: [
                      ...gameState.rack.map(
                        (tile) => TileView(
                          tile: tile,
                          selected: tile.id == _selectedTileId &&
                              _selectedSetId == null,
                          onTap: () => _onTileTap(tile),
                        ),
                      ),
                      EmptySlot(
                        label: _hasSelection ? 'Rack' : '+',
                        onTap: _hasSelection
                            ? () => _onDrop(TileDestination.rack)
                            : _addTile,
                      ),
                    ],
                  ),
                ],
              ),
              floatingActionButton: FloatingActionButton.extended(
                onPressed: canSubmit
                    ? () => widget.isArena
                        ? _onArenaSubmit(gameState)
                        : _onSolvePressed(gameState)
                    : null,
                icon: Icon(widget.isArena ? Icons.check : Icons.play_arrow),
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
}

class _DropZone extends StatelessWidget {
  const _DropZone({
    required this.label,
    required this.enabled,
    required this.onTap,
  });

  final String label;
  final bool enabled;
  final VoidCallback onTap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      color: enabled
          ? theme.colorScheme.primaryContainer
          : theme.colorScheme.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(12),
      child: InkWell(
        onTap: enabled ? onTap : null,
        borderRadius: BorderRadius.circular(12),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 16),
          child: Center(
            child: Text(
              label,
              style: theme.textTheme.labelLarge,
            ),
          ),
        ),
      ),
    );
  }
}
