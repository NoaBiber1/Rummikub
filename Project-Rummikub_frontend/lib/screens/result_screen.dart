import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/firebase_service.dart';
import '../widgets/rummikub_tiles.dart';
import '../widgets/solution_steps_viewer.dart';

/// Shows before vs after. Saves history + global moves unless [readOnly].
class ResultScreen extends StatefulWidget {
  const ResultScreen({
    super.key,
    required this.beforeState,
    required this.afterState,
    required this.tilesDropped,
    this.executionTimeMs,
    this.algorithmScore,
    this.challengeId,
    this.algorithmAfterState,
    this.relativeScore,
    this.solutionSteps = const [],
    this.readOnly = false,
    this.firebaseService,
  });

  final GameState beforeState;
  final GameState afterState;
  final int tilesDropped;

  /// Solver time reported by `/solve`; null for arena attempts and history.
  final int? executionTimeMs;
  final int? algorithmScore;

  /// When set, this is a human arena attempt (not a solver publish).
  final String? challengeId;

  /// Solver output for this puzzle, shown after a community attempt.
  final GameState? algorithmAfterState;

  /// Backend score of a community attempt: tiles placed minus the algorithm's.
  final int? relativeScore;

  /// Step-by-step moves from `/solve`; the walkthrough is hidden when empty.
  final List<SolutionStep> solutionSteps;
  final bool readOnly;

  /// Test seam; defaults to the Firestore-backed service.
  final FirebaseService? firebaseService;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  String? _saveError;
  bool _saved = false;

  /// Rack tiles that ended up on the board, in rack order.
  List<Tile> get _placedFromRack {
    final afterRack = widget.afterState.rack.map((t) => t.id).toSet();
    return widget.beforeState.rack
        .where((tile) => !afterRack.contains(tile.id))
        .toList();
  }

  @override
  void initState() {
    super.initState();
    if (!widget.readOnly) {
      _save();
    }
  }

  Future<void> _save() async {
    try {
      final firebase = widget.firebaseService ?? FirebaseService();
      final challengeId = widget.challengeId;
      if (challengeId != null) {
        await firebase.saveArenaAttempt(
          challengeId: challengeId,
          beforeState: widget.beforeState,
          afterState: widget.afterState,
          tilesDropped: widget.tilesDropped,
          algorithmScore: widget.algorithmScore ?? 0,
        );
      } else {
        await firebase.saveSolverSolution(
          beforeState: widget.beforeState,
          afterState: widget.afterState,
          tilesDropped: widget.tilesDropped,
        );
      }
      if (mounted) {
        context.read<GameProvider>().updateGameState(widget.afterState);
        setState(() => _saved = true);
      }
    } catch (error) {
      if (mounted) {
        setState(() => _saveError = _readable(error));
      }
    }
  }

  static String _readable(Object error) {
    final text = error.toString();
    const prefix = 'Exception: ';
    return text.startsWith(prefix) ? text.substring(prefix.length) : text;
  }

  String get _headline {
    if (widget.afterState.isRackEmpty && widget.tilesDropped > 0) {
      return 'You Won! Rack is empty.';
    }
    if (widget.algorithmScore != null) {
      final target = widget.algorithmScore!;
      if (widget.tilesDropped >= target) {
        return widget.tilesDropped > target
            ? 'You beat the algorithm!'
            : 'You matched the algorithm.';
      }
      return 'Under the algorithm score.';
    }
    if (widget.tilesDropped == 0) {
      return 'No legal move this turn.';
    }
    return 'Solution';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final placed = _placedFromRack;
    final moved = placed.map((tile) => tile.id).toSet();
    final algorithmScore = widget.algorithmScore;
    final executionTimeMs = widget.executionTimeMs;

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.readOnly ? 'Game details' : 'Result'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            _headline,
            style: theme.textTheme.headlineSmall?.copyWith(
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 8),
          Text('Tiles dropped from rack: ${widget.tilesDropped}'),
          if (executionTimeMs != null) Text('Solved in $executionTimeMs ms'),
          if (algorithmScore != null) Text('Algorithm score: $algorithmScore'),
          if (widget.relativeScore != null)
            Text(
              'Score vs algorithm: ${widget.relativeScore! > 0 ? '+' : ''}'
              '${widget.relativeScore}',
              key: const ValueKey('relative-score'),
            ),
          if (!widget.readOnly && !_saved && _saveError == null)
            const Padding(
              padding: EdgeInsets.only(top: 8),
              child: Text('Saving to history...'),
            ),
          if (!widget.readOnly && _saved)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                widget.challengeId != null
                    ? 'Saved to history.'
                    : 'Saved to history and published to community challenges.',
              ),
            ),
          if (_saveError != null)
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'Could not save: $_saveError',
                style: TextStyle(color: theme.colorScheme.error),
              ),
            ),
          const SizedBox(height: 24),
          Text('Placed from your rack', style: theme.textTheme.titleLarge),
          const SizedBox(height: 8),
          if (placed.isEmpty)
            Text(
              widget.tilesDropped == 0 && widget.algorithmScore == null
                  ? 'None of your tiles fit on the board. Keep them and draw a tile.'
                  : 'No rack tiles were placed.',
              style: theme.textTheme.bodyMedium,
            )
          else
            Wrap(
              key: const ValueKey('placed-from-rack'),
              spacing: 8,
              runSpacing: 8,
              children: [
                for (final tile in placed)
                  TileView(tile: tile, highlighted: true),
              ],
            ),
          if (widget.solutionSteps.isNotEmpty) ...[
            const SizedBox(height: 24),
            Text('How to get there', style: theme.textTheme.titleLarge),
            const SizedBox(height: 8),
            SolutionStepsViewer(
              key: const ValueKey('solution-steps'),
              steps: widget.solutionSteps,
            ),
          ],
          const SizedBox(height: 24),
          Text('After', style: theme.textTheme.titleLarge),
          const SizedBox(height: 8),
          Text(
            moved.isEmpty
                ? 'The board is unchanged.'
                : 'Highlighted tiles came from your rack.',
            style: theme.textTheme.bodySmall,
          ),
          const SizedBox(height: 8),
          GameBoardView(
            gameState: widget.afterState,
            highlightedTileIds: moved,
          ),
          const SizedBox(height: 32),
          Text('Before', style: theme.textTheme.titleLarge),
          const SizedBox(height: 8),
          GameBoardView(gameState: widget.beforeState),
          if (widget.algorithmAfterState != null) ...[
            const SizedBox(height: 32),
            Text(
              'Algorithm solution',
              style: theme.textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Text(
              'The board after the solver move (${widget.algorithmScore ?? 0} tiles dropped).',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            GameBoardView(gameState: widget.algorithmAfterState!),
          ],
        ],
      ),
    );
  }
}
