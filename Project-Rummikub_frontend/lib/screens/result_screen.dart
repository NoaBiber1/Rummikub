import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/firebase_service.dart';
import '../widgets/rummikub_tiles.dart';

/// Shows before vs after. Saves history + global moves unless [readOnly].
class ResultScreen extends StatefulWidget {
  const ResultScreen({
    super.key,
    required this.beforeState,
    required this.afterState,
    required this.tilesDropped,
    this.algorithmScore,
    this.challengeId,
    this.algorithmAfterState,
    this.readOnly = false,
  });

  final GameState beforeState;
  final GameState afterState;
  final int tilesDropped;
  final int? algorithmScore;

  /// When set, this is a human arena attempt (not a solver publish).
  final String? challengeId;

  /// Solver output for this puzzle, shown after a community attempt.
  final GameState? algorithmAfterState;
  final bool readOnly;

  @override
  State<ResultScreen> createState() => _ResultScreenState();
}

class _ResultScreenState extends State<ResultScreen> {
  String? _saveError;
  bool _saved = false;

  Set<String> get _movedFromRack {
    final beforeRack = widget.beforeState.rack.map((t) => t.id).toSet();
    final afterRack = widget.afterState.rack.map((t) => t.id).toSet();
    return beforeRack.difference(afterRack);
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
      final firebase = FirebaseService();
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
        setState(() => _saveError = error.toString());
      }
    }
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
    return 'Solution';
  }

  @override
  Widget build(BuildContext context) {
    final moved = _movedFromRack;
    final algorithmScore = widget.algorithmScore;

    return Scaffold(
      appBar: AppBar(
        title: Text(widget.readOnly ? 'Game details' : 'Result'),
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Text(
            _headline,
            style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
          ),
          const SizedBox(height: 8),
          Text('Tiles dropped from rack: ${widget.tilesDropped}'),
          if (algorithmScore != null)
            Text('Algorithm score: $algorithmScore'),
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
                style: TextStyle(color: Theme.of(context).colorScheme.error),
              ),
            ),
          const SizedBox(height: 24),
          Text(
            'After',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 8),
          Text(
            moved.isEmpty
                ? 'No rack tiles were placed.'
                : 'Highlighted tiles came from the rack.',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 8),
          GameBoardView(
            gameState: widget.afterState,
            highlightedTileIds: moved,
          ),
          const SizedBox(height: 32),
          Text(
            'Before',
            style: Theme.of(context).textTheme.titleLarge,
          ),
          const SizedBox(height: 8),
          GameBoardView(gameState: widget.beforeState),
          if (widget.algorithmAfterState != null) ...[
            const SizedBox(height: 32),
            Text(
              'Algorithm solution',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            const SizedBox(height: 8),
            Text(
              'The board after the solver move (${widget.algorithmScore ?? 0} tiles dropped).',
              style: Theme.of(context).textTheme.bodySmall,
            ),
            const SizedBox(height: 8),
            GameBoardView(gameState: widget.algorithmAfterState!),
          ],
        ],
      ),
    );
  }
}
