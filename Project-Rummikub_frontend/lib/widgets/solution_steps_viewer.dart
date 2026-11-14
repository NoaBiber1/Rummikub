import 'package:flutter/material.dart';

import '../models/models.dart';
import 'rummikub_tiles.dart';

/// Walks through `solution_steps` one at a time: `Source ➔ Target`, the tiles
/// that move, and Previous / Next controls.
class SolutionStepsViewer extends StatefulWidget {
  const SolutionStepsViewer({super.key, required this.steps});

  final List<SolutionStep> steps;

  @override
  State<SolutionStepsViewer> createState() => _SolutionStepsViewerState();
}

class _SolutionStepsViewerState extends State<SolutionStepsViewer> {
  int _index = 0;

  @override
  void didUpdateWidget(SolutionStepsViewer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (_index >= widget.steps.length) {
      _index = widget.steps.isEmpty ? 0 : widget.steps.length - 1;
    }
  }

  static String _sourceLabel(SolutionStep step) {
    if (step.type == SolutionStepType.placeFromRack) return 'Rack';
    if (step.sourceSetId == null) return 'New set';
    return 'Set ${step.sourceSetId}';
  }

  static String _targetLabel(SolutionStep step) {
    if (step.targetSetId == null) return 'Free';
    return 'Set ${step.targetSetId}';
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final steps = widget.steps;

    if (steps.isEmpty) {
      return Text(
        'No steps: the board is already optimal.',
        style: theme.textTheme.bodyMedium?.copyWith(
          color: theme.colorScheme.onSurfaceVariant,
        ),
      );
    }

    final step = steps[_index];
    final isFirst = _index == 0;
    final isLast = _index == steps.length - 1;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        Text(
          'Step ${_index + 1} of ${steps.length}',
          style: theme.textTheme.titleMedium,
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Chip(label: Text(_sourceLabel(step))),
            const Padding(
              padding: EdgeInsets.symmetric(horizontal: 8),
              child: Icon(Icons.arrow_forward),
            ),
            Chip(label: Text(_targetLabel(step))),
          ],
        ),
        const SizedBox(height: 8),
        Wrap(
          spacing: 4,
          runSpacing: 8,
          children: step.tiles.map((t) => TileView(tile: t)).toList(),
        ),
        const SizedBox(height: 8),
        Text(step.description, style: theme.textTheme.bodyMedium),
        const SizedBox(height: 12),
        Row(
          children: [
            OutlinedButton.icon(
              onPressed: isFirst ? null : () => setState(() => _index--),
              icon: const Icon(Icons.chevron_left),
              label: const Text('Previous Step'),
            ),
            const SizedBox(width: 12),
            FilledButton.icon(
              onPressed: isLast ? null : () => setState(() => _index++),
              icon: const Icon(Icons.chevron_right),
              label: const Text('Next Step'),
            ),
          ],
        ),
      ],
    );
  }
}
