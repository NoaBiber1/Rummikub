import 'package:flutter/material.dart';

import '../models/models.dart';

const kTileColors = ['RED', 'BLUE', 'BLACK', 'YELLOW', 'JOKER'];

Color colorForTile(String color) {
  switch (color) {
    case 'RED':
      return Colors.red.shade700;
    case 'BLUE':
      return Colors.blue.shade700;
    case 'BLACK':
      return Colors.grey.shade900;
    case 'YELLOW':
      return Colors.amber.shade600;
    case 'JOKER':
      return Colors.purple.shade700;
    default:
      return Colors.grey;
  }
}

class TileView extends StatelessWidget {
  const TileView({
    super.key,
    required this.tile,
    this.onTap,
    this.selected = false,
    this.highlighted = false,
  });

  final Tile tile;
  final VoidCallback? onTap;
  final bool selected;
  final bool highlighted;

  @override
  Widget build(BuildContext context) {
    final background = colorForTile(tile.color);
    final foreground =
        tile.color == 'YELLOW' ? Colors.black87 : Colors.white;

    return Material(
      color: background,
      borderRadius: BorderRadius.circular(8),
      elevation: selected || highlighted ? 4 : 1,
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: Container(
          width: 48,
          height: 64,
          decoration: BoxDecoration(
            borderRadius: BorderRadius.circular(8),
            border: Border.all(
              color: selected
                  ? Theme.of(context).colorScheme.primary
                  : highlighted
                      ? Colors.lightGreenAccent
                      : Colors.transparent,
              width: selected || highlighted ? 3 : 0,
            ),
          ),
          child: Center(
            child: Text(
              tile.color == 'JOKER' ? 'J' : '${tile.value}',
              style: TextStyle(
                color: foreground,
                fontWeight: FontWeight.bold,
                fontSize: 20,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class EmptySlot extends StatelessWidget {
  const EmptySlot({
    super.key,
    required this.label,
    this.onTap,
    this.compact = false,
  });

  final String label;
  final VoidCallback? onTap;
  final bool compact;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Material(
      color: theme.colorScheme.surfaceContainerHighest,
      borderRadius: BorderRadius.circular(8),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(8),
        child: SizedBox(
          width: compact ? 28 : 48,
          height: 64,
          child: Center(
            child: Text(
              label,
              textAlign: TextAlign.center,
              style: theme.textTheme.labelSmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class BoardSetCard extends StatelessWidget {
  const BoardSetCard({
    super.key,
    required this.setIndex,
    required this.tileSet,
    this.selectedTileId,
    this.highlightedTileIds = const {},
    this.onTileTap,
    this.onInsertAt,
    this.readOnly = false,
  });

  final int setIndex;
  final TileSet tileSet;
  final String? selectedTileId;
  final Set<String> highlightedTileIds;
  final ValueChanged<Tile>? onTileTap;
  final ValueChanged<int>? onInsertAt;
  final bool readOnly;

  @override
  Widget build(BuildContext context) {
    final isInvalid = tileSet.isInvalid;
    final borderColor = isInvalid
        ? Theme.of(context).colorScheme.error
        : Theme.of(context).colorScheme.outlineVariant;

    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(12),
        side: BorderSide(color: borderColor, width: isInvalid ? 2 : 1),
      ),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Set ${setIndex + 1}  ·  ${tileSet.type}',
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: isInvalid
                        ? Theme.of(context).colorScheme.error
                        : null,
                  ),
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 4,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: _setChildren(),
            ),
          ],
        ),
      ),
    );
  }

  List<Widget> _setChildren() {
    final tiles = tileSet.tiles;
    final views = <Widget>[];
    final canInsert = !readOnly && onInsertAt != null;

    if (canInsert) {
      views.add(
        EmptySlot(
          label: '+',
          compact: true,
          onTap: () => onInsertAt!(0),
        ),
      );
    }

    for (var i = 0; i < tiles.length; i++) {
      final tile = tiles[i];
      views.add(
        TileView(
          tile: tile,
          selected: tile.id == selectedTileId,
          highlighted: highlightedTileIds.contains(tile.id),
          onTap: onTileTap == null ? null : () => onTileTap!(tile),
        ),
      );
      if (canInsert) {
        views.add(
          EmptySlot(
            label: '+',
            compact: true,
            onTap: () => onInsertAt!(i + 1),
          ),
        );
      }
    }

    return views;
  }
}

class SectionHeader extends StatelessWidget {
  const SectionHeader({super.key, required this.title, required this.subtitle});

  final String title;
  final String subtitle;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Row(
      children: [
        Text(
          title,
          style: theme.textTheme.titleLarge?.copyWith(
            fontWeight: FontWeight.bold,
          ),
        ),
        const SizedBox(width: 8),
        Text(
          subtitle,
          style: theme.textTheme.bodyMedium?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class GameBoardView extends StatelessWidget {
  const GameBoardView({
    super.key,
    required this.gameState,
    this.highlightedTileIds = const {},
  });

  final GameState gameState;
  final Set<String> highlightedTileIds;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        SectionHeader(
          title: 'Board',
          subtitle: '${gameState.numberOfSets} sets',
        ),
        const SizedBox(height: 8),
        if (gameState.isBoardEmpty)
          Text(
            'No sets on the board.',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
          )
        else
          ...gameState.board.asMap().entries.map(
            (entry) => BoardSetCard(
              setIndex: entry.key,
              tileSet: entry.value,
              highlightedTileIds: highlightedTileIds,
              readOnly: true,
            ),
          ),
        const SizedBox(height: 24),
        SectionHeader(
          title: 'Rack',
          subtitle: '${gameState.rackSize} tiles',
        ),
        const SizedBox(height: 8),
        if (gameState.isRackEmpty)
          Text(
            'No tiles in the rack.',
            style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                  color: Theme.of(context).colorScheme.onSurfaceVariant,
                ),
          )
        else
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: gameState.rack
                .map(
                  (tile) => TileView(
                    tile: tile,
                    highlighted: highlightedTileIds.contains(tile.id),
                  ),
                )
                .toList(),
          ),
      ],
    );
  }
}

class TileEditResult {
  const TileEditResult.save({required this.color, required this.value})
      : delete = false;

  const TileEditResult.delete()
      : delete = true,
        color = 'RED',
        value = 1;

  final bool delete;
  final String color;
  final int value;
}

class TileEditDialog extends StatefulWidget {
  const TileEditDialog({
    super.key,
    this.tile,
    this.allowDelete = true,
  });

  final Tile? tile;
  final bool allowDelete;

  @override
  State<TileEditDialog> createState() => _TileEditDialogState();
}

class _TileEditDialogState extends State<TileEditDialog> {
  late String _color;
  late int _value;

  @override
  void initState() {
    super.initState();
    _color = widget.tile?.color ?? 'RED';
    _value = widget.tile?.value ?? 1;
  }

  void _onColorChanged(String? color) {
    if (color == null) return;
    setState(() {
      _color = color;
      if (color == 'JOKER') {
        _value = 0;
      } else if (_value < 1 || _value > 13) {
        _value = 1;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final isJoker = _color == 'JOKER';
    final isNew = widget.tile == null;

    return AlertDialog(
      title: Text(isNew ? 'Add tile' : 'Edit tile'),
      content: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          DropdownButtonFormField<String>(
            initialValue: _color,
            decoration: const InputDecoration(labelText: 'Color'),
            items: kTileColors
                .map(
                  (color) => DropdownMenuItem(
                    value: color,
                    child: Text(color),
                  ),
                )
                .toList(),
            onChanged: _onColorChanged,
          ),
          const SizedBox(height: 12),
          DropdownButtonFormField<int>(
            key: ValueKey(_color),
            initialValue: isJoker ? 0 : _value,
            decoration: const InputDecoration(labelText: 'Value'),
            items: isJoker
                ? const [
                    DropdownMenuItem(value: 0, child: Text('0 (Joker)')),
                  ]
                : List.generate(
                    13,
                    (i) => DropdownMenuItem(
                      value: i + 1,
                      child: Text('${i + 1}'),
                    ),
                  ),
            onChanged: isJoker
                ? null
                : (value) {
                    if (value == null) return;
                    setState(() => _value = value);
                  },
          ),
        ],
      ),
      actions: [
        if (widget.allowDelete && !isNew)
          TextButton(
            onPressed: () {
              Navigator.of(context).pop(const TileEditResult.delete());
            },
            style: TextButton.styleFrom(
              foregroundColor: Theme.of(context).colorScheme.error,
            ),
            child: const Text('Delete'),
          ),
        TextButton(
          onPressed: () => Navigator.of(context).pop(),
          child: const Text('Cancel'),
        ),
        FilledButton(
          onPressed: () {
            Navigator.of(context).pop(
              TileEditResult.save(color: _color, value: _value),
            );
          },
          child: Text(isNew ? 'Add' : 'Save'),
        ),
      ],
    );
  }
}
