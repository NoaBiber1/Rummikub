import 'package:flutter/material.dart';

import '../models/models.dart';

import 'tile_widget.dart';

export 'tile_edit_dialog.dart';

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
    final foreground = tile.color == 'YELLOW' ? Colors.black87 : Colors.white;

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
  });

  final String label;
  final VoidCallback? onTap;

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
          width: 48,
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

/// Whether the tile at [index] of a set with [tileCount] tiles can cut the set
/// right after itself.
///
/// The cut keeps the tile on the left, so every tile can split except the
/// last: nothing would move right. That also rules out a one-tile set.
bool canSplitAfter(int index, int tileCount) => index < tileCount - 1;

class BoardSetCard extends StatelessWidget {
  const BoardSetCard({
    super.key,
    required this.setIndex,
    required this.tileSet,
    this.selectedTileId,
    this.highlightedTileIds = const {},
    this.onTileTap,
    this.onTileEdited,
    this.onTileDeleted,
    this.onSplitAfter,
    this.onAutoSplit,
    this.onUniteWithNext,
    this.readOnly = false,
  });

  final int setIndex;
  final TileSet tileSet;
  final String? selectedTileId;
  final Set<String> highlightedTileIds;
  final ValueChanged<Tile>? onTileTap;

  /// Give each tile Edit / Delete buttons (shown on hover or selection).
  final void Function(Tile original, Tile updated)? onTileEdited;
  final ValueChanged<Tile>? onTileDeleted;

  /// Gives tiles a scissors button (see [canSplitAfter]); the value is the
  /// index of the tile that stays last in this set.
  final ValueChanged<int>? onSplitAfter;

  /// Shows an "Auto-split" button in the header.
  final VoidCallback? onAutoSplit;

  /// Shows a button under the tiles that joins this set with the one below.
  final VoidCallback? onUniteWithNext;
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
            Row(
              children: [
                Expanded(
                  child: Text(
                    'Set ${setIndex + 1}  ·  ${tileSet.type}',
                    style: Theme.of(context).textTheme.labelLarge?.copyWith(
                          color: isInvalid
                              ? Theme.of(context).colorScheme.error
                              : null,
                        ),
                  ),
                ),
                if (!readOnly && onAutoSplit != null)
                  TextButton.icon(
                    onPressed: onAutoSplit,
                    icon: const Icon(Icons.auto_fix_high, size: 16),
                    label: const Text('Auto-split'),
                  ),
              ],
            ),
            const SizedBox(height: 8),
            Wrap(
              spacing: 4,
              runSpacing: 8,
              crossAxisAlignment: WrapCrossAlignment.center,
              children: _setChildren(),
            ),
            if (!readOnly && onUniteWithNext != null)
              Align(
                child: IconButton(
                  key: const Key('set_unite'),
                  tooltip: 'Unite with set below',
                  visualDensity: VisualDensity.compact,
                  iconSize: 20,
                  icon: const Icon(Icons.call_merge),
                  onPressed: onUniteWithNext,
                ),
              ),
          ],
        ),
      ),
    );
  }

  List<Widget> _setChildren() {
    final tiles = tileSet.tiles;
    return [
      for (var i = 0; i < tiles.length; i++)
        TileWidget(
          tile: tiles[i],
          selected: tiles[i].id == selectedTileId,
          highlighted: highlightedTileIds.contains(tiles[i].id),
          onTap: onTileTap == null ? null : () => onTileTap!(tiles[i]),
          onEdit: readOnly || onTileEdited == null
              ? null
              : (updated) => onTileEdited!(tiles[i], updated),
          onDelete: readOnly || onTileDeleted == null
              ? null
              : () => onTileDeleted!(tiles[i]),
          onSplit: readOnly ||
                  onSplitAfter == null ||
                  !canSplitAfter(i, tiles.length)
              ? null
              : () => onSplitAfter!(i),
        ),
    ];
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
