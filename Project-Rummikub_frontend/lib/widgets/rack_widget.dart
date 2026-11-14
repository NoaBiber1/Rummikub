import 'package:flutter/material.dart';

import '../models/models.dart';
import '../models/rummikub_tiles.dart';
import 'rummikub_tiles.dart';
import 'tile_picker_popover.dart';
import 'tile_widget.dart';

const _frameColor = Color(0xFF795548);
const _shelfColor = Color(0xFF4E342E);
const _ledgeColor = Color(0xFFBCAAA4);

/// The player's rack: header with a `+` button that adds tiles, then the tiles
/// on a two-shelf rack like the real one. Each tile sits on the shelf it was on
/// in the rack photo ([Tile.rackRow]), in photo order.
class RackWidget extends StatelessWidget {
  const RackWidget({
    super.key,
    required this.tiles,
    required this.onTilesAdded,
    this.selectedTileId,
    this.onTileTap,
    this.onTileEdited,
    this.onTileDeleted,
    this.onClear,
    this.extra,
  });

  final List<Tile> tiles;
  final ValueChanged<List<Tile>> onTilesAdded;
  final String? selectedTileId;
  final ValueChanged<Tile>? onTileTap;
  final void Function(Tile original, Tile updated)? onTileEdited;
  final ValueChanged<Tile>? onTileDeleted;

  /// Shows a clear button that empties the rack (the rack itself stays).
  final VoidCallback? onClear;

  /// Shown after the last tile of the shelf the next tile lands on, e.g. a
  /// slot that receives a selected tile.
  final Widget? extra;

  @override
  Widget build(BuildContext context) {
    final shelves = splitRackRows(tiles);
    final extraShelf = shelves[0].length < kRackRowCapacity ? 0 : 1;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Expanded(
              child: SectionHeader(
                title: 'Rack',
                subtitle: '${tiles.length} tiles',
              ),
            ),
            if (onClear != null)
              IconButton(
                key: const Key('rack_clear'),
                tooltip: 'Clear rack',
                icon: const Icon(Icons.delete_sweep),
                onPressed: tiles.isEmpty ? null : onClear,
              ),
            TilePickerButton(
              onTilesSelected: onTilesAdded,
              tooltip: 'Add tiles to rack',
            ),
          ],
        ),
        const SizedBox(height: 8),
        // Shelves are one tile row each, like the real thing; a long rack
        // scrolls sideways, both shelves together, instead of wrapping.
        LayoutBuilder(
          builder: (context, constraints) => SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: ConstrainedBox(
              constraints: BoxConstraints(minWidth: constraints.maxWidth),
              child: IntrinsicWidth(
                child: DecoratedBox(
                  key: const Key('rack_tiles'),
                  decoration: BoxDecoration(
                    color: _frameColor,
                    borderRadius: BorderRadius.circular(14),
                    boxShadow: const [
                      BoxShadow(
                        color: Colors.black26,
                        blurRadius: 4,
                        offset: Offset(0, 2),
                      ),
                    ],
                  ),
                  child: Padding(
                    padding: const EdgeInsets.all(10),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        for (var row = 0; row < shelves.length; row++) ...[
                          if (row > 0) const SizedBox(height: 10),
                          _shelf(
                            row,
                            shelves[row],
                            showExtra: row == extraShelf,
                          ),
                        ],
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }

  Widget _shelf(int row, List<Tile> shelfTiles, {required bool showExtra}) {
    return Container(
      key: Key('rack_row_$row'),
      height: 82,
      padding: const EdgeInsets.symmetric(horizontal: 6),
      decoration: BoxDecoration(
        color: _shelfColor,
        borderRadius: BorderRadius.circular(6),
        border: const Border(bottom: BorderSide(color: _ledgeColor, width: 6)),
      ),
      child: Row(
        children: [
          for (final tile in shelfTiles)
            Padding(
              padding: const EdgeInsets.only(right: 4),
              child: TileWidget(
                tile: tile,
                selected: tile.id == selectedTileId,
                onTap: onTileTap == null ? null : () => onTileTap!(tile),
                onEdit: onTileEdited == null
                    ? null
                    : (updated) => onTileEdited!(tile, updated),
                onDelete:
                    onTileDeleted == null ? null : () => onTileDeleted!(tile),
              ),
            ),
          if (showExtra && extra != null) extra!,
        ],
      ),
    );
  }
}
