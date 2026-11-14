import 'package:flutter/material.dart';

import '../models/models.dart';
import 'rummikub_tiles.dart';

/// A [TileView] that shows Edit, Delete and Split buttons while hovered or
/// selected.
///
/// Edit opens [TileEditDialog] and reports the changed tile through [onEdit];
/// Delete calls [onDelete]; Split calls [onSplit] (only tiles inside a set
/// can cut it, so the rest leave it null). With no callback it is a plain tile.
class TileWidget extends StatefulWidget {
  const TileWidget({
    super.key,
    required this.tile,
    this.selected = false,
    this.highlighted = false,
    this.onTap,
    this.onEdit,
    this.onDelete,
    this.onSplit,
  });

  final Tile tile;
  final bool selected;
  final bool highlighted;
  final VoidCallback? onTap;
  final ValueChanged<Tile>? onEdit;
  final VoidCallback? onDelete;

  /// Cuts the set right after this tile.
  final VoidCallback? onSplit;

  @override
  State<TileWidget> createState() => _TileWidgetState();
}

class _TileWidgetState extends State<TileWidget> {
  bool _hovered = false;

  bool get _hasActions =>
      widget.onEdit != null ||
      widget.onDelete != null ||
      widget.onSplit != null;

  Future<void> _edit() async {
    final result = await showDialog<TileEditResult>(
      context: context,
      builder: (_) => TileEditDialog(tile: widget.tile),
    );
    if (result == null) return;
    if (result.delete) {
      widget.onDelete?.call();
    } else {
      widget.onEdit?.call(
        widget.tile.copyWith(color: result.color, value: result.value),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    final view = TileView(
      tile: widget.tile,
      selected: widget.selected,
      highlighted: widget.highlighted,
      onTap: widget.onTap,
    );
    if (!_hasActions) return view;

    return MouseRegion(
      onEnter: (_) => setState(() => _hovered = true),
      onExit: (_) => setState(() => _hovered = false),
      child: Stack(
        children: [
          view,
          if (_hovered || widget.selected)
            Positioned(
              top: 2,
              left: 2,
              right: 2,
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  if (widget.onEdit != null)
                    _ActionButton(
                      key: const Key('tile_edit'),
                      tooltip: 'Edit',
                      icon: Icons.edit,
                      onPressed: _edit,
                    ),
                  if (widget.onDelete != null)
                    _ActionButton(
                      key: const Key('tile_delete'),
                      tooltip: 'Delete',
                      icon: Icons.delete,
                      onPressed: widget.onDelete!,
                    ),
                ],
              ),
            ),
          // Bottom right, on the side the cut falls: a third button does not
          // fit in the top row of a 48 px tile.
          if ((_hovered || widget.selected) && widget.onSplit != null)
            Positioned(
              bottom: 2,
              right: 2,
              child: _ActionButton(
                key: const Key('tile_split'),
                tooltip: 'Split set here',
                icon: Icons.content_cut,
                onPressed: widget.onSplit!,
              ),
            ),
        ],
      ),
    );
  }
}

class _ActionButton extends StatelessWidget {
  const _ActionButton({
    super.key,
    required this.tooltip,
    required this.icon,
    required this.onPressed,
  });

  final String tooltip;
  final IconData icon;
  final VoidCallback onPressed;

  @override
  Widget build(BuildContext context) {
    return Tooltip(
      message: tooltip,
      child: Material(
        color: Colors.black54,
        shape: const CircleBorder(),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onPressed,
          child: Padding(
            padding: const EdgeInsets.all(3),
            child: Icon(icon, size: 14, color: Colors.white),
          ),
        ),
      ),
    );
  }
}
