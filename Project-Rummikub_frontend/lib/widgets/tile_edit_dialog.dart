import 'package:flutter/material.dart';

import '../models/models.dart';
import 'rummikub_tiles.dart' show kTileColors;

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
