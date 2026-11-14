import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../models/models.dart';
import 'rummikub_tiles.dart';

const _pickerColors = ['RED', 'BLUE', 'BLACK', 'YELLOW'];
const _maxQuantity = 8;

/// Menu for adding tiles: pick a color (hover or tap) to expand its numbers
/// 1-13, or a joker. Tapping a tile toggles it into a pending selection that
/// survives switching colors; Add commits the whole selection at once.
class TilePickerPopover extends StatefulWidget {
  const TilePickerPopover({
    super.key,
    required this.onTilesSelected,
    this.onCancel,
  });

  /// Called once by Add with [quantity] fresh copies of every selected tile,
  /// in the order they were selected.
  final ValueChanged<List<Tile>> onTilesSelected;

  /// Shows a Cancel button when set.
  final VoidCallback? onCancel;

  @override
  State<TilePickerPopover> createState() => _TilePickerPopoverState();
}

class _TilePickerPopoverState extends State<TilePickerPopover> {
  final _quantityController = TextEditingController(text: '1');
  final _selected = <(String color, int value)>[];
  String? _expanded;

  @override
  void dispose() {
    _quantityController.dispose();
    super.dispose();
  }

  /// Copies added per selected tile.
  int get _quantity {
    final parsed = int.tryParse(_quantityController.text) ?? 1;
    return parsed.clamp(1, _maxQuantity);
  }

  void _setQuantity(int value) {
    final clamped = value.clamp(1, _maxQuantity);
    _quantityController.text = '$clamped';
    setState(() {});
  }

  void _toggle(String color, int value) {
    setState(() {
      if (!_selected.remove((color, value))) _selected.add((color, value));
    });
  }

  void _add() {
    widget.onTilesSelected([
      for (final (color, value) in _selected)
        for (var i = 0; i < _quantity; i++) Tile(color: color, value: value),
    ]);
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _quantityRow(),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              for (final color in _pickerColors)
                _categoryChip(
                    color, color[0] + color.substring(1).toLowerCase()),
              _categoryChip('JOKER', 'Joker'),
            ],
          ),
          const SizedBox(height: 12),
          if (_expanded == 'JOKER') _jokerOptions(),
          if (_expanded != null && _expanded != 'JOKER')
            _numberGrid(_expanded!),
          const SizedBox(height: 12),
          _actionRow(),
        ],
      ),
    );
  }

  Widget _quantityRow() {
    return Row(
      mainAxisSize: MainAxisSize.min,
      children: [
        const Text('Quantity'),
        const SizedBox(width: 8),
        IconButton(
          key: const Key('quantity_minus'),
          icon: const Icon(Icons.remove_circle_outline),
          onPressed: () => _setQuantity(_quantity - 1),
        ),
        SizedBox(
          width: 48,
          child: TextField(
            key: const Key('quantity_field'),
            controller: _quantityController,
            textAlign: TextAlign.center,
            keyboardType: TextInputType.number,
            inputFormatters: [FilteringTextInputFormatter.digitsOnly],
            decoration: const InputDecoration(
              isDense: true,
              border: OutlineInputBorder(),
            ),
            onChanged: (_) => setState(() {}),
          ),
        ),
        IconButton(
          key: const Key('quantity_plus'),
          icon: const Icon(Icons.add_circle_outline),
          onPressed: () => _setQuantity(_quantity + 1),
        ),
      ],
    );
  }

  Widget _actionRow() {
    final count = _selected.length;
    return Row(
      children: [
        Expanded(
          child: Text(
            '$count selected',
            key: const Key('picker_count'),
            overflow: TextOverflow.ellipsis,
          ),
        ),
        TextButton(
          key: const Key('picker_clear'),
          onPressed: count == 0 ? null : () => setState(_selected.clear),
          child: const Text('Clear'),
        ),
        if (widget.onCancel != null)
          TextButton(
            key: const Key('picker_cancel'),
            onPressed: widget.onCancel,
            child: const Text('Cancel'),
          ),
        const SizedBox(width: 4),
        FilledButton(
          key: const Key('picker_add'),
          onPressed: count == 0 ? null : _add,
          child: const Text('Add'),
        ),
      ],
    );
  }

  Widget _categoryChip(String color, String label) {
    final selected = _expanded == color;
    return MouseRegion(
      onEnter: (_) => setState(() => _expanded = color),
      child: ChoiceChip(
        key: Key('color_$color'),
        label: Text(
          label,
          style: TextStyle(
            color: color == 'YELLOW' ? Colors.black87 : Colors.white,
          ),
        ),
        selected: selected,
        showCheckmark: false,
        backgroundColor: colorForTile(color),
        selectedColor: colorForTile(color),
        side: BorderSide(
          color: selected ? Colors.white : Colors.transparent,
          width: 2,
        ),
        onSelected: (_) => setState(() => _expanded = color),
      ),
    );
  }

  Widget _numberGrid(String color) {
    return Wrap(
      spacing: 6,
      runSpacing: 6,
      children: [
        for (var value = 1; value <= 13; value++)
          _tileButton(
            key: Key('tile_${color}_$value'),
            color: color,
            label: '$value',
            selected: _selected.contains((color, value)),
            onTap: () => _toggle(color, value),
          ),
      ],
    );
  }

  // The tile model has a single joker; the two buttons only differ in look,
  // so selecting either marks both.
  Widget _jokerOptions() {
    final selected = _selected.contains(('JOKER', 0));
    return Wrap(
      spacing: 6,
      children: [
        _tileButton(
          key: const Key('joker_black'),
          color: 'BLACK',
          label: 'J',
          selected: selected,
          onTap: () => _toggle('JOKER', 0),
        ),
        _tileButton(
          key: const Key('joker_red'),
          color: 'RED',
          label: 'J',
          selected: selected,
          onTap: () => _toggle('JOKER', 0),
        ),
      ],
    );
  }

  Widget _tileButton({
    required Key key,
    required String color,
    required String label,
    required bool selected,
    required VoidCallback onTap,
  }) {
    return SizedBox(
      width: 40,
      height: 48,
      child: Stack(
        children: [
          Positioned.fill(
            child: Material(
              key: key,
              color: colorForTile(color),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(6),
                side: BorderSide(
                  color: selected
                      ? Theme.of(context).colorScheme.primary
                      : Colors.transparent,
                  width: 3,
                ),
              ),
              child: InkWell(
                customBorder: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(6),
                ),
                onTap: onTap,
                child: Center(
                  child: Text(
                    label,
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: color == 'YELLOW' ? Colors.black87 : Colors.white,
                    ),
                  ),
                ),
              ),
            ),
          ),
          if (selected)
            const Positioned(
              top: 0,
              right: 0,
              child: IgnorePointer(
                child: Icon(
                  Icons.check_circle,
                  key: Key('picker_selected_badge'),
                  size: 16,
                  color: Colors.white,
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// The `+` button: opens a [TilePickerPopover] and closes it on Add or Cancel.
class TilePickerButton extends StatelessWidget {
  const TilePickerButton({
    super.key,
    required this.onTilesSelected,
    this.tooltip = 'Add tiles',
  });

  final ValueChanged<List<Tile>> onTilesSelected;
  final String tooltip;

  Future<void> _open(BuildContext context) {
    return showDialog<void>(
      context: context,
      builder: (dialogContext) => Dialog(
        child: ConstrainedBox(
          constraints: const BoxConstraints(maxWidth: 420),
          child: SingleChildScrollView(
            child: TilePickerPopover(
              onCancel: () => Navigator.of(dialogContext).pop(),
              onTilesSelected: (tiles) {
                Navigator.of(dialogContext).pop();
                onTilesSelected(tiles);
              },
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return IconButton(
      key: const Key('tile_picker_button'),
      tooltip: tooltip,
      icon: const Icon(Icons.add_circle),
      onPressed: () => _open(context),
    );
  }
}
