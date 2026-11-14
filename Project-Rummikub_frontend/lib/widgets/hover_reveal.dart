import 'package:flutter/material.dart';

typedef HoverRevealBuilder = Widget Function(
  BuildContext context,
  bool revealed,
  VoidCallback close,
);

/// Reveals [builder]'s controls while the pointer is over the region, while
/// keyboard focus is inside it, or after a tap (touch screens have no hover).
///
/// A tap toggles. Leaving with the mouse closes a tap-opened region, so a
/// click on desktop never leaves controls stuck open behind the pointer.
/// [builder] decides how `revealed` looks; the region itself paints nothing.
class HoverReveal extends StatefulWidget {
  const HoverReveal({
    super.key,
    required this.builder,
    this.enabled = true,
    this.onRevealChanged,
    this.cursor = SystemMouseCursors.click,
    this.opaque = true,
  });

  final HoverRevealBuilder builder;

  /// False: never reveals, whatever the pointer does.
  final bool enabled;

  /// Called whenever `revealed` flips, after the state is updated.
  final ValueChanged<bool>? onRevealChanged;

  final MouseCursor cursor;

  /// False lets regions drawn behind this one hover too, for a card that
  /// sits on a surface that should stay lit.
  final bool opaque;

  @override
  State<HoverReveal> createState() => _HoverRevealState();
}

class _HoverRevealState extends State<HoverReveal> {
  bool _hovered = false;
  bool _focused = false;
  bool _tapped = false;
  bool _lastRevealed = false;

  bool get _revealed => widget.enabled && (_hovered || _focused || _tapped);

  void _update(VoidCallback change) {
    if (!mounted) return;
    setState(change);
    final revealed = _revealed;
    if (revealed != _lastRevealed) {
      _lastRevealed = revealed;
      widget.onRevealChanged?.call(revealed);
    }
  }

  /// Folds a tap-opened region. A mouse that is still over it keeps it open:
  /// no exit event would ever bring it back.
  void _close() => _update(() => _tapped = false);

  @override
  void didUpdateWidget(HoverReveal oldWidget) {
    super.didUpdateWidget(oldWidget);
    // Losing `enabled` flips `revealed` without any pointer event.
    if (_revealed != _lastRevealed) {
      _lastRevealed = _revealed;
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (mounted) widget.onRevealChanged?.call(_lastRevealed);
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Focus(
      // Only reports whether focus is somewhere inside; the buttons are the
      // actual focus targets.
      canRequestFocus: false,
      skipTraversal: true,
      onFocusChange: (focused) => _update(() => _focused = focused),
      child: MouseRegion(
        opaque: widget.opaque,
        cursor: widget.enabled ? widget.cursor : SystemMouseCursors.forbidden,
        onEnter: (_) => _update(() => _hovered = true),
        onExit: (_) => _update(() {
          _hovered = false;
          _tapped = false;
        }),
        child: GestureDetector(
          behavior: HitTestBehavior.opaque,
          onTap:
              widget.enabled ? () => _update(() => _tapped = !_tapped) : null,
          child: widget.builder(context, _revealed, _close),
        ),
      ),
    );
  }
}
