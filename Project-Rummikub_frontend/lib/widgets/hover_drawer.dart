import 'dart:async';
import 'dart:ui' show lerpDouble;

import 'package:flutter/material.dart';

import 'auth_brand.dart';

/// One row of the drawer, or a divider when [isDivider].
class DrawerEntry {
  const DrawerEntry({
    required this.icon,
    required this.label,
    required this.onTap,
    this.entryKey,
  }) : isDivider = false;

  const DrawerEntry.divider()
      : icon = Icons.horizontal_rule,
        label = '',
        onTap = null,
        entryKey = null,
        isDivider = true;

  final IconData icon;
  final String label;

  /// Null disables the row.
  final VoidCallback? onTap;
  final Key? entryKey;
  final bool isDivider;
}

/// A menu button in a screen corner and the drawer behind it.
///
/// - Hovering the button (or the open drawer) slides the drawer out; leaving
///   both closes it after a short grace period, unless it is pinned.
/// - Clicking the button pins the drawer open. Clicking again unpins it; a
///   mouse still over it keeps it open until it leaves.
/// - Touch screens have no hover, so the tap is the whole interaction: pin,
///   then unpin.
///
/// Fill a [Stack] with it (`Positioned.fill`). It only receives pointer events
/// where it is drawn: the button when closed, the panel when open.
class HoverDrawer extends StatefulWidget {
  const HoverDrawer({
    super.key,
    required this.entries,
    this.panelWidth = 288,
    this.closeDelay = const Duration(milliseconds: 160),
    this.slideDuration = const Duration(milliseconds: 240),
  });

  final List<DrawerEntry> entries;
  final double panelWidth;

  /// Pointer-gap tolerance, so crossing from button to panel never flickers.
  final Duration closeDelay;
  final Duration slideDuration;

  @override
  State<HoverDrawer> createState() => _HoverDrawerState();
}

class _HoverDrawerState extends State<HoverDrawer> {
  static const _buttonSize = 52.0;
  static const _inset = 8.0;

  bool _hovering = false;
  bool _pinned = false;
  Timer? _closeTimer;

  bool get _open => _pinned || _hovering;

  void _enter() {
    _closeTimer?.cancel();
    if (!_hovering) setState(() => _hovering = true);
  }

  void _exit() {
    _closeTimer?.cancel();
    _closeTimer = Timer(widget.closeDelay, () {
      if (mounted) setState(() => _hovering = false);
    });
  }

  void _togglePin() => setState(() => _pinned = !_pinned);

  /// After choosing a row: an unpinned drawer gets out of the way.
  void _afterChoice() {
    _closeTimer?.cancel();
    if (_hovering) setState(() => _hovering = false);
  }

  @override
  void dispose() {
    _closeTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final height = constraints.maxHeight.isFinite
            ? constraints.maxHeight
            : MediaQuery.sizeOf(context).height;
        // Keep the button clear of a notch or status bar.
        final safe = MediaQuery.paddingOf(context);
        return Align(
          alignment: Alignment.topLeft,
          child: MouseRegion(
            onEnter: (_) => _enter(),
            onExit: (_) => _exit(),
            child: TweenAnimationBuilder<double>(
              tween: Tween<double>(end: _open ? 1 : 0),
              duration: widget.slideDuration,
              curve: Curves.easeOutCubic,
              builder: (context, t, _) {
                const collapsed = _buttonSize + _inset * 2;
                return SizedBox(
                  width: lerpDouble(
                    collapsed + safe.left,
                    widget.panelWidth + safe.left,
                    t,
                  ),
                  height: lerpDouble(collapsed + safe.top, height, t),
                  child: Stack(
                    clipBehavior: Clip.hardEdge,
                    children: [
                      if (t > 0)
                        Positioned(
                          left: -(1 - t) * (widget.panelWidth + safe.left),
                          top: 0,
                          bottom: 0,
                          width: widget.panelWidth + safe.left,
                          child: Opacity(
                            opacity: Curves.easeIn.transform(t),
                            child: _Panel(
                              key: const ValueKey('drawer-panel'),
                              entries: widget.entries,
                              pinned: _pinned,
                              onChosen: _afterChoice,
                            ),
                          ),
                        ),
                      Positioned(
                        left: _inset + safe.left,
                        top: _inset + safe.top,
                        child: _MenuButton(
                          key: const ValueKey('drawer-toggle'),
                          pinned: _pinned,
                          onPressed: _togglePin,
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
          ),
        );
      },
    );
  }
}

class _MenuButton extends StatefulWidget {
  const _MenuButton({super.key, required this.pinned, required this.onPressed});

  final bool pinned;
  final VoidCallback onPressed;

  @override
  State<_MenuButton> createState() => _MenuButtonState();
}

class _MenuButtonState extends State<_MenuButton> {
  bool _hovered = false;
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    final lit = _hovered || _focused;
    return Semantics(
      button: true,
      toggled: widget.pinned,
      label: widget.pinned ? 'Menu, pinned open' : 'Menu',
      hint: widget.pinned ? 'Unpin the menu' : 'Pin the menu open',
      excludeSemantics: true,
      onTap: widget.onPressed,
      child: AnimatedScale(
        scale: lit ? 1.08 : 1,
        duration: const Duration(milliseconds: 160),
        curve: Curves.easeOut,
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            AnimatedContainer(
              duration: const Duration(milliseconds: 160),
              width: _HoverDrawerState._buttonSize,
              height: _HoverDrawerState._buttonSize,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: lit ? const Color(0xFFE02A31) : AuthColors.red,
                border: Border.all(
                  color: widget.pinned ? Colors.white : const Color(0x66FFFFFF),
                  width: widget.pinned ? 3 : 2,
                ),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x66000000),
                    blurRadius: lit ? 14 : 8,
                    offset: const Offset(0, 3),
                  ),
                ],
              ),
              child: Material(
                type: MaterialType.transparency,
                shape: const CircleBorder(),
                child: InkWell(
                  customBorder: const CircleBorder(),
                  onTap: widget.onPressed,
                  onHover: (value) => setState(() => _hovered = value),
                  onFocusChange: (value) => setState(() => _focused = value),
                  child: const Icon(Icons.menu_rounded, color: Colors.white),
                ),
              ),
            ),
            if (widget.pinned)
              const Positioned(
                right: -2,
                bottom: -2,
                child: CircleAvatar(
                  radius: 10,
                  backgroundColor: Colors.white,
                  child: Icon(
                    Icons.push_pin,
                    size: 12,
                    color: AuthColors.red,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }
}

class _Panel extends StatelessWidget {
  const _Panel({
    super.key,
    required this.entries,
    required this.pinned,
    required this.onChosen,
  });

  final List<DrawerEntry> entries;
  final bool pinned;
  final VoidCallback onChosen;

  @override
  Widget build(BuildContext context) {
    return Semantics(
      container: true,
      label: 'Menu',
      child: DecoratedBox(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [AuthColors.royal, AuthColors.navy],
          ),
          borderRadius: BorderRadius.horizontal(right: Radius.circular(28)),
          border: Border(
            right: BorderSide(color: AuthColors.red, width: 4),
          ),
          boxShadow: [
            BoxShadow(
              color: Color(0x99000000),
              blurRadius: 24,
              offset: Offset(6, 0),
            ),
          ],
        ),
        child: SafeArea(
          right: false,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // The menu button sits over the top-left corner of this header.
              SizedBox(
                height: _HoverDrawerState._buttonSize +
                    _HoverDrawerState._inset * 2,
                child: Padding(
                  padding: const EdgeInsets.only(left: 72, right: 16),
                  child: const Align(
                    alignment: Alignment.centerLeft,
                    child: RummikubHelperWordmark(compact: true),
                  ),
                ),
              ),
              const Divider(color: Color(0x33FFFFFF), height: 1),
              Expanded(
                child: ListView(
                  padding: const EdgeInsets.fromLTRB(12, 12, 12, 8),
                  children: [
                    for (final entry in entries)
                      if (entry.isDivider)
                        const Padding(
                          padding: EdgeInsets.symmetric(vertical: 8),
                          child: Divider(color: Color(0x33FFFFFF), height: 1),
                        )
                      else
                        _DrawerTile(
                          key: entry.entryKey,
                          entry: entry,
                          onChosen: onChosen,
                        ),
                  ],
                ),
              ),
              Padding(
                padding: const EdgeInsets.fromLTRB(20, 4, 20, 16),
                child: Text(
                  pinned
                      ? 'Pinned open. Click the menu button to unpin.'
                      : 'Click the menu button to keep this open.',
                  style: const TextStyle(color: Colors.white60, fontSize: 12),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class _DrawerTile extends StatefulWidget {
  const _DrawerTile({super.key, required this.entry, required this.onChosen});

  final DrawerEntry entry;
  final VoidCallback onChosen;

  @override
  State<_DrawerTile> createState() => _DrawerTileState();
}

class _DrawerTileState extends State<_DrawerTile> {
  bool _hovered = false;
  bool _focused = false;

  @override
  Widget build(BuildContext context) {
    final entry = widget.entry;
    final enabled = entry.onTap != null;
    final lit = enabled && (_hovered || _focused);
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Material(
        type: MaterialType.transparency,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: enabled
              ? () {
                  widget.onChosen();
                  entry.onTap!();
                }
              : null,
          onHover: (value) => setState(() => _hovered = value),
          onFocusChange: (value) => setState(() => _focused = value),
          child: AnimatedContainer(
            duration: const Duration(milliseconds: 150),
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 14),
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              color: lit ? const Color(0x2EFFFFFF) : const Color(0x00FFFFFF),
            ),
            child: Row(
              children: [
                AnimatedContainer(
                  duration: const Duration(milliseconds: 150),
                  width: 4,
                  height: lit ? 24 : 0,
                  margin: const EdgeInsets.only(right: 10),
                  decoration: BoxDecoration(
                    color: AuthColors.red,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                AnimatedScale(
                  scale: lit ? 1.12 : 1,
                  duration: const Duration(milliseconds: 150),
                  child: Icon(
                    entry.icon,
                    color: !enabled
                        ? Colors.white30
                        : lit
                            ? Colors.white
                            : AuthColors.ivory,
                  ),
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: AnimatedDefaultTextStyle(
                    duration: const Duration(milliseconds: 150),
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w700,
                      color: !enabled
                          ? Colors.white30
                          : lit
                              ? Colors.white
                              : AuthColors.ivory,
                    ),
                    child: Text(entry.label),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}
