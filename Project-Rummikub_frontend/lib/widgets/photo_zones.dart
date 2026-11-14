import 'dart:math' as math;
import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'auth_brand.dart';
import 'brand_buttons.dart';
import 'hover_reveal.dart';

/// An accepted photo, as the zones show it.
class ZonePhoto {
  const ZonePhoto({
    required this.bytes,
    required this.width,
    required this.height,
    required this.label,
  });

  final Uint8List bytes;
  final int width;
  final int height;

  /// File name and size: the preview's tooltip and screen-reader label.
  final String label;
}

/// The surface of the board or the rack: it lights up under the pointer and,
/// while empty, offers Camera and Gallery.
///
/// Controls stay in the tree while hidden (faded, ignoring pointers), so a
/// keyboard Tab reaches them, and focus alone reveals them. A tap reveals them
/// on touch screens.
class PhotoZone extends StatelessWidget {
  const PhotoZone({
    super.key,
    required this.slot,
    required this.surface,
    required this.hint,
    required this.hasPhoto,
    required this.active,
    required this.canCapture,
    required this.cameraSupported,
    required this.onCamera,
    required this.onGallery,
    required this.semanticsLabel,
    this.note,
    this.hitClipper,
    this.controlsAlignment = Alignment.center,
    this.compactControls = false,
  });

  /// `table` or `rack`; prefixes the widget keys tests look up.
  final String slot;

  /// Paints the surface for a glow of 0..1.
  final Widget Function(double glow) surface;

  /// Shown while the zone is empty and idle.
  final Widget hint;
  final bool hasPhoto;

  /// The step the user should do next: a steady soft glow.
  final bool active;

  /// False while busy or before the table photo exists.
  final bool canCapture;
  final bool cameraSupported;
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  final String semanticsLabel;

  /// Why the buttons are disabled, shown under them.
  final String? note;

  /// Limits hover to a shape (the board's trapezoid).
  final CustomClipper<Path>? hitClipper;
  final Alignment controlsAlignment;
  final bool compactControls;

  @override
  Widget build(BuildContext context) {
    final zone = HoverReveal(
      key: ValueKey('$slot-zone'),
      cursor: SystemMouseCursors.basic,
      builder: (context, revealed, close) {
        final glow = revealed ? 1.0 : (active && canCapture ? 0.35 : 0.0);
        final showControls = revealed && !hasPhoto;
        return Stack(
          fit: StackFit.expand,
          children: [
            TweenAnimationBuilder<double>(
              tween: Tween<double>(end: glow),
              duration: const Duration(milliseconds: 180),
              builder: (context, value, _) => surface(value),
            ),
            if (!hasPhoto)
              IgnorePointer(
                child: AnimatedOpacity(
                  opacity: revealed ? 0 : 1,
                  duration: const Duration(milliseconds: 160),
                  child: hint,
                ),
              ),
            if (!hasPhoto)
              Align(
                alignment: controlsAlignment,
                child: _SourceButtons(
                  slot: slot,
                  visible: showControls,
                  compact: compactControls,
                  cameraEnabled: canCapture && cameraSupported,
                  galleryEnabled: canCapture,
                  onCamera: () {
                    close();
                    onCamera();
                  },
                  onGallery: () {
                    close();
                    onGallery();
                  },
                  note: note ??
                      (cameraSupported
                          ? null
                          : 'No camera on this device: use Gallery.'),
                ),
              ),
          ],
        );
      },
    );
    return Semantics(
      container: true,
      label: semanticsLabel,
      child: hitClipper == null
          ? zone
          : ClipPath(clipper: hitClipper, child: zone),
    );
  }
}

class _SourceButtons extends StatelessWidget {
  const _SourceButtons({
    required this.slot,
    required this.visible,
    required this.compact,
    required this.cameraEnabled,
    required this.galleryEnabled,
    required this.onCamera,
    required this.onGallery,
    required this.note,
  });

  final String slot;
  final bool visible;
  final bool compact;
  final bool cameraEnabled;
  final bool galleryEnabled;
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  final String? note;

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      ignoring: !visible,
      child: AnimatedOpacity(
        opacity: visible ? 1 : 0,
        duration: const Duration(milliseconds: 180),
        child: AnimatedScale(
          scale: visible ? 1 : 0.9,
          duration: const Duration(milliseconds: 180),
          curve: Curves.easeOutBack,
          // Scales down on a narrow phone instead of overflowing.
          child: FittedBox(
            fit: BoxFit.scaleDown,
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    FilledButton.icon(
                      key: ValueKey('$slot-camera'),
                      onPressed: cameraEnabled ? onCamera : null,
                      style: brandButtonStyle(compact: compact),
                      icon: const Icon(Icons.photo_camera_rounded),
                      label: const Text('Camera'),
                    ),
                    SizedBox(width: compact ? 8 : 12),
                    FilledButton.icon(
                      key: ValueKey('$slot-gallery'),
                      onPressed: galleryEnabled ? onGallery : null,
                      style: brandLightButtonStyle(compact: compact),
                      icon: const Icon(Icons.photo_library_rounded),
                      label: const Text('Gallery'),
                    ),
                  ],
                ),
                if (note != null)
                  Padding(
                    padding: const EdgeInsets.only(top: 6),
                    child: Text(
                      note!,
                      textAlign: TextAlign.center,
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.w600,
                        shadows: [
                          Shadow(color: Color(0xCC000000), blurRadius: 4)
                        ],
                      ),
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

/// A small photo card resting on the zone. Under the pointer it dims and
/// offers Replace (which asks Camera or Gallery) and Delete.
class PhotoPreviewCard extends StatefulWidget {
  const PhotoPreviewCard({
    super.key,
    required this.slot,
    required this.photo,
    required this.size,
    required this.cameraSupported,
    required this.canReplace,
    required this.canRemove,
    required this.onCamera,
    required this.onGallery,
    required this.onRemove,
    this.tilt = 0,
  });

  final String slot;
  final ZonePhoto photo;
  final Size size;
  final bool cameraSupported;

  /// False while the screen is busy, or (rack) while the table photo it
  /// depends on is missing.
  final bool canReplace;

  /// False while the screen is busy.
  final bool canRemove;
  final VoidCallback onCamera;
  final VoidCallback onGallery;
  final VoidCallback onRemove;

  /// Radians; a little tilt makes it look set down, not placed.
  final double tilt;

  @override
  State<PhotoPreviewCard> createState() => _PhotoPreviewCardState();
}

class _PhotoPreviewCardState extends State<PhotoPreviewCard> {
  /// After Replace: the overlay asks for a source.
  bool _choosing = false;

  /// Decode at display size: a 1920 px photo for a 150 px card would
  /// otherwise sit in memory at full size.
  int? _decodeWidth(BuildContext context) {
    final photo = widget.photo;
    if (photo.width <= 0 || photo.height <= 0) return null;
    final scale = math.max(
      widget.size.width / photo.width,
      widget.size.height / photo.height,
    );
    final target =
        (photo.width * scale * MediaQuery.devicePixelRatioOf(context)).ceil();
    return target < photo.width ? target : null;
  }

  void _replace(VoidCallback close) {
    if (widget.cameraSupported) {
      setState(() => _choosing = true);
    } else {
      close();
      widget.onGallery();
    }
  }

  void _choose(VoidCallback close, VoidCallback pick) {
    setState(() => _choosing = false);
    close();
    pick();
  }

  @override
  Widget build(BuildContext context) {
    final slot = widget.slot;
    final canReplace = widget.canReplace;
    final canRemove = widget.canRemove;
    return HoverReveal(
      key: ValueKey('$slot-preview-card'),
      opaque: false,
      cursor: SystemMouseCursors.basic,
      onRevealChanged: (revealed) {
        if (!revealed && _choosing) setState(() => _choosing = false);
      },
      builder: (context, revealed, close) {
        return Transform.rotate(
          angle: widget.tilt,
          child: SizedBox.fromSize(
            size: widget.size,
            child: DecoratedBox(
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(16),
                boxShadow: [
                  BoxShadow(
                    color: const Color(0x80000000),
                    blurRadius: revealed ? 18 : 10,
                    offset: const Offset(0, 5),
                  ),
                ],
              ),
              child: Padding(
                padding: const EdgeInsets.all(4),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      Tooltip(
                        message: widget.photo.label,
                        child: Image.memory(
                          widget.photo.bytes,
                          key: ValueKey('$slot-preview'),
                          fit: BoxFit.cover,
                          cacheWidth: _decodeWidth(context),
                          gaplessPlayback: true,
                          semanticLabel: widget.photo.label,
                          errorBuilder: (context, error, stackTrace) =>
                              const ColoredBox(
                            color: AuthColors.navy,
                            child: Center(
                              child: Text(
                                'Preview unavailable',
                                textAlign: TextAlign.center,
                              ),
                            ),
                          ),
                        ),
                      ),
                      IgnorePointer(
                        ignoring: !revealed,
                        child: AnimatedOpacity(
                          opacity: revealed ? 1 : 0,
                          duration: const Duration(milliseconds: 160),
                          child: ColoredBox(
                            color: const Color(0xC0071A45),
                            child: Center(
                              child: _choosing
                                  ? _Stacked(
                                      children: [
                                        FilledButton.icon(
                                          key: ValueKey('$slot-camera'),
                                          onPressed: canReplace
                                              ? () => _choose(
                                                    close,
                                                    widget.onCamera,
                                                  )
                                              : null,
                                          style:
                                              brandButtonStyle(compact: true),
                                          icon: const Icon(
                                            Icons.photo_camera_rounded,
                                            size: 18,
                                          ),
                                          label: const Text('Camera'),
                                        ),
                                        FilledButton.icon(
                                          key: ValueKey('$slot-gallery'),
                                          onPressed: canReplace
                                              ? () => _choose(
                                                    close,
                                                    widget.onGallery,
                                                  )
                                              : null,
                                          style: brandLightButtonStyle(
                                            compact: true,
                                          ),
                                          icon: const Icon(
                                            Icons.photo_library_rounded,
                                            size: 18,
                                          ),
                                          label: const Text('Gallery'),
                                        ),
                                      ],
                                    )
                                  : _Stacked(
                                      children: [
                                        FilledButton.icon(
                                          key: ValueKey('$slot-replace'),
                                          onPressed: canReplace
                                              ? () => _replace(close)
                                              : null,
                                          style:
                                              brandButtonStyle(compact: true),
                                          icon: const Icon(
                                            Icons.swap_horiz_rounded,
                                            size: 18,
                                          ),
                                          label: const Text('Replace'),
                                        ),
                                        FilledButton.icon(
                                          key: ValueKey('$slot-remove'),
                                          onPressed: canRemove
                                              ? () {
                                                  close();
                                                  widget.onRemove();
                                                }
                                              : null,
                                          style: brandLightButtonStyle(
                                            compact: true,
                                          ),
                                          icon: const Icon(
                                            Icons.delete_outline_rounded,
                                            size: 18,
                                          ),
                                          label: const Text('Delete'),
                                        ),
                                      ],
                                    ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}

/// Two compact buttons, one above the other, shrinking to fit a small card.
class _Stacked extends StatelessWidget {
  const _Stacked({required this.children});

  final List<Widget> children;

  @override
  Widget build(BuildContext context) {
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          children.first,
          const SizedBox(height: 8),
          children.last,
        ],
      ),
    );
  }
}
