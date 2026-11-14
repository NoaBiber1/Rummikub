import 'dart:math' as math;

import 'package:flutter/material.dart';

/// Where everything sits in the first-person tabletop view, from one [Size].
///
/// Top to bottom: header, the far rack, the board seen in perspective (narrow
/// far edge, wide near edge) with a rack along each slanted side, the player's
/// rack in the foreground, the Recognize button. Pure math, so it is tested
/// without pumping widgets.
class SceneLayout {
  const SceneLayout._({
    required this.size,
    required this.compact,
    required this.header,
    required this.topRack,
    required this.leftRack,
    required this.rightRack,
    required this.sideAngle,
    required this.board,
    required this.boardInset,
    required this.playerRack,
    required this.recognize,
    required this.startOver,
  });

  /// Below this height the scene scrolls instead of squeezing.
  static const minHeight = 620.0;

  /// Wider windows center the scene at this width.
  static const maxWidth = 1100.0;

  factory SceneLayout.of(Size size) {
    final w = size.width;
    final h = math.max(size.height, minHeight);
    final compact = w < 520;
    final pad = compact ? 10.0 : 20.0;
    const headerH = 104.0;

    // Bottom up: Start over, Recognize, the player's rack.
    final startOver = Rect.fromLTWH((w - 200) / 2, h - 36, 200, 32);
    final buttonW = math.min(w - 2 * pad, 360.0);
    final recognize = Rect.fromLTWH((w - buttonW) / 2, h - 94, buttonW, 56);
    final playerW = math.min(w - 2 * pad, 640.0);
    final playerH = (playerW / 4.4).clamp(70.0, 150.0);
    final playerRack = Rect.fromLTWH(
      (w - playerW) / 2,
      recognize.top - 16 - playerH,
      playerW,
      playerH,
    );

    // Top down: the far rack is small, being far away.
    final topW = (w * 0.36).clamp(150.0, 420.0);
    final topH = (topW / 6.5).clamp(28.0, 60.0);
    final topRack = Rect.fromLTWH((w - topW) / 2, headerH, topW, topH);

    final sideT = (w * 0.07).clamp(30.0, 64.0);
    const gap = 8.0;
    final boardTop = topRack.bottom + 12;
    final boardBottom = playerRack.top - 14;
    final boardH = math.max(boardBottom - boardTop, 120.0);
    final boardW = w - 2 * (pad + sideT + gap);
    final inset = math.min(boardW * 0.10, boardH * 0.16);
    final board = Rect.fromLTWH(pad + sideT + gap, boardTop, boardW, boardH);

    // Side racks lie along the slanted edges: a vertical rack tilted by the
    // edge's angle, pushed out along the edge's outward normal.
    final edge = math.sqrt(inset * inset + boardH * boardH);
    final angle = math.atan2(inset, boardH);
    final mid = Offset(board.left + inset / 2, boardTop + boardH / 2);
    final outward = Offset(-boardH, -inset) / edge;
    final center = mid + outward * (sideT / 2 + gap * 0.75);
    final left = Rect.fromCenter(
      center: center,
      width: sideT,
      height: edge * 0.86,
    );
    final right = Rect.fromCenter(
      center: Offset(w - center.dx, center.dy),
      width: sideT,
      height: edge * 0.86,
    );

    return SceneLayout._(
      size: Size(w, h),
      compact: compact,
      header: Rect.fromLTWH(0, 0, w, headerH),
      topRack: topRack,
      leftRack: left,
      rightRack: right,
      sideAngle: angle,
      board: board,
      boardInset: inset,
      playerRack: playerRack,
      recognize: recognize,
      startOver: startOver,
    );
  }

  final Size size;
  final bool compact;
  final Rect header;
  final Rect topRack;

  /// Unrotated boxes; rotate the left one by [sideAngle], the right one by
  /// minus that, around their centers.
  final Rect leftRack;
  final Rect rightRack;
  final double sideAngle;

  /// Bounding box of the board, including its front edge.
  final Rect board;

  /// How far each far corner is pulled in from the near corner.
  final double boardInset;
  final Rect playerRack;
  final Rect recognize;
  final Rect startOver;
}

abstract final class _Wood {
  static const dark = Color(0xFF8A4F1F);
  static const mid = Color(0xFFB06B2C);
  static const light = Color(0xFFD99A55);
  static const highlight = Color(0xFFF2C98E);
  static const grain = Color(0xFF7A4119);
  static const outline = Color(0xFF3F2310);
}

/// A cartoon wooden rack seen from the front: a tall back panel above a
/// slotted ledge, thick outline, soft shadow. Draws with the ledge at the
/// bottom; rotate it to put the ledge on another side.
///
/// [glow] runs 0..1: the wood brightens and a warm halo appears.
class RackPainter extends CustomPainter {
  const RackPainter({this.glow = 0, this.rows = 1});

  final double glow;

  /// 1 for the far racks; 2 for the player's, like a real rack.
  final int rows;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final radius = Radius.circular(math.min(h * 0.22, 18));
    final body = RRect.fromRectAndRadius(Offset.zero & size, radius);
    final stroke = math.max(2.5, h * 0.04);

    canvas.drawRRect(
      body.shift(Offset(0, h * 0.09)),
      Paint()
        ..color = const Color(0x59000000)
        ..maskFilter = MaskFilter.blur(BlurStyle.normal, h * 0.09),
    );
    if (glow > 0) {
      canvas.drawRRect(
        body.inflate(4 + 4 * glow),
        Paint()
          ..color = const Color(0xFFFFE08A).withValues(alpha: 0.7 * glow)
          ..maskFilter = MaskFilter.blur(BlurStyle.normal, 8 + 6 * glow),
      );
    }

    Color lit(Color color) => Color.lerp(color, Colors.white, 0.14 * glow)!;
    final rowH = h / rows;
    for (var row = 0; row < rows; row++) {
      final top = row * rowH;
      final first = row == 0;
      final last = row == rows - 1;
      final small = Radius.circular(rowH * 0.1);
      final rowRadius = Radius.circular(math.min(rowH * 0.22, 18));
      final backBottom = top + rowH * 0.62;
      final ledgeTop = top + rowH * 0.56;

      final back = RRect.fromLTRBAndCorners(
        0,
        top,
        w,
        backBottom,
        topLeft: first ? rowRadius : small,
        topRight: first ? rowRadius : small,
      );
      canvas.drawRRect(
        back,
        Paint()
          ..shader = LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [lit(_Wood.mid), lit(_Wood.dark)],
          ).createShader(back.outerRect),
      );
      _grain(canvas, Rect.fromLTRB(0, top, w, backBottom));

      final ledge = RRect.fromLTRBAndCorners(
        0,
        ledgeTop,
        w,
        top + rowH,
        topLeft: small,
        topRight: small,
        bottomLeft: last ? rowRadius : small,
        bottomRight: last ? rowRadius : small,
      );
      canvas.drawRRect(
        ledge,
        Paint()
          ..shader = LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [lit(_Wood.light), lit(_Wood.mid)],
          ).createShader(ledge.outerRect),
      );
      final inset = rowH * 0.12 + 4;
      canvas.drawLine(
        Offset(inset, ledgeTop + stroke * 1.4),
        Offset(w - inset, ledgeTop + stroke * 1.4),
        Paint()
          ..color = _Wood.highlight.withValues(alpha: 0.9)
          ..strokeWidth = math.max(2, rowH * 0.045)
          ..strokeCap = StrokeCap.round,
      );
      // Tile slots: short grooves along the ledge, about one tile apart.
      final slots = math.max(3, (w / (rowH * 0.9)).floor());
      final groove = Paint()
        ..color = _Wood.grain.withValues(alpha: 0.55)
        ..strokeWidth = math.max(1.5, rowH * 0.025)
        ..strokeCap = StrokeCap.round;
      for (var i = 1; i < slots; i++) {
        final x = w * i / slots;
        canvas.drawLine(
          Offset(x, ledgeTop + rowH * 0.18),
          Offset(x, top + rowH * 0.92),
          groove,
        );
      }
      canvas.drawLine(
        Offset(0, ledgeTop),
        Offset(w, ledgeTop),
        Paint()
          ..color = _Wood.outline
          ..strokeWidth = stroke * 0.8,
      );
      if (!last) {
        canvas.drawLine(
          Offset(0, top + rowH),
          Offset(w, top + rowH),
          Paint()
            ..color = _Wood.outline
            ..strokeWidth = stroke,
        );
      }
    }

    canvas.drawRRect(
      body,
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = stroke
        ..strokeJoin = StrokeJoin.round
        ..color = _Wood.outline,
    );
  }

  /// Three lazy curves, the cartoon way of saying "wood".
  void _grain(Canvas canvas, Rect area) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round
      ..strokeWidth = math.max(1.5, area.height * 0.04)
      ..color = _Wood.grain.withValues(alpha: 0.35);
    for (final (fx, fy, fw) in const [
      (0.08, 0.30, 0.22),
      (0.46, 0.55, 0.26),
      (0.74, 0.28, 0.18),
    ]) {
      final start =
          Offset(area.left + area.width * fx, area.top + area.height * fy);
      final length = area.width * fw;
      canvas.drawPath(
        Path()
          ..moveTo(start.dx, start.dy)
          ..quadraticBezierTo(
            start.dx + length / 2,
            start.dy - area.height * 0.12,
            start.dx + length,
            start.dy,
          ),
        paint,
      );
    }
  }

  @override
  bool shouldRepaint(RackPainter old) => old.glow != glow || old.rows != rows;
}

/// The board: a green felt trapezoid in a wooden frame with a visible front
/// edge, stitched like a cartoon card table. [glow] 0..1 brightens the felt.
class BoardPainter extends CustomPainter {
  const BoardPainter({required this.inset, this.glow = 0});

  /// How far the far corners are pulled in from the near ones.
  final double inset;
  final double glow;

  /// Height of the front edge of the slab.
  static double slabOf(Size size) => (size.height * 0.05).clamp(8.0, 18.0);

  /// The outer trapezoid, front edge included: also the hover hit area.
  static Path outline(Size size, double inset) {
    return roundedPolygon(
      [
        Offset(inset, 0),
        Offset(size.width - inset, 0),
        Offset(size.width, size.height),
        Offset(0, size.height),
      ],
      22,
    );
  }

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final slab = slabOf(size);
    final topH = size.height - slab;
    final frame = <Offset>[
      Offset(inset, 0),
      Offset(w - inset, 0),
      Offset(w, topH),
      Offset(0, topH),
    ];
    final stroke = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3
      ..strokeJoin = StrokeJoin.round
      ..color = _Wood.outline;

    // Front edge of the slab, under the top face.
    final front = RRect.fromLTRBR(
      0,
      topH - 16,
      w,
      size.height,
      const Radius.circular(14),
    );
    canvas.drawRRect(
      front.shift(const Offset(0, 6)),
      Paint()
        ..color = const Color(0x66000000)
        ..maskFilter = const MaskFilter.blur(BlurStyle.normal, 8),
    );
    canvas.drawRRect(front, Paint()..color = _Wood.dark);
    canvas.drawRRect(front, stroke);

    // Wooden frame.
    final framePath = roundedPolygon(frame, 22);
    canvas.drawPath(
      framePath,
      Paint()
        ..shader = const LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [_Wood.light, _Wood.mid],
        ).createShader(Offset.zero & size),
    );
    canvas.drawPath(framePath, stroke);

    // Felt.
    final rim = (w * 0.02).clamp(8.0, 16.0);
    final felt = roundedPolygon(insetPolygon(frame, rim), 14);
    final lightFelt =
        Color.lerp(const Color(0xFF3CB574), const Color(0xFF66D896), glow)!;
    final darkFelt =
        Color.lerp(const Color(0xFF1B7A4B), const Color(0xFF2C9E64), glow)!;
    canvas.drawPath(
      felt,
      Paint()
        ..shader = RadialGradient(
          center: const Alignment(0, -0.1),
          radius: 0.85,
          colors: [lightFelt, darkFelt],
        ).createShader(Offset.zero & Size(w, topH)),
    );
    canvas.drawPath(
      felt,
      stroke
        ..strokeWidth = 2
        ..color = const Color(0x66000000),
    );

    // Gloss: a pale band across the far third.
    canvas.save();
    canvas.clipPath(felt);
    canvas.drawRect(
      Rect.fromLTWH(0, 0, w, topH * 0.30),
      Paint()..color = Colors.white.withValues(alpha: 0.07 + 0.05 * glow),
    );
    canvas.restore();

    // Stitching.
    _dashed(
      canvas,
      roundedPolygon(insetPolygon(frame, rim + 8), 10),
      Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2
        ..strokeCap = StrokeCap.round
        ..color = Colors.white.withValues(alpha: 0.42),
    );
  }

  void _dashed(Canvas canvas, Path path, Paint paint) {
    for (final metric in path.computeMetrics()) {
      for (var d = 0.0; d < metric.length; d += 14) {
        canvas.drawPath(
            metric.extractPath(d, math.min(d + 7, metric.length)), paint);
      }
    }
  }

  @override
  bool shouldRepaint(BoardPainter old) =>
      old.inset != inset || old.glow != glow;
}

/// [points] (a convex polygon, clockwise on screen) with every corner rounded
/// by [radius] using a quadratic curve through the corner.
Path roundedPolygon(List<Offset> points, double radius) {
  final n = points.length;
  final path = Path();
  for (var i = 0; i < n; i++) {
    final prev = points[(i + n - 1) % n];
    final corner = points[i];
    final next = points[(i + 1) % n];
    final before = corner + _toward(corner, prev, radius);
    final after = corner + _toward(corner, next, radius);
    if (i == 0) {
      path.moveTo(before.dx, before.dy);
    } else {
      path.lineTo(before.dx, before.dy);
    }
    path.quadraticBezierTo(corner.dx, corner.dy, after.dx, after.dy);
  }
  return path..close();
}

/// A vector from [from] toward [to], at most [length] long (never past the
/// midpoint, so short edges still round cleanly).
Offset _toward(Offset from, Offset to, double length) {
  final delta = to - from;
  final distance = delta.distance;
  if (distance == 0) return Offset.zero;
  return delta / distance * math.min(length, distance / 2);
}

/// [points] moved inward by [distance] along every edge's normal. Convex
/// polygons only, clockwise on screen.
List<Offset> insetPolygon(List<Offset> points, double distance) {
  final n = points.length;
  // Each edge i runs points[i] -> points[i + 1], shifted along its inward normal.
  final origins = <Offset>[];
  final directions = <Offset>[];
  for (var i = 0; i < n; i++) {
    final a = points[i];
    final b = points[(i + 1) % n];
    final direction = (b - a) / (b - a).distance;
    final normal = Offset(-direction.dy, direction.dx);
    origins.add(a + normal * distance);
    directions.add(direction);
  }
  return [
    for (var i = 0; i < n; i++)
      _intersect(
        origins[(i + n - 1) % n],
        directions[(i + n - 1) % n],
        origins[i],
        directions[i],
      ),
  ];
}

Offset _intersect(Offset p1, Offset d1, Offset p2, Offset d2) {
  final cross = d1.dx * d2.dy - d1.dy * d2.dx;
  if (cross.abs() < 1e-9) return p2;
  final delta = p2 - p1;
  final t = (delta.dx * d2.dy - delta.dy * d2.dx) / cross;
  return p1 + d1 * t;
}

/// Clips to the board's outline, so hovering the corners of its bounding box
/// does not count as hovering the board.
class BoardClipper extends CustomClipper<Path> {
  const BoardClipper(this.inset);

  final double inset;

  @override
  Path getClip(Size size) => BoardPainter.outline(size, inset);

  @override
  bool shouldReclip(BoardClipper old) => old.inset != inset;
}
