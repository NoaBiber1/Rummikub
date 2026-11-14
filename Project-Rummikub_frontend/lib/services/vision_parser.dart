import 'dart:math' as math;

import 'package:flutter/foundation.dart';

import '../logic/set_validator.dart';
import '../models/models.dart';

/// One tile box from a Roboflow workflow output, after label mapping.
class TileDetection {
  const TileDetection({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.confidence,
    required this.color,
    required this.value,
  });

  final double x;
  final double y;
  final double width;
  final double height;
  final double confidence;
  final String color;
  final int value;

  Tile toTile({int? rackRow}) =>
      Tile(color: color, value: value, rackRow: rackRow);

  String get shortLabel => color == 'JOKER' ? 'J' : '$value';
}

/// Defensive parser for Roboflow Workflow JSON (output keys are not hard-coded).
class VisionParser {
  /// Unknown / unmapped class names still become Joker tiles so the box is shown.
  static const bool passThroughRawDetections = true;

  /// Emergency diagnostic switch. Production groups board rows into melds and
  /// sorts rack rows spatially before constructing the game state.
  static const bool disableSpatialGrouping = false;

  static const defaultConfidenceCutoff = 0.15;
  static int lastRawCount = 0;
  static int lastRecognized = 0;
  static int lastKept = 0;
  static List<String> lastSampleLabels = const [];

  /// Maps workflow class names such as `Red7`, `Orange4`, `Joker` onto app tiles.
  ///
  /// Orange is the model's name for the fourth color; the app uses `YELLOW`.
  static (String color, int value)? parseLabel(String raw) {
    final compact = raw.trim().replaceAll(RegExp(r'[\s_-]+'), '');
    if (compact.isEmpty) return null;

    if (RegExp(r'^jokers?$', caseSensitive: false).hasMatch(compact)) {
      return ('JOKER', 0);
    }

    final match = RegExp(
      r'^(red|blue|black|yellow|orange)(\d{1,2})$',
      caseSensitive: false,
    ).firstMatch(compact);
    if (match == null) return null;

    final value = int.tryParse(match.group(2)!);
    if (value == null || value < 1 || value > 13) return null;

    const colors = {
      'RED': 'RED',
      'BLUE': 'BLUE',
      'BLACK': 'BLACK',
      'YELLOW': 'YELLOW',
      'ORANGE': 'YELLOW',
    };
    final color = colors[match.group(1)!.toUpperCase()];
    if (color == null) return null;
    return (color, value);
  }

  /// Pulls detections out of a workflow run body without assuming output names.
  static List<TileDetection> detectionsFromResponse(
    dynamic decoded, {
    double confidenceCutoff = defaultConfidenceCutoff,
  }) {
    final entries = _outputEntries(decoded);
    if (entries.isEmpty) {
      throw const FormatException(
        'Roboflow returned no workflow outputs to parse.',
      );
    }

    List<Map<String, dynamic>>? best;
    var bestScore = -1;
    var foundPredictionList = false;

    for (final entry in entries) {
      final candidates = <List<Map<String, dynamic>>>[];
      _collectPredictionLists(entry, candidates);
      if (candidates.isEmpty) continue;
      foundPredictionList = true;
      for (final candidate in candidates) {
        var recognizedHere = 0;
        for (final raw in candidate) {
          final label =
              (raw['class'] ?? raw['class_name'] ?? raw['className'] ?? '')
                  .toString();
          if (parseLabel(label) != null) recognizedHere += 1;
        }
        final score = recognizedHere * 1000 + candidate.length;
        if (best == null || score > bestScore) {
          best = candidate;
          bestScore = score;
        }
      }
    }

    if (!foundPredictionList || best == null) {
      throw const FormatException(
        'Roboflow workflow returned no tile detections. Check the workflow outputs.',
      );
    }

    final detections = <TileDetection>[];
    var recognized = 0;
    String? sampleUnknown;
    final sampleLabels = <String>[];
    for (final raw in best) {
      final label =
          (raw['class'] ?? raw['class_name'] ?? raw['className'] ?? '')
              .toString();
      if (sampleLabels.length < 12 && label.isNotEmpty) {
        sampleLabels.add(label);
      }

      var confidence = (raw['confidence'] as num?)?.toDouble() ??
          (raw['confidence_score'] as num?)?.toDouble() ??
          1.0;
      if (confidence > 1.0) confidence /= 100.0;

      var parsed = parseLabel(label);
      if (parsed == null) {
        if (label.isNotEmpty) sampleUnknown ??= label;
        // The Roboflow Workflow labels boxes the classifier could not confirm as
        // "Unknown". Never turn those into a Joker: a wrong wild tile silently
        // changes the solver's answer, whereas a missing tile is visible and
        // correctable on the correction screen.
        if (label.toLowerCase() == 'unknown') continue;
        if (!passThroughRawDetections) continue;
        // Other unrecognized class names still become tiles so the raw box reaches the UI.
        parsed = ('JOKER', 0);
      } else {
        recognized += 1;
      }
      if (confidence < confidenceCutoff) continue;

      detections.add(
        TileDetection(
          x: (raw['x'] as num?)?.toDouble() ?? 0,
          y: (raw['y'] as num?)?.toDouble() ?? 0,
          width: (raw['width'] as num?)?.toDouble() ?? 0,
          height: (raw['height'] as num?)?.toDouble() ?? 0,
          confidence: confidence,
          color: parsed.$1,
          value: parsed.$2,
        ),
      );
    }

    lastRawCount = best.length;
    lastRecognized = recognized;
    lastKept = detections.length;
    lastSampleLabels = List<String>.from(sampleLabels);
    if (kDebugMode) {
      debugPrint('Raw detections: ${best.length}');
      debugPrint('After filtering: ${detections.length}');
    }
    if (best.isNotEmpty && recognized == 0 && !passThroughRawDetections) {
      final sample = sampleUnknown ?? 'unknown';
      throw FormatException(
        'Roboflow returned tile labels we do not recognize (example: "$sample"). '
        'Expected ColorNumber such as Red7, Orange4, or Joker.',
      );
    }

    return detections;
  }

  /// Table photo → board tiles; rack photo → rack tiles.
  ///
  /// When [disableSpatialGrouping] is true, each detection is a standalone tile
  /// (no row clustering, gap splitting, or series/meld grouping).
  static GameState gameStateFromPhotos({
    required List<TileDetection> boardDetections,
    required List<TileDetection> rackDetections,
  }) {
    return GameState(
      board: boardSetsFromDetections(boardDetections),
      rack: rackTilesFromDetections(rackDetections),
    );
  }

  static List<TileSet> boardSetsFromDetections(List<TileDetection> detections) {
    if (detections.isEmpty) return const [];
    if (disableSpatialGrouping) {
      return [
        for (final detection in detections)
          TileSet(type: 'INVALID', tiles: [detection.toTile()]),
      ];
    }
    return _groupBoard(detections)
        .map(
          (group) => TileSet(
            type: 'INVALID',
            tiles: group.map((d) => d.toTile()).toList(),
          ),
        )
        .toList();
  }

  static List<Tile> rackTilesFromDetections(List<TileDetection> detections) {
    if (detections.isEmpty) return const [];
    if (disableSpatialGrouping) {
      return [for (final detection in detections) detection.toTile()];
    }
    final shelves = _rackShelves(detections);
    return [
      for (var shelf = 0; shelf < shelves.length; shelf++)
        for (final d in shelves[shelf]) d.toTile(rackRow: shelf),
    ];
  }

  static List<Map<String, dynamic>> _outputEntries(dynamic decoded) {
    if (decoded is List) {
      return [
        for (final item in decoded)
          if (item is Map) _asStringMap(item),
      ];
    }
    if (decoded is! Map) return const [];
    final map = _asStringMap(decoded);
    for (final key in ['outputs', 'output', 'result', 'results']) {
      final value = map[key];
      if (value is List) {
        return [
          for (final item in value)
            if (item is Map) _asStringMap(item),
        ];
      }
      if (value is Map) return [_asStringMap(value)];
    }
    return [map];
  }

  static void _collectPredictionLists(
    dynamic node,
    List<List<Map<String, dynamic>>> candidates, {
    String? key,
  }) {
    if (_isImageBlob(node, key)) return;

    if (node is List) {
      final detections = <Map<String, dynamic>>[];
      var maps = 0;
      for (final item in node) {
        if (item is! Map) continue;
        maps += 1;
        final asMap = _asStringMap(item);
        if (_looksLikeDetection(asMap)) detections.add(asMap);
      }
      if (detections.isNotEmpty && detections.length == maps) {
        candidates.add(detections);
        return;
      }
      if (detections.isNotEmpty) {
        candidates.add(detections);
      }
      for (final item in node) {
        if (item is Map && _looksLikeDetection(_asStringMap(item))) continue;
        _collectPredictionLists(item, candidates, key: key);
      }
      return;
    }

    if (node is! Map) return;
    if (_looksLikeDetection(_asStringMap(node))) return;

    final map = _asStringMap(node);
    final wrapped = map['predictions'];
    if (wrapped is List) {
      candidates.add([
        for (final item in wrapped)
          if (item is Map && _looksLikeDetection(_asStringMap(item)))
            _asStringMap(item),
      ]);
    }

    for (final entry in map.entries) {
      if (entry.key == 'predictions' && entry.value is List) continue;
      _collectPredictionLists(entry.value, candidates, key: entry.key);
    }
  }

  static bool _looksLikeDetection(Map<String, dynamic> map) {
    final label = map['class'] ?? map['class_name'] ?? map['className'];
    final hasLabel =
        (label is String && label.isNotEmpty) || map['class_id'] != null;
    return hasLabel && map['x'] is num && map['y'] is num;
  }

  static bool _isImageBlob(dynamic value, String? key) {
    final keyLower = key?.toLowerCase() ?? '';
    if (keyLower.contains('visualization') ||
        keyLower.contains('output_image') ||
        keyLower == 'label_visualization') {
      return true;
    }
    if (value is String && value.length > 512) return true;
    if (value is Map) {
      final type = value['type']?.toString().toLowerCase();
      if (type == 'base64' || type == 'jpeg' || type == 'png') return true;
      final inner = value['value'];
      if (inner is String && inner.length > 512) return true;
    }
    return false;
  }

  /// A new shelf starts where tiles are this many tile heights apart across
  /// the shelves. Measured on real rack photos: the gap between the two shelves
  /// is 0.8-1.05 tile heights, spacing inside a shelf at most 0.35.
  static const double _shelfGapRatio = 0.5;

  /// Rack photos are held by hand and shot at an angle, so shelves are rarely
  /// level and their y ranges overlap: grouping by y alone merges them into
  /// one row or shreds them. Instead the photo is straightened first, using
  /// the direction along the shelves ([_rackTilt]), then cut across them.
  ///
  /// Returns at most two shelves (a real rack has two), top first, each read
  /// left to right along the shelf. Detections that sit apart from both main
  /// shelves (a stray box on the background) join the nearest one.
  static List<List<TileDetection>> _rackShelves(
    List<TileDetection> detections,
  ) {
    final tilt = _rackTilt(detections);
    final cosT = math.cos(tilt);
    final sinT = math.sin(tilt);
    // u runs along the shelves, v across them.
    double u(TileDetection d) => d.x * cosT + d.y * sinT;
    double v(TileDetection d) => d.y * cosT - d.x * sinT;

    final byV = [...detections]..sort((a, b) => v(a).compareTo(v(b)));
    final gap = _median([for (final d in detections) d.height]) * _shelfGapRatio;
    final clusters = <List<TileDetection>>[
      [byV.first],
    ];
    for (final d in byV.skip(1)) {
      if (v(d) - v(clusters.last.last) > gap) {
        clusters.add([d]);
      } else {
        clusters.last.add(d);
      }
    }

    var shelves = clusters;
    if (clusters.length > 2) {
      double meanV(List<TileDetection> c) =>
          c.fold<double>(0, (sum, d) => sum + v(d)) / c.length;
      final bySize = [for (var i = 0; i < clusters.length; i++) i]..sort((a, b) {
          final bySize = clusters[b].length.compareTo(clusters[a].length);
          return bySize != 0 ? bySize : a.compareTo(b);
        });
      final anchors = [bySize[0], bySize[1]]..sort();
      shelves = [for (final i in anchors) [...clusters[i]]];
      for (var i = 0; i < clusters.length; i++) {
        if (anchors.contains(i)) continue;
        final nearest = (meanV(clusters[i]) - meanV(shelves[0])).abs() <=
                (meanV(clusters[i]) - meanV(shelves[1])).abs()
            ? 0
            : 1;
        shelves[nearest].addAll(clusters[i]);
      }
    }
    for (final shelf in shelves) {
      shelf.sort((a, b) => u(a).compareTo(u(b)));
    }
    return shelves;
  }

  /// Angle (radians) of the shelves against the image's x axis: the median
  /// direction from each tile to its nearest neighbour on its right, within a
  /// 30 degree cone. The cone keeps out the tiles of the other shelf, which
  /// are often the nearest ones overall. Zero when there are too few pairs.
  static double _rackTilt(List<TileDetection> detections) {
    const cone = math.pi / 6;
    final angles = <double>[];
    for (final a in detections) {
      double? bestDist;
      double? bestAngle;
      for (final b in detections) {
        final dx = b.x - a.x;
        if (dx <= 0) continue;
        final dy = b.y - a.y;
        final angle = math.atan2(dy, dx);
        if (angle.abs() > cone) continue;
        final dist = math.sqrt(dx * dx + dy * dy);
        if (bestDist == null || dist < bestDist) {
          bestDist = dist;
          bestAngle = angle;
        }
      }
      if (bestAngle != null) angles.add(bestAngle);
    }
    return angles.length >= 3 ? _median(angles) : 0;
  }

  // ---------------------------------------------------------------------
  // Rotation-aware board grouping.
  //
  // Mirrors project_rummikub_backend/vision_grouping.py (_group_board) and uses
  // the same thresholds: tiles are linked when their centers are at most about
  // one tile pitch apart (the median nearest-neighbour center distance, which
  // does not depend on rotation or box padding), components with several
  // parallel lines are split along the perpendicular axis, and every meld is
  // ordered along its principal axis so melds lying at any angle (even upside
  // down) read in ascending order. Two rule-based guardrails follow: an invalid
  // line that cuts into valid sets is cut, and end-to-end fragments whose
  // union is a valid set are rejoined.
  // ---------------------------------------------------------------------

  /// Link tolerance above the pitch: 3 * MAD of the nearest-neighbour center
  /// distances, clamped to this range. Stacked rows sit ~1.5 pitches apart.
  static const double _pitchTolerance = 0.15;
  static const double _maxPitchTolerance = 0.30;

  /// Fragments whose facing ends are this many pitches apart may be rejoined.
  static const double _rejoinRatio = 1.6;
  static const double _meldSpreadRatio = 0.6;
  static const double _meldClusterRatio = 0.5;

  static bool _hasGeometry(TileDetection d) =>
      d.x.isFinite &&
      d.y.isFinite &&
      d.width.isFinite &&
      d.height.isFinite &&
      d.width > 0 &&
      d.height > 0;

  static double _median(List<double> values) {
    final sorted = [...values]..sort();
    final mid = sorted.length ~/ 2;
    if (sorted.length.isOdd) return sorted[mid];
    return (sorted[mid - 1] + sorted[mid]) / 2;
  }

  /// Distance between two box centers.
  static double _centerDist(TileDetection a, TileDetection b) {
    final dx = a.x - b.x;
    final dy = a.y - b.y;
    return math.sqrt(dx * dx + dy * dy);
  }

  /// Connected components (as index lists) of boxes whose centers are at most
  /// [threshold] apart.
  static List<List<int>> _linkComponents(
    List<int> items,
    List<TileDetection> all,
    double threshold,
  ) {
    final parent = List<int>.generate(items.length, (i) => i);
    int find(int i) {
      while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
      }
      return i;
    }

    for (var i = 0; i < items.length; i++) {
      for (var j = i + 1; j < items.length; j++) {
        if (_centerDist(all[items[i]], all[items[j]]) <= threshold) {
          parent[find(j)] = find(i);
        }
      }
    }
    final groups = <int, List<int>>{};
    for (var i = 0; i < items.length; i++) {
      groups.putIfAbsent(find(i), () => []).add(items[i]);
    }
    return groups.values.toList();
  }

  /// Unit axis of the box centers; sign chosen so x > 0 (else y > 0).
  ///
  /// Two tiles use the vector between them, three or more the dominant
  /// eigenvector of the 2x2 covariance matrix.
  static (double, double) _principalAxis(
    List<int> items,
    List<TileDetection> all,
  ) {
    var ax = 1.0;
    var ay = 0.0;
    if (items.length == 2) {
      final dx = all[items[1]].x - all[items[0]].x;
      final dy = all[items[1]].y - all[items[0]].y;
      final norm = math.sqrt(dx * dx + dy * dy);
      if (norm < 1e-12) return (1.0, 0.0);
      ax = dx / norm;
      ay = dy / norm;
    } else {
      var cx = 0.0;
      var cy = 0.0;
      for (final i in items) {
        cx += all[i].x;
        cy += all[i].y;
      }
      cx /= items.length;
      cy /= items.length;
      var sxx = 0.0;
      var syy = 0.0;
      var sxy = 0.0;
      for (final i in items) {
        final dx = all[i].x - cx;
        final dy = all[i].y - cy;
        sxx += dx * dx;
        syy += dy * dy;
        sxy += dx * dy;
      }
      final theta = 0.5 * math.atan2(2 * sxy, sxx - syy);
      ax = math.cos(theta);
      ay = math.sin(theta);
    }
    if (ax.abs() < 1e-6) {
      ax = 0.0;
      ay = ay.abs();
    } else if (ax < 0) {
      ax = -ax;
      ay = -ay;
    }
    return (ax, ay);
  }

  /// Splits a component of >= 3 tiles into parallel lines if it is not one line.
  static List<List<int>> _perpendicularClusters(
    List<int> items,
    List<TileDetection> all,
    double tileLong,
  ) {
    final (ax, ay) = _principalAxis(items, all);
    var cx = 0.0;
    var cy = 0.0;
    for (final i in items) {
      cx += all[i].x;
      cy += all[i].y;
    }
    cx /= items.length;
    cy /= items.length;

    final keyed = [
      for (final i in items) (-ay * (all[i].x - cx) + ax * (all[i].y - cy), i),
    ]..sort((a, b) {
        final byOffset = a.$1.compareTo(b.$1);
        return byOffset != 0 ? byOffset : a.$2.compareTo(b.$2);
      });

    // Full perpendicular range (max - min): touching rows of upright tiles are
    // only one tile height apart (+-0.5 * tileLong from the centroid), which a
    // max |offset| <= 0.6 * tileLong test would never split.
    if (keyed.last.$1 - keyed.first.$1 <= _meldSpreadRatio * tileLong) {
      return [items];
    }
    final clusters = <List<int>>[
      [keyed.first.$2],
    ];
    for (var k = 1; k < keyed.length; k++) {
      if (keyed[k].$1 - keyed[k - 1].$1 > _meldClusterRatio * tileLong) {
        clusters.add([]);
      }
      clusters.last.add(keyed[k].$2);
    }
    return clusters;
  }

  /// Orders [items] by projection on the principal axis, then flips the order
  /// when non-joker values mostly descend (the meld was photographed rotated).
  static List<int> _orderAlongAxis(List<int> items, List<TileDetection> all) {
    if (items.length < 2) return [...items];
    final (ax, ay) = _principalAxis(items, all);
    final ordered = [...items]..sort((a, b) {
        final pa = all[a].x * ax + all[a].y * ay;
        final pb = all[b].x * ax + all[b].y * ay;
        final byProjection = pa.compareTo(pb);
        return byProjection != 0 ? byProjection : a.compareTo(b);
      });

    var ascending = 0;
    var descending = 0;
    int? previous;
    for (final i in ordered) {
      final tile = all[i];
      if (tile.color == 'JOKER') continue;
      if (previous != null) {
        if (tile.value > previous) {
          ascending += 1;
        } else if (tile.value < previous) {
          descending += 1;
        }
      }
      previous = tile.value;
    }
    return descending > ascending ? ordered.reversed.toList() : ordered;
  }

  /// Cuts an axis-ordered line wherever neighbouring centers are farther apart
  /// than [threshold] (local gap peaks; also breaks L-shaped chains).
  static List<List<int>> _splitAtGapPeaks(
    List<int> ordered,
    List<TileDetection> all,
    double threshold,
  ) {
    if (ordered.isEmpty || threshold <= 0) return [ordered];
    final pieces = <List<int>>[
      [ordered.first],
    ];
    for (var k = 1; k < ordered.length; k++) {
      if (_centerDist(all[ordered[k - 1]], all[ordered[k]]) > threshold) {
        pieces.add([]);
      }
      pieces.last.add(ordered[k]);
    }
    return pieces;
  }

  static bool _isValidMeld(List<int> items, List<TileDetection> all) {
    final tiles = [for (final i in items) all[i].toTile()];
    return SetValidator.setTilesAreValid(tiles) ||
        SetValidator.setTilesAreValid(tiles.reversed.toList());
  }

  /// Guardrail: an invalid line that cuts into valid sets is cut there (a run
  /// and a group touching on one line).
  static List<List<int>> _splitIntoValidSets(
    List<int> line,
    List<TileDetection> all,
  ) {
    if (line.length < 2 * SetValidator.minSetSize || _isValidMeld(line, all)) {
      return [line];
    }
    final pieces = SetValidator.splitIntoValidSets(
      [for (final i in line) all[i].toTile()],
    );
    if (pieces == null) return [line];
    final out = <List<int>>[];
    var start = 0;
    for (final piece in pieces) {
      out.add(line.sublist(start, start + piece.length));
      start += piece.length;
    }
    return out;
  }

  /// Guardrail: rejoins end-to-end fragments when only their union is a valid
  /// set (a sloppy gap inside a real meld). Edits [parts] in place.
  static void _rejoinFragments(
    List<List<int>> parts,
    List<TileDetection> all,
    double reach,
  ) {
    var merged = true;
    while (merged) {
      merged = false;
      outer:
      for (var a = 0; a < parts.length; a++) {
        for (var b = 0; b < parts.length; b++) {
          if (a == b) continue;
          if (_centerDist(all[parts[a].last], all[parts[b].first]) > reach) {
            continue;
          }
          if (_isValidMeld(parts[a], all) && _isValidMeld(parts[b], all)) {
            continue;
          }
          final union = _orderAlongAxis([...parts[a], ...parts[b]], all);
          if (!_isValidMeld(union, all)) continue;
          parts[a] = union;
          parts.removeAt(b);
          merged = true;
          break outer;
        }
      }
    }
  }

  /// Groups board detections into meld candidates at any rotation.
  static List<List<TileDetection>> _groupBoard(List<TileDetection> detections) {
    final usable = [
      for (var i = 0; i < detections.length; i++)
        if (_hasGeometry(detections[i])) i,
    ];
    // Detections without usable geometry cannot be grouped; keep them as
    // single-tile sets so nothing is silently lost.
    final unusable = [
      for (var i = 0; i < detections.length; i++)
        if (!_hasGeometry(detections[i])) [detections[i]],
    ];
    if (usable.isEmpty) return unusable;

    List<List<int>> components;
    var tileLong = 0.0;
    var pitch = 0.0;
    var linkThreshold = 0.0;
    if (usable.length == 1) {
      components = [usable];
    } else {
      final tileShort = _median([
        for (final i in usable)
          math.min(detections[i].width, detections[i].height),
      ]);
      tileLong = _median([
        for (final i in usable)
          math.max(detections[i].width, detections[i].height),
      ]);
      final nearest = <double>[
        for (final i in usable)
          usable
              .where((j) => j != i)
              .map((j) => _centerDist(detections[i], detections[j]))
              .reduce(math.min),
      ];
      // The floor guards against duplicate boxes sharing one center.
      pitch = math.max(_median(nearest), 0.5 * tileShort);
      final mad = _median([for (final d in nearest) (d - pitch).abs()]);
      final tolerance =
          (3 * mad / pitch).clamp(_pitchTolerance, _maxPitchTolerance);
      linkThreshold = pitch * (1 + tolerance);
      components = _linkComponents(usable, detections, linkThreshold);
    }

    // Axis-ordered member indices of every meld candidate.
    final parts = <List<int>>[];
    for (final component in components) {
      final clusters = component.length >= 3
          ? _perpendicularClusters(component, detections, tileLong)
          : [component];
      final lines = clusters.length == 1
          ? [component]
          : [
              for (final cluster in clusters)
                ..._linkComponents(cluster, detections, linkThreshold),
            ];
      for (final line in lines) {
        final ordered = _orderAlongAxis(line, detections);
        for (final piece
            in _splitAtGapPeaks(ordered, detections, linkThreshold)) {
          parts.addAll(_splitIntoValidSets(piece, detections));
        }
      }
    }
    if (pitch > 0) _rejoinFragments(parts, detections, _rejoinRatio * pitch);

    // (centroid y, centroid x, ordered member indices)
    final entries = <(double, double, List<int>)>[];
    for (final part in parts) {
      var cx = 0.0;
      var cy = 0.0;
      for (final i in part) {
        cx += detections[i].x;
        cy += detections[i].y;
      }
      entries.add((
        cy / part.length,
        cx / part.length,
        _orderAlongAxis(part, detections),
      ));
    }

    // Deterministic output order: centroid (y, x).
    final indexed = [for (var k = 0; k < entries.length; k++) (k, entries[k])];
    indexed.sort((a, b) {
      final byY = a.$2.$1.compareTo(b.$2.$1);
      if (byY != 0) return byY;
      final byX = a.$2.$2.compareTo(b.$2.$2);
      return byX != 0 ? byX : a.$1.compareTo(b.$1);
    });
    return [
      for (final entry in indexed) [for (final i in entry.$2.$3) detections[i]],
      ...unusable,
    ];
  }

  static Map<String, dynamic> _asStringMap(Map<dynamic, dynamic> map) {
    return map.map((key, value) => MapEntry(key.toString(), value));
  }
}
