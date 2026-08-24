import 'dart:convert';

import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import '../models/models.dart';

/// Calls the Roboflow hosted YOLOv8 model and maps detections to a [GameState].
class VisionService {
  VisionService({
    this.apiKey = const String.fromEnvironment('ROBOFLOW_API_KEY'),
    this.modelId = const String.fromEnvironment(
      'ROBOFLOW_MODEL_ID',
      defaultValue: 'rummikub-tiles',
    ),
    this.modelVersion = const String.fromEnvironment(
      'ROBOFLOW_MODEL_VERSION',
      defaultValue: '1',
    ),
  });

  final String apiKey;
  final String modelId;
  final String modelVersion;

  /// Sends [image] to Roboflow and returns the recognized board and rack.
  ///
  /// If the API key is missing or the request fails, returns [_mockGameState]
  /// so the correction flow can still be exercised.
  Future<GameState> processImage(XFile image) async {
    if (apiKey.isEmpty) {
      return _mockGameState();
    }

    try {
      final bytes = await image.readAsBytes();
      final uri = Uri.parse(
        'https://detect.roboflow.com/$modelId/$modelVersion?api_key=$apiKey',
      );

      final response = await http.post(
        uri,
        headers: const {'Content-Type': 'application/x-www-form-urlencoded'},
        body: base64Encode(bytes),
      );

      if (response.statusCode < 200 || response.statusCode >= 300) {
        return _mockGameState();
      }

      final decoded = jsonDecode(response.body);
      if (decoded is! Map<String, dynamic>) {
        return _mockGameState();
      }

      final rawPredictions = decoded['predictions'];
      if (rawPredictions is! List) {
        return _mockGameState();
      }

      final detections = rawPredictions
          .whereType<Map<String, dynamic>>()
          .map(_Detection.fromJson)
          .where((detection) => detection != null)
          .cast<_Detection>()
          .toList();

      return _gameStateFromDetections(detections);
    } catch (_) {
      return _mockGameState();
    }
  }

  /// Sample board + rack used when Roboflow is unavailable.
  GameState _mockGameState() {
    return GameState(
      board: [
        TileSet(
          type: 'RUN',
          tiles: [
            Tile(color: 'RED', value: 7),
            Tile(color: 'RED', value: 8),
            Tile(color: 'RED', value: 9),
          ],
        ),
        TileSet(
          type: 'GROUP',
          tiles: [
            Tile(color: 'BLACK', value: 4),
            Tile(color: 'BLUE', value: 4),
            Tile(color: 'YELLOW', value: 4),
          ],
        ),
        TileSet(
          type: 'INVALID',
          tiles: [
            Tile(color: 'RED', value: 10),
            Tile(color: 'RED', value: 11),
          ],
        ),
      ],
      rack: [
        Tile(color: 'BLUE', value: 2),
        Tile(color: 'YELLOW', value: 13),
        Tile(color: 'BLACK', value: 6),
        Tile(color: 'JOKER', value: 0),
      ],
    );
  }

  GameState _gameStateFromDetections(List<_Detection> detections) {
    if (detections.isEmpty) {
      return GameState.empty();
    }

    final rows = _clusterRows(detections);
    if (rows.isEmpty) {
      return GameState.empty();
    }

    // Bottom of the photo (largest y) is the player's rack; rows above are sets.
    final rackTiles = rows.last.map((d) => d.toTile()).toList();
    final board = rows
        .take(rows.length - 1)
        .expand(_splitRowIntoSets)
        .toList();

    return GameState(board: board, rack: rackTiles);
  }

  List<List<_Detection>> _clusterRows(List<_Detection> detections) {
    final sorted = [...detections]..sort((a, b) => a.y.compareTo(b.y));
    final avgHeight =
        sorted.fold<double>(0, (sum, d) => sum + d.height) / sorted.length;
    final threshold = avgHeight * 0.6;

    final rows = <List<_Detection>>[];
    for (final detection in sorted) {
      if (rows.isEmpty || (detection.y - rows.last.last.y).abs() > threshold) {
        rows.add([detection]);
      } else {
        rows.last.add(detection);
      }
    }

    for (final row in rows) {
      row.sort((a, b) => a.x.compareTo(b.x));
    }
    return rows;
  }

  Iterable<TileSet> _splitRowIntoSets(List<_Detection> row) {
    if (row.isEmpty) return const [];

    final avgWidth =
        row.fold<double>(0, (sum, d) => sum + d.width) / row.length;
    final gapThreshold = avgWidth * 1.5;

    final groups = <List<_Detection>>[
      [row.first],
    ];
    for (var i = 1; i < row.length; i++) {
      final gap = row[i].x - row[i - 1].x;
      if (gap > gapThreshold) {
        groups.add([row[i]]);
      } else {
        groups.last.add(row[i]);
      }
    }

    return groups.map(
      (group) => TileSet(
        type: 'INVALID',
        tiles: group.map((d) => d.toTile()).toList(),
      ),
    );
  }
}

class _Detection {
  const _Detection({
    required this.x,
    required this.y,
    required this.width,
    required this.height,
    required this.color,
    required this.value,
  });

  final double x;
  final double y;
  final double width;
  final double height;
  final String color;
  final int value;

  static _Detection? fromJson(Map<String, dynamic> json) {
    final label = (json['class'] ?? json['class_name'] ?? '').toString();
    final parsed = _parseLabel(label);
    if (parsed == null) return null;

    return _Detection(
      x: (json['x'] as num?)?.toDouble() ?? 0,
      y: (json['y'] as num?)?.toDouble() ?? 0,
      width: (json['width'] as num?)?.toDouble() ?? 0,
      height: (json['height'] as num?)?.toDouble() ?? 0,
      color: parsed.$1,
      value: parsed.$2,
    );
  }

  Tile toTile() => Tile(color: color, value: value);

  static (String, int)? _parseLabel(String raw) {
    final label = raw.trim().toUpperCase().replaceAll('-', '_');
    if (label.isEmpty) return null;
    if (label.contains('JOKER')) return ('JOKER', 0);

    final parts = label.split('_');
    if (parts.length < 2) return null;

    final color = parts.first;
    final value = int.tryParse(parts.last);
    if (value == null) return null;
    if (!['RED', 'BLUE', 'BLACK', 'YELLOW'].contains(color)) return null;
    if (value < 1 || value > 13) return null;
    return (color, value);
  }
}
