import 'dart:convert';
import 'dart:typed_data';

import 'vision_parser.dart';

/// Exact JPEG + Workflow JSON the app sent/received.
class RoboflowDebugDump {
  const RoboflowDebugDump({
    required this.tableJpeg,
    required this.rackJpeg,
    required this.tableJson,
    required this.rackJson,
    required this.unfilteredSummary,
    this.tableDetections = const [],
    this.rackDetections = const [],
  });

  final Uint8List tableJpeg;
  final Uint8List rackJpeg;
  final String tableJson;
  final String rackJson;
  final String unfilteredSummary;
  final List<TileDetection> tableDetections;
  final List<TileDetection> rackDetections;
}

String workflowJsonForExport(dynamic decoded) {
  const encoder = JsonEncoder.withIndent('  ');
  return encoder.convert(_omitHugeBlobs(decoded));
}

dynamic _omitHugeBlobs(dynamic node, [String? key]) {
  if (node is String && node.length > 400) {
    return '[omitted ${node.length} chars]';
  }
  if (node is Map) {
    return {
      for (final entry in node.entries)
        entry.key.toString(): _omitHugeBlobs(
          entry.value,
          entry.key.toString(),
        ),
    };
  }
  if (node is List) {
    final keyLower = key?.toLowerCase() ?? '';
    if (keyLower.contains('visual') || keyLower.contains('image')) {
      return '[omitted ${node.length} items]';
    }
    return [for (final item in node) _omitHugeBlobs(item, key)];
  }
  return node;
}
