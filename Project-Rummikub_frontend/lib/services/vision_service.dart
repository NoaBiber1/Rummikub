import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

import '../config/roboflow_config.dart';
import '../config/solver_config.dart';
import '../models/models.dart';
import 'roboflow_debug_dump.dart';
import 'vision_image.dart';
import 'vision_parser.dart';

/// User-visible failure from the Roboflow workflow client.
class VisionException implements Exception {
  const VisionException(this.message, {this.statusCode});

  final String message;
  final int? statusCode;

  @override
  String toString() => message;
}

/// Calls the Roboflow Workflow "Rummikub v1 Logic" over REST and maps tiles.
class VisionService {
  VisionService({
    http.Client? httpClient,
    String? apiKey,
    String? workflowUrl,
    String? fallbackUrl,
    double? confidenceCutoff,
    this.timeout = const Duration(seconds: 60),
    this.maxAttempts = 3,
  })  : _client = httpClient ?? http.Client(),
        _apiKey = apiKey ?? RoboflowConfig.apiKey,
        _workflowUrl = workflowUrl ??
            (kIsWeb ? SolverConfig.visionUrl : RoboflowConfig.runUrl),
        _fallbackUrl =
            fallbackUrl ?? (kIsWeb ? '' : RoboflowConfig.inferFallbackUrl),
        _confidenceCutoff = confidenceCutoff ?? RoboflowConfig.confidenceCutoff;

  final http.Client _client;
  final String _apiKey;
  final String _workflowUrl;
  final String _fallbackUrl;
  final double _confidenceCutoff;
  final Duration timeout;
  final int maxAttempts;

  final List<String> lastSummaries = [];
  RoboflowDebugDump? lastDump;

  /// Roboflow itself needs the key. The backend proxy (web, or a
  /// `ROBOFLOW_WORKFLOW_URL` pointing at it) supplies its own key, or runs the
  /// local pipeline and needs none.
  bool get requiresApiKey =>
      Uri.tryParse(_workflowUrl)?.host.endsWith('roboflow.com') ?? true;

  /// False when recognition cannot work at all with this build's settings.
  bool get isConfigured => !requiresApiKey || _apiKey.isNotEmpty;

  /// Runs the workflow on [boardImage] and [rackImage], then builds a [GameState].
  Future<GameState> processBoardAndRack({
    required XFile boardImage,
    required XFile rackImage,
  }) async {
    return processBoardAndRackFromBytes(
      boardImage: await boardImage.readAsBytes(),
      rackImage: await rackImage.readAsBytes(),
    );
  }

  Future<GameState> processBoardAndRackFromBytes({
    required Uint8List boardImage,
    required Uint8List rackImage,
  }) async {
    lastSummaries.clear();
    // The photos are independent, so both uploads and both inferences
    // overlap: wall time is the slower photo, not the sum. eagerError reports
    // the first failure at once instead of waiting out the other retries.
    final raw = await Future.wait<dynamic>(
      [runWorkflow(boardImage), runWorkflow(rackImage)],
      eagerError: true,
    );
    final boardRaw = raw[0];
    final rackRaw = raw[1];
    final List<TileDetection> boardAll;
    final List<TileDetection> rackAll;
    final int tableRawCount;
    try {
      boardAll = VisionParser.detectionsFromResponse(
        boardRaw,
        confidenceCutoff: _confidenceCutoff,
      );
      tableRawCount = VisionParser.lastRawCount;
      rackAll = VisionParser.detectionsFromResponse(
        rackRaw,
        confidenceCutoff: _confidenceCutoff,
      );
    } on FormatException catch (error) {
      throw VisionException(error.message);
    }
    // The dump references the photos instead of copying them: the bytes that
    // were sent exist once, in the capture screen.
    lastDump = RoboflowDebugDump(
      tableJpeg: boardImage,
      rackJpeg: rackImage,
      tableJson: workflowJsonForExport(boardRaw),
      rackJson: workflowJsonForExport(rackRaw),
      unfilteredSummary:
          'table raw=$tableRawCount kept=${boardAll.length} ${_fmt(boardAll)} | '
          'rack raw=${VisionParser.lastRawCount} kept=${rackAll.length} ${_fmt(rackAll)}',
      tableDetections: boardAll,
      rackDetections: rackAll,
    );
    lastSummaries.addAll([
      '${boardImage.length}B table raw=${boardAll.length} ${_fmt(boardAll)}',
      '${rackImage.length}B rack raw=${rackAll.length} ${_fmt(rackAll)}',
    ]);
    return VisionParser.gameStateFromPhotos(
      boardDetections: boardAll,
      rackDetections: rackAll,
    );
  }

  static String _fmt(List<TileDetection> detections) {
    return detections
        .take(12)
        .map((d) => '${d.color}${d.value}@${d.confidence.toStringAsFixed(2)}')
        .toList()
        .toString();
  }

  /// Detects tiles in a single photo via [runWorkflow].
  Future<List<TileDetection>> detectTiles(XFile image) async {
    final bytes = await image.readAsBytes();
    return detectTilesFromBytes(bytes);
  }

  Future<List<TileDetection>> detectTilesFromBytes(Uint8List bytes) async {
    final decoded = await runWorkflow(bytes);
    try {
      final detections = VisionParser.detectionsFromResponse(
        decoded,
        confidenceCutoff: _confidenceCutoff,
      );
      lastSummaries.add(
        '${bytes.length}B raw=${VisionParser.lastRawCount} '
        'kept=${detections.length} '
        '${detections.take(8).map((d) => '${d.color}${d.value}').toList()}',
      );
      return detections;
    } on FormatException catch (error) {
      lastSummaries.add('${bytes.length}B parse-error: ${error.message}');
      throw VisionException(error.message);
    }
  }

  /// Posts one image to the workflow endpoint with timeout and retries.
  Future<dynamic> runWorkflow(Uint8List imageBytes) async {
    if (!isConfigured) {
      throw const VisionException(
        'Tile recognition is not set up in this build (missing Roboflow API key). '
        'Enter the tiles manually, or rebuild with --dart-define=ROBOFLOW_API_KEY=your_key.',
      );
    }
    if (imageBytes.isEmpty) {
      throw const VisionException('The selected photo could not be read.');
    }
    if (imageBytes.length < minVisionPhotoBytes) {
      throw const VisionException(
        'That photo was empty or too small to recognize. Try again, or pick from the gallery.',
      );
    }

    final payload = jsonEncode({
      'api_key': _apiKey,
      'inputs': {
        RoboflowConfig.imageInputName: {
          'type': 'base64',
          'value': base64Encode(imageBytes),
        },
      },
    });

    http.Response? lastResponse;
    Object? lastError;
    var url = _workflowUrl;
    var triedFallback = false;

    for (var attempt = 0; attempt < maxAttempts; attempt++) {
      try {
        final response = await _client
            .post(
              Uri.parse(url),
              headers: const {'Content-Type': 'application/json'},
              body: payload,
            )
            .timeout(timeout);
        lastResponse = response;

        if (response.statusCode == 404 &&
            !triedFallback &&
            _fallbackUrl.isNotEmpty &&
            _fallbackUrl != url) {
          url = _fallbackUrl;
          triedFallback = true;
          continue;
        }

        if (_shouldRetryStatus(response.statusCode) &&
            attempt < maxAttempts - 1) {
          await Future<void>.delayed(_backoff(attempt));
          continue;
        }

        return _decodeSuccess(response);
      } on VisionException {
        rethrow;
      } on TimeoutException catch (error) {
        lastError = error;
        if (attempt >= maxAttempts - 1) break;
        await Future<void>.delayed(_backoff(attempt));
      } on http.ClientException catch (error) {
        lastError = error;
        if (attempt >= maxAttempts - 1) break;
        await Future<void>.delayed(_backoff(attempt));
      } catch (error) {
        lastError = error;
        if (attempt >= maxAttempts - 1) break;
        await Future<void>.delayed(_backoff(attempt));
      }
    }

    if (lastResponse != null) {
      throw _exceptionForResponse(lastResponse);
    }
    if (lastError is TimeoutException) {
      throw const VisionException(
        'Roboflow timed out. Try a smaller photo or try again.',
      );
    }
    throw VisionException(
      kIsWeb
          ? 'Could not reach the vision backend. Start the Python API on port 8000 and try again.'
          : 'Could not reach Roboflow. Check your internet connection.',
    );
  }

  dynamic _decodeSuccess(http.Response response) {
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw _exceptionForResponse(response);
    }
    if (_isHtmlResponse(response)) {
      throw const VisionException(
        'Vision proxy is not available on this host. Deploy the backend and try again.',
        statusCode: 200,
      );
    }
    if (response.body.isEmpty) {
      throw const VisionException('Roboflow returned an empty response.');
    }
    try {
      return jsonDecode(response.body);
    } on FormatException {
      throw const VisionException(
        'Roboflow returned a response that is not valid JSON.',
      );
    }
  }

  VisionException _exceptionForResponse(http.Response response) {
    final detail = _shortErrorMessage(response.body);
    switch (response.statusCode) {
      case 401:
      case 403:
        return VisionException(
          'Roboflow rejected the API key. Check ROBOFLOW_API_KEY.',
          statusCode: response.statusCode,
        );
      case 404:
        return VisionException(
          'Roboflow workflow not found (${RoboflowConfig.workspace}/'
          '${RoboflowConfig.workflowId}).',
          statusCode: 404,
        );
      case 413:
        return const VisionException(
          'Photo is too large for Roboflow. Try a closer shot or a smaller image.',
          statusCode: 413,
        );
      case 429:
        return const VisionException(
          'Too many Roboflow requests. Try again in a moment.',
          statusCode: 429,
        );
      default:
        if (response.statusCode == 200 && _isHtmlResponse(response)) {
          return const VisionException(
            'Vision proxy is not available on this host. Deploy the backend and try again.',
            statusCode: 200,
          );
        }
        return VisionException(
          detail ?? 'Roboflow request failed (${response.statusCode}).',
          statusCode: response.statusCode,
        );
    }
  }

  static bool _isHtmlResponse(http.Response response) {
    final contentType = response.headers['content-type'] ?? '';
    if (contentType.contains('text/html')) return true;
    final trimmed = response.body.trimLeft().toLowerCase();
    return trimmed.startsWith('<!doctype html') || trimmed.startsWith('<html');
  }

  static bool _shouldRetryStatus(int statusCode) {
    return statusCode == 408 || statusCode == 429 || statusCode >= 500;
  }

  static Duration _backoff(int attempt) {
    return Duration(milliseconds: 400 * (1 << attempt));
  }

  static String? _shortErrorMessage(String body) {
    if (body.isEmpty || body.length > 8000) return null;
    try {
      final decoded = jsonDecode(body);
      if (decoded is! Map) return null;
      for (final key in ['error', 'message', 'detail']) {
        final value = decoded[key];
        if (value is String && value.isNotEmpty && value.length < 400) {
          return value;
        }
      }
    } catch (_) {}
    return null;
  }
}
