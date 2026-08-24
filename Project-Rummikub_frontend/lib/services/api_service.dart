import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/solver_config.dart';
import '../models/json_map.dart';
import '../models/models.dart';

/// HTTP client for the Python solver backend.
class ApiService {
  static const _genericError = 'Failed to solve the board. Please try again.';

  /// Posts [currentState] to `/solve`.
  ///
  /// Contract (frozen): response has `original_state`, `optimal_state`
  /// (`board`, `tiles_used_from_rack`, `remaining_rack`), `is_valid`,
  /// `execution_time_ms`.
  Future<SolveResult> solveBoard(GameState currentState) async {
    final http.Response response;
    try {
      response = await http.post(
        Uri.parse(SolverConfig.solveUrl),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode(currentState.toJson()),
      );
    } catch (_) {
      throw Exception(_genericError);
    }

    if (response.statusCode == 200) {
      try {
        return _parseSolveResult(response.body, currentState);
      } catch (_) {
        throw Exception(_genericError);
      }
    }

    if (response.statusCode == 400) {
      throw Exception(_detailFromBody(response.body));
    }

    throw Exception(_genericError);
  }

  SolveResult _parseSolveResult(String body, GameState currentState) {
    final decoded = asStringKeyedMap(jsonDecode(body));

    final originalJson = decoded['original_state'];
    final original = originalJson is Map
        ? GameState.fromJson(asStringKeyedMap(originalJson))
        : currentState;

    final optimal = decoded['optimal_state'];
    if (optimal is! Map) {
      throw Exception(_genericError);
    }
    final optimalMap = asStringKeyedMap(optimal);

    final remaining = optimalMap['rack'] ?? optimalMap['remaining_rack'];
    final optimalState = GameState.fromJson({
      'board': optimalMap['board'],
      'rack': remaining,
    });

    return SolveResult(
      originalState: original,
      optimalState: optimalState,
      tilesUsedFromRack: (optimalMap['tiles_used_from_rack'] as num?)?.toInt() ??
          0,
      isValid: decoded['is_valid'] == true,
      executionTimeMs: (decoded['execution_time_ms'] as num?)?.toInt() ?? 0,
    );
  }

  String _detailFromBody(String body) {
    try {
      final decoded = jsonDecode(body);
      if (decoded is Map<String, dynamic> && decoded['detail'] != null) {
        final detail = decoded['detail'];
        if (detail is String && detail.isNotEmpty) {
          return detail;
        }
        return detail.toString();
      }
    } catch (_) {
      // Fall through to the generic message.
    }
    return _genericError;
  }
}
