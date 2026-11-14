import 'dart:async';
import 'dart:convert';

import 'package:http/http.dart' as http;

import '../config/solver_config.dart';
import '../models/json_map.dart';
import '../models/models.dart';

/// User-visible failure from the solver backend.
class SolverException implements Exception {
  const SolverException(this.message);

  final String message;

  @override
  String toString() => message;
}

/// A turn recorded by `/game/turn`.
class GameTurn {
  const GameTurn({required this.sessionId, required this.turnIndex});

  final String sessionId;
  final int turnIndex;
}

/// HTTP client for the Python solver backend.
class ApiService {
  ApiService({
    http.Client? client,
    String? solveUrl,
    this.timeout = const Duration(seconds: 30),
  })  : _client = client ?? http.Client(),
        _solveUrl = solveUrl ?? SolverConfig.solveUrl;

  final http.Client _client;
  final String _solveUrl;

  /// The solver itself stops after 5 seconds; the rest covers the network and
  /// a cold Cloud Run start.
  final Duration timeout;

  static const _genericError =
      'The solver could not solve this board. Please try again.';

  /// Posts [currentState] to `/solve`.
  ///
  /// Contract (frozen): response has `original_state`, `optimal_state`
  /// (`board`, `tiles_used_from_rack`, `remaining_rack`), `is_valid`,
  /// `execution_time_ms`.
  ///
  /// With [sessionId] the backend also publishes the position as a community
  /// challenge for that game ([turnIndex] is the turn recorded for it).
  Future<SolveResult> solveBoard(
    GameState currentState, {
    String? sessionId,
    int? turnIndex,
  }) async {
    final http.Response response;
    try {
      final solveUri = Uri.parse(_solveUrl);
      response = await _client
          .post(
            sessionId == null
                ? solveUri
                : solveUri.replace(queryParameters: {
                    'session_id': sessionId,
                    if (turnIndex != null) 'turn_index': '$turnIndex',
                  }),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode(currentState.toJson()),
          )
          .timeout(timeout);
    } on TimeoutException {
      throw SolverException(
        'The solver did not answer within ${timeout.inSeconds} seconds. '
        'Check that the backend is running and try again.',
      );
    } catch (_) {
      throw SolverException(
        'Could not reach the solver at ${Uri.parse(_solveUrl).host}. '
        'Check that the backend is running and try again.',
      );
    }

    switch (response.statusCode) {
      case 200:
        try {
          return _parseSolveResult(response.body, currentState);
        } catch (_) {
          throw const SolverException(
            'The solver sent an answer the app cannot read. Please try again.',
          );
        }
      case 400:
        throw SolverException(_detailFromBody(response.body));
      case 422:
        throw const SolverException(
          'The board could not be sent to the solver: some tile data is invalid.',
        );
      default:
        throw const SolverException(_genericError);
    }
  }

  /// `/game/turn` and `/game/end` live next to `/solve`.
  Uri _endpoint(String name) {
    final solve = Uri.parse(_solveUrl);
    final path = solve.path.endsWith('/solve')
        ? solve.path.substring(0, solve.path.length - '/solve'.length)
        : solve.path;
    return solve.replace(path: '$path/$name', query: '');
  }

  /// Records a turn via `POST /game/turn`. Without [sessionId] the backend
  /// opens a new game for [userId]. Returns the game's id and the turn index.
  Future<GameTurn> recordTurn({
    required String userId,
    required GameState state,
    String? sessionId,
    Map<String, dynamic>? moveApplied,
  }) async {
    final http.Response response;
    try {
      response = await _client
          .post(
            _endpoint('game/turn'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode({
              'user_id': userId,
              'session_id': sessionId,
              'board': [for (final set in state.board) set.toJson()],
              'rack': [for (final tile in state.rack) tile.toJson()],
              'move_applied': moveApplied,
            }),
          )
          .timeout(timeout);
    } catch (_) {
      throw const SolverException('Could not record the turn.');
    }
    if (response.statusCode != 200) {
      throw SolverException(_detailFromBody(response.body));
    }
    final decoded = asStringKeyedMap(jsonDecode(response.body));
    return GameTurn(
      sessionId: decoded['session_id'] as String,
      turnIndex: (decoded['turn_index'] as num).toInt(),
    );
  }

  /// Marks the game [sessionId] as packed via `POST /game/end`.
  Future<void> endGame(String sessionId) async {
    final http.Response response;
    try {
      response = await _client
          .post(
            _endpoint('game/end'),
            headers: const {'Content-Type': 'application/json'},
            body: jsonEncode({'session_id': sessionId}),
          )
          .timeout(timeout);
    } catch (_) {
      throw const SolverException('Could not end the game.');
    }
    if (response.statusCode != 200) {
      throw SolverException(_detailFromBody(response.body));
    }
  }

  /// Community positions published by `/solve`, via `GET /challenge/list`.
  Future<List<CommunityChallenge>> listChallenges() async {
    final body = await _send(
      () => _client.get(_endpoint('challenge/list')),
      'Could not load the challenges.',
    );
    return asMapList(jsonDecode(body)).map(CommunityChallenge.fromJson).toList();
  }

  /// Submits [board] (the board after the player's turn) for challenge
  /// [challengeId] via `POST /challenge/{id}/submit`.
  Future<ChallengeResult> submitChallenge({
    required String challengeId,
    required String userId,
    required GameState board,
  }) async {
    final body = await _send(
      () => _client.post(
        _endpoint('challenge/$challengeId/submit'),
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode({
          'user_id': userId,
          'board': [for (final set in board.board) set.toJson()],
        }),
      ),
      'Could not submit the challenge.',
    );
    return ChallengeResult.fromJson(asStringKeyedMap(jsonDecode(body)));
  }

  /// Player ranking via `GET /leaderboard`, best first.
  Future<List<RankedPlayer>> getLeaderboard() async {
    final body = await _send(
      () => _client.get(_endpoint('leaderboard')),
      'Could not load the leaderboard.',
    );
    return asMapList(jsonDecode(body)).map(RankedPlayer.fromJson).toList();
  }

  /// Runs [request]; returns the body of a 200, else throws [SolverException].
  Future<String> _send(
    Future<http.Response> Function() request,
    String failureMessage,
  ) async {
    final http.Response response;
    try {
      response = await request().timeout(timeout);
    } catch (_) {
      throw SolverException(failureMessage);
    }
    if (response.statusCode != 200) {
      throw SolverException(_detailFromBody(response.body));
    }
    return response.body;
  }

  SolveResult _parseSolveResult(String body, GameState currentState) {
    final decoded = asStringKeyedMap(jsonDecode(body));

    final originalJson = decoded['original_state'];
    final original = originalJson is Map
        ? GameState.fromJson(asStringKeyedMap(originalJson))
        : currentState;

    final optimal = decoded['optimal_state'];
    if (optimal is! Map) {
      throw const FormatException('optimal_state is missing');
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
      tilesUsedFromRack:
          (optimalMap['tiles_used_from_rack'] as num?)?.toInt() ?? 0,
      isValid: decoded['is_valid'] == true,
      executionTimeMs: (decoded['execution_time_ms'] as num?)?.toInt() ?? 0,
      solutionSteps: _parseSolutionSteps(optimalMap['solution_steps']),
    );
  }

  /// Steps only power the optional walkthrough, so a missing or malformed
  /// list yields no steps instead of failing the whole solve.
  List<SolutionStep> _parseSolutionSteps(Object? raw) {
    if (raw is! List) return const [];
    try {
      return [
        for (final step in raw) SolutionStep.fromJson(asStringKeyedMap(step)),
      ];
    } catch (_) {
      return const [];
    }
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
