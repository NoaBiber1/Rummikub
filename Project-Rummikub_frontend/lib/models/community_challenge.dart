import 'game_state.dart';

/// A position published by the backend (`GET /challenge/list`).
class CommunityChallenge {
  const CommunityChallenge({
    required this.id,
    required this.initialState,
    required this.aiScore,
  });

  final String id;
  final GameState initialState;

  /// Rack tiles the solver placed on this position.
  final int aiScore;

  factory CommunityChallenge.fromJson(Map<String, dynamic> json) {
    return CommunityChallenge(
      id: json['id'] as String,
      initialState: GameState.fromJson(json),
      aiScore: (json['ai_score'] as num).toInt(),
    );
  }
}

/// The backend's verdict on a submitted turn (`POST /challenge/{id}/submit`).
class ChallengeResult {
  const ChallengeResult({
    required this.humanPlaced,
    required this.aiPlaced,
    required this.score,
  });

  final int humanPlaced;
  final int aiPlaced;

  /// `humanPlaced - aiPlaced`: positive means the player beat the algorithm.
  final int score;

  factory ChallengeResult.fromJson(Map<String, dynamic> json) {
    return ChallengeResult(
      humanPlaced: (json['human_placed'] as num).toInt(),
      aiPlaced: (json['ai_placed'] as num).toInt(),
      score: (json['score'] as num).toInt(),
    );
  }

  /// Mirrors the backend rule so the app can show a score before it answers.
  factory ChallengeResult.compare({
    required int humanPlaced,
    required int aiPlaced,
  }) {
    return ChallengeResult(
      humanPlaced: humanPlaced,
      aiPlaced: aiPlaced,
      score: humanPlaced - aiPlaced,
    );
  }
}

/// One row of `GET /leaderboard`.
class RankedPlayer {
  const RankedPlayer({
    required this.rank,
    required this.userId,
    required this.totalPoints,
    required this.challengesWon,
  });

  final int rank;
  final String userId;
  final int totalPoints;
  final int challengesWon;

  factory RankedPlayer.fromJson(Map<String, dynamic> json) {
    return RankedPlayer(
      rank: (json['rank'] as num).toInt(),
      userId: json['user_id'] as String,
      totalPoints: (json['total_points'] as num).toInt(),
      challengesWon: (json['challenges_won'] as num).toInt(),
    );
  }
}

/// Rack tiles of [initial] that sit on [board] — the number a challenge scores.
int rackTilesPlaced(GameState initial, GameState board) {
  final onBoard = {
    for (final set in board.board)
      for (final tile in set.tiles) tile.id,
  };
  return initial.rack.where((tile) => onBoard.contains(tile.id)).length;
}
