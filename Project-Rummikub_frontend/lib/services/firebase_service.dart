import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../logic/game_state_fingerprint.dart';
import '../models/json_map.dart';
import '../models/models.dart';

class GameHistoryEntry {
  const GameHistoryEntry({
    required this.id,
    required this.userId,
    required this.tilesDropped,
    required this.beforeState,
    required this.afterState,
    this.timestamp,
    this.source = 'solver',
    this.challengeId,
    this.algorithmScore,
  });

  final String id;
  final String userId;
  final int tilesDropped;
  final GameState beforeState;
  final GameState afterState;
  final DateTime? timestamp;

  /// `solver` = system move, `arena` = human challenge attempt.
  final String source;
  final String? challengeId;
  final int? algorithmScore;
}

class LeaderboardEntry {
  const LeaderboardEntry({
    required this.uid,
    required this.displayName,
    required this.totalTilesDropped,
  });

  final String uid;
  final String displayName;
  final int totalTilesDropped;
}

class UserProfile {
  const UserProfile({
    required this.uid,
    required this.displayName,
    required this.totalTilesDropped,
  });

  final String uid;
  final String displayName;
  final int totalTilesDropped;
}

/// Firestore client aligned with architecture collections.
class FirebaseService {
  FirebaseService({FirebaseFirestore? firestore, FirebaseAuth? auth})
      : _firestore = firestore ?? FirebaseFirestore.instance,
        _auth = auth ?? FirebaseAuth.instance;

  final FirebaseFirestore _firestore;
  final FirebaseAuth _auth;

  User get _requireUser {
    final user = _auth.currentUser;
    if (user == null) {
      throw Exception('You must be signed in.');
    }
    return user;
  }

  Future<void> ensureCurrentUser() async {
    final user = _auth.currentUser;
    if (user == null) return;

    final ref = _firestore.collection('users').doc(user.uid);
    final snapshot = await ref.get();
    if (snapshot.exists) return;

    await ref.set({
      'display_name': _defaultDisplayName(user),
      'total_tiles_dropped': 0,
    });
  }

  String _defaultDisplayName(User user) {
    final named = user.displayName;
    if (named != null && named.trim().isNotEmpty) return named.trim();
    final email = user.email;
    if (email != null && email.contains('@')) {
      return email.split('@').first;
    }
    return 'Player';
  }

  Future<UserProfile?> getCurrentUserProfile() async {
    final user = _auth.currentUser;
    if (user == null) return null;
    await ensureCurrentUser();
    final snapshot = await _firestore.collection('users').doc(user.uid).get();
    final data = snapshot.data() ?? {};
    return UserProfile(
      uid: user.uid,
      displayName: (data['display_name'] as String?) ?? _defaultDisplayName(user),
      totalTilesDropped: (data['total_tiles_dropped'] as num?)?.toInt() ?? 0,
    );
  }

  Future<void> updateDisplayName(String displayName) async {
    final user = _requireUser;
    final trimmed = displayName.trim();
    if (trimmed.isEmpty) {
      throw Exception('Display name cannot be empty.');
    }
    await _firestore.collection('users').doc(user.uid).set(
      {'display_name': trimmed},
      SetOptions(merge: true),
    );
    await user.updateDisplayName(trimmed);
  }

  /// Solver output: personal history, global arena puzzle, leaderboard.
  Future<void> saveSolverSolution({
    required GameState beforeState,
    required GameState afterState,
    required int tilesDropped,
  }) async {
    final user = _requireUser;
    await ensureCurrentUser();

    await _writeHistory(
      userId: user.uid,
      beforeState: beforeState,
      afterState: afterState,
      tilesDropped: tilesDropped,
      source: 'solver',
    );
    await _publishSolverChallenge(
      beforeState: beforeState,
      afterState: afterState,
      tilesDropped: tilesDropped,
    );

    if (tilesDropped > 0) {
      await _firestore.collection('users').doc(user.uid).set(
        {'total_tiles_dropped': FieldValue.increment(tilesDropped)},
        SetOptions(merge: true),
      );
    }
  }

  /// Human arena attempt: history + `arena_attempts`. Does not publish a puzzle
  /// or count toward the solver leaderboard.
  Future<void> saveArenaAttempt({
    required String challengeId,
    required GameState beforeState,
    required GameState afterState,
    required int tilesDropped,
    required int algorithmScore,
  }) async {
    final user = _requireUser;
    await ensureCurrentUser();

    await _writeHistory(
      userId: user.uid,
      beforeState: beforeState,
      afterState: afterState,
      tilesDropped: tilesDropped,
      source: 'arena',
      challengeId: challengeId,
      algorithmScore: algorithmScore,
    );

    await _firestore.collection('arena_attempts').add({
      'challenge_id': challengeId,
      'user_id': user.uid,
      'timestamp': FieldValue.serverTimestamp(),
      'tiles_dropped': tilesDropped,
      'algorithm_score': algorithmScore,
      'beat_algorithm': tilesDropped > algorithmScore,
      'before_state': beforeState.toJson(),
      'after_state': afterState.toJson(),
    });
  }

  Future<void> _writeHistory({
    required String userId,
    required GameState beforeState,
    required GameState afterState,
    required int tilesDropped,
    required String source,
    String? challengeId,
    int? algorithmScore,
  }) async {
    await _firestore.collection('games_history').add({
      'user_id': userId,
      'timestamp': FieldValue.serverTimestamp(),
      'tiles_dropped': tilesDropped,
      'before_state': beforeState.toJson(),
      'after_state': afterState.toJson(),
      'source': source,
      if (challengeId != null) 'challenge_id': challengeId,
      if (algorithmScore != null) 'algorithm_score': algorithmScore,
    });
  }

  Future<void> _publishSolverChallenge({
    required GameState beforeState,
    required GameState afterState,
    required int tilesDropped,
  }) async {
    final id = GameStateFingerprint.challengeId(beforeState);
    final ref = _firestore.collection('global_arena').doc(id);
    final existing = await ref.get();
    final data = existing.data();
    final previousScore = (data?['algorithm_score'] as num?)?.toInt() ?? -1;
    final keepPrevious = existing.exists && previousScore > tilesDropped;
    final previousAfter = data?['algorithm_after_state'];
    final title = keepPrevious
        ? (data?['title'] as String? ??
            _solverChallengeTitle(beforeState, previousScore))
        : _solverChallengeTitle(beforeState, tilesDropped);

    await ref.set(
      {
        'title': title,
        'initial_state': beforeState.toJson(),
        'algorithm_score': keepPrevious ? previousScore : tilesDropped,
        'algorithm_after_state': keepPrevious && previousAfter != null
            ? previousAfter
            : afterState.toJson(),
        'source': 'solver',
        'fingerprint': GameStateFingerprint.canonical(beforeState),
        'updated_at': FieldValue.serverTimestamp(),
        if (!existing.exists) 'created_at': FieldValue.serverTimestamp(),
      },
      SetOptions(merge: true),
    );
  }

  String _solverChallengeTitle(GameState beforeState, int tilesDropped) {
    return 'Solver: ${beforeState.numberOfSets} sets, '
        '${beforeState.rackSize} rack → $tilesDropped';
  }

  /// Stores a human correction of vision output for later model training.
  /// No identifying user fields (academic requirement).
  Future<void> saveVisionCorrection({
    required GameState visionOutput,
    required GameState userCorrectedOutput,
  }) async {
    await _firestore.collection('vision_corrections').add({
      'vision_output': visionOutput.toJson(),
      'user_corrected_output': userCorrectedOutput.toJson(),
      'timestamp': FieldValue.serverTimestamp(),
    });
  }

  Future<List<LeaderboardEntry>> getLeaderboard() async {
    final snapshot = await _firestore
        .collection('users')
        .orderBy('total_tiles_dropped', descending: true)
        .limit(10)
        .get();

    return snapshot.docs.map((doc) {
      final data = doc.data();
      return LeaderboardEntry(
        uid: doc.id,
        displayName: (data['display_name'] as String?) ?? doc.id,
        totalTilesDropped: (data['total_tiles_dropped'] as num?)?.toInt() ?? 0,
      );
    }).toList();
  }

  Future<List<GameHistoryEntry>> getRecentGames({int limit = 20}) async {
    final user = _requireUser;
    final snapshot = await _firestore
        .collection('games_history')
        .where('user_id', isEqualTo: user.uid)
        .get();

    final games = snapshot.docs.map(_historyFromDoc).toList();
    games.sort((a, b) {
      final aTime = a.timestamp ?? DateTime.fromMillisecondsSinceEpoch(0);
      final bTime = b.timestamp ?? DateTime.fromMillisecondsSinceEpoch(0);
      return bTime.compareTo(aTime);
    });
    return games.take(limit).toList();
  }

  GameHistoryEntry _historyFromDoc(QueryDocumentSnapshot<Map<String, dynamic>> doc) {
    final data = doc.data();
    final timestamp = data['timestamp'];
    return GameHistoryEntry(
      id: doc.id,
      userId: data['user_id'] as String? ?? '',
      tilesDropped: (data['tiles_dropped'] as num?)?.toInt() ?? 0,
      beforeState: GameState.fromJson(asStringKeyedMap(data['before_state'])),
      afterState: GameState.fromJson(asStringKeyedMap(data['after_state'])),
      timestamp: timestamp is Timestamp ? timestamp.toDate() : null,
      source: data['source'] as String? ?? 'solver',
      challengeId: data['challenge_id'] as String?,
      algorithmScore: (data['algorithm_score'] as num?)?.toInt(),
    );
  }

  Future<List<ArenaChallenge>> getArenaChallenges() async {
    try {
      await _deleteNonSolverChallenges();
      final snapshot = await _firestore.collection('global_arena').get();
      final challenges = snapshot.docs
          .map((doc) => ArenaChallenge.fromFirestore(doc.id, doc.data()))
          .where(_isSolverChallenge)
          .toList();
      challenges.sort((a, b) {
        return b.algorithmScore.compareTo(a.algorithmScore);
      });
      return challenges;
    } catch (_) {
      return [];
    }
  }

  bool _isSolverChallenge(ArenaChallenge challenge) {
    return challenge.source == 'solver' &&
        challenge.algorithmAfterState != null;
  }

  /// Drops hand-seeded practice puzzles so the arena is only real solver games.
  Future<void> _deleteNonSolverChallenges() async {
    final existing = await _firestore.collection('global_arena').get();
    final batch = _firestore.batch();
    var deletes = 0;
    for (final doc in existing.docs) {
      final data = doc.data();
      final source = data['source'] as String? ?? 'seed';
      final hasSolution = data['algorithm_after_state'] is Map;
      if (source == 'solver' && hasSolution) continue;
      batch.delete(doc.reference);
      deletes++;
    }
    if (deletes > 0) {
      await batch.commit();
    }
  }
}
