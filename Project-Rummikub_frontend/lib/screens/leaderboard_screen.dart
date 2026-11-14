import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/api_service.dart';

/// Ranks players by the points they earned beating the algorithm in community
/// challenges.
class LeaderboardScreen extends StatefulWidget {
  const LeaderboardScreen({super.key, this.api});

  /// Test seam; defaults to the game provider's client.
  final ApiService? api;

  @override
  State<LeaderboardScreen> createState() => _LeaderboardScreenState();
}

class _LeaderboardScreenState extends State<LeaderboardScreen> {
  late final ApiService _api;
  late Future<List<RankedPlayer>> _future;

  @override
  void initState() {
    super.initState();
    _api = widget.api ?? context.read<GameProvider>().api;
    _future = _api.getLeaderboard();
  }

  Future<void> _reload() async {
    final future = _api.getLeaderboard();
    setState(() => _future = future);
    await future.catchError((Object _) => <RankedPlayer>[]);
  }

  /// Backend ids are Firebase uids; show a short form.
  static String _name(String userId) =>
      userId.length > 8 ? 'Player ${userId.substring(0, 6)}' : userId;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Leaderboard')),
      body: FutureBuilder<List<RankedPlayer>>(
        future: _future,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(
              child: Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Text(
                      'Could not load leaderboard.\n${snapshot.error}',
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    FilledButton(onPressed: _reload, child: const Text('Retry')),
                  ],
                ),
              ),
            );
          }

          final players = snapshot.data ?? [];
          if (players.isEmpty) {
            return const Center(
              child: Text('No challenges played yet.'),
            );
          }

          return RefreshIndicator(
            onRefresh: _reload,
            child: ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              itemCount: players.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (context, index) {
                final player = players[index];
                return Card(
                  child: ListTile(
                    key: ValueKey('rank-${player.rank}'),
                    leading: CircleAvatar(child: Text('${player.rank}')),
                    title: Text(_name(player.userId)),
                    subtitle: Text(
                      '${player.challengesWon} '
                      'challenge${player.challengesWon == 1 ? '' : 's'} won',
                    ),
                    trailing: Text(
                      '${player.totalPoints} pts',
                      style: Theme.of(context).textTheme.titleSmall,
                    ),
                  ),
                );
              },
            ),
          );
        },
      ),
    );
  }
}
