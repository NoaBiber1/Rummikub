import 'package:flutter/material.dart';

import '../services/firebase_service.dart';
import 'result_screen.dart';

/// Lists recently saved games from Firestore for the signed-in user.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  late Future<List<GameHistoryEntry>> _gamesFuture;

  @override
  void initState() {
    super.initState();
    _gamesFuture = FirebaseService().getRecentGames();
  }

  Future<void> _reload() async {
    final future = FirebaseService().getRecentGames();
    setState(() => _gamesFuture = future);
    await future;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('History'),
      ),
      body: FutureBuilder<List<GameHistoryEntry>>(
        future: _gamesFuture,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return _Message(
              text: 'Could not load history.\n${snapshot.error}',
              onRetry: _reload,
            );
          }

          final games = snapshot.data ?? [];
          if (games.isEmpty) {
            return RefreshIndicator(
              onRefresh: _reload,
              child: ListView(
                physics: const AlwaysScrollableScrollPhysics(),
                children: const [
                  SizedBox(height: 120),
                  _Message(
                    text:
                        'No games yet.\nPlay a game from the Play tab, then pull to refresh.',
                  ),
                ],
              ),
            );
          }

          return RefreshIndicator(
            onRefresh: _reload,
            child: ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              itemCount: games.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (context, index) {
                final game = games[index];
                final isWin = game.afterState.isRackEmpty && game.tilesDropped > 0;
                return Card(
                  child: ListTile(
                    leading: Icon(
                      isWin ? Icons.emoji_events : Icons.sports_esports,
                      color: isWin
                          ? Theme.of(context).colorScheme.primary
                          : Theme.of(context).colorScheme.outline,
                    ),
                    title: Text(_historyTitle(game, isWin)),
                    subtitle: Text(
                      '${_formatTimestamp(game.timestamp)}  ·  '
                      '${game.tilesDropped} tiles dropped',
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () {
                      Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (_) => ResultScreen(
                            beforeState: game.beforeState,
                            afterState: game.afterState,
                            tilesDropped: game.tilesDropped,
                            algorithmScore: game.algorithmScore,
                            challengeId: game.challengeId,
                            readOnly: true,
                          ),
                        ),
                      );
                    },
                  ),
                );
              },
            ),
          );
        },
      ),
    );
  }

  String _historyTitle(GameHistoryEntry game, bool isWin) {
    if (game.source == 'arena') {
      final target = game.algorithmScore;
      if (target != null) {
        if (game.tilesDropped > target) return 'Beat the algorithm';
        if (game.tilesDropped == target) return 'Matched the algorithm';
        return 'Challenge attempt';
      }
      return 'Challenge';
    }
    return isWin ? 'Win' : 'Solved';
  }

  String _formatTimestamp(DateTime? value) {
    if (value == null) return 'Just now';
    final date = value.toLocal();
    final month = date.month.toString().padLeft(2, '0');
    final day = date.day.toString().padLeft(2, '0');
    final hour = date.hour.toString().padLeft(2, '0');
    final minute = date.minute.toString().padLeft(2, '0');
    return '${date.year}-$month-$day $hour:$minute';
  }
}

class _Message extends StatelessWidget {
  const _Message({required this.text, this.onRetry});

  final String text;
  final VoidCallback? onRetry;

  @override
  Widget build(BuildContext context) {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(text, textAlign: TextAlign.center),
            if (onRetry != null) ...[
              const SizedBox(height: 16),
              FilledButton(onPressed: onRetry, child: const Text('Retry')),
            ],
          ],
        ),
      ),
    );
  }
}
