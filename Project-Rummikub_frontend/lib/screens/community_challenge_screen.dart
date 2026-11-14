import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/api_service.dart';
import 'correction_screen.dart';

/// Lists the positions the backend published from solved boards. Picking one
/// opens the correction screen as a puzzle; the submit there is scored against
/// the algorithm.
class CommunityChallengeScreen extends StatefulWidget {
  const CommunityChallengeScreen({super.key, this.api});

  /// Test seam; defaults to the game provider's client.
  final ApiService? api;

  @override
  State<CommunityChallengeScreen> createState() =>
      _CommunityChallengeScreenState();
}

class _CommunityChallengeScreenState extends State<CommunityChallengeScreen> {
  late final ApiService _api;
  late Future<List<CommunityChallenge>> _future;

  @override
  void initState() {
    super.initState();
    _api = widget.api ?? context.read<GameProvider>().api;
    _future = _api.listChallenges();
  }

  Future<void> _reload() async {
    final future = _api.listChallenges();
    setState(() => _future = future);
    await future.catchError((Object _) => <CommunityChallenge>[]);
  }

  void _play(CommunityChallenge challenge, int number) {
    context.read<GameProvider>().beginArena(challenge.initialState);
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CorrectionScreen(
          arenaChallenge: ArenaChallenge(
            id: challenge.id,
            title: 'Challenge #$number',
            initialState: challenge.initialState,
            algorithmScore: challenge.aiScore,
            source: CorrectionScreen.communitySource,
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Community challenges')),
      body: FutureBuilder<List<CommunityChallenge>>(
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
                      'Could not load challenges.\n${snapshot.error}',
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 16),
                    FilledButton(onPressed: _reload, child: const Text('Retry')),
                  ],
                ),
              ),
            );
          }

          final challenges = snapshot.data ?? [];
          if (challenges.isEmpty) {
            return const Center(
              child: Padding(
                padding: EdgeInsets.all(24),
                child: Text(
                  'No community challenges yet.\n'
                  'Solve a board from Play to publish the position here.',
                  textAlign: TextAlign.center,
                ),
              ),
            );
          }

          return RefreshIndicator(
            onRefresh: _reload,
            child: ListView.separated(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(16),
              itemCount: challenges.length,
              separatorBuilder: (_, __) => const SizedBox(height: 8),
              itemBuilder: (context, index) {
                final challenge = challenges[index];
                return Card(
                  child: ListTile(
                    key: ValueKey('challenge-${challenge.id}'),
                    title: Text('Challenge #${index + 1}'),
                    subtitle: Text(
                      'Algorithm placed ${challenge.aiScore}  ·  '
                      '${challenge.initialState.rack.length} on the rack',
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _play(challenge, index + 1),
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
