import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/firebase_service.dart';
import 'correction_screen.dart';

/// Lists community puzzles from `global_arena`.
class ArenaListScreen extends StatefulWidget {
  const ArenaListScreen({super.key});

  @override
  State<ArenaListScreen> createState() => _ArenaListScreenState();
}

class _ArenaListScreenState extends State<ArenaListScreen> {
  late Future<List<ArenaChallenge>> _future;

  @override
  void initState() {
    super.initState();
    _future = FirebaseService().getArenaChallenges();
  }

  Future<void> _reload() async {
    final future = FirebaseService().getArenaChallenges();
    setState(() => _future = future);
    await future;
  }

  void _play(ArenaChallenge challenge) {
    context.read<GameProvider>().beginArena(challenge.initialState);
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (_) => CorrectionScreen(arenaChallenge: challenge),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Community challenges'),
      ),
      body: FutureBuilder<List<ArenaChallenge>>(
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
                  'Solve a board from Play to publish the position and the algorithm solution here.',
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
                    title: Text(challenge.title),
                    subtitle: Text(
                      'Algorithm dropped ${challenge.algorithmScore}  ·  '
                      '${challenge.initialState.rackSize} on the rack',
                    ),
                    trailing: const Icon(Icons.chevron_right),
                    onTap: () => _play(challenge),
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
