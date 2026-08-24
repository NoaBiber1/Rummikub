import 'package:flutter/material.dart';

import '../services/firebase_service.dart';
import 'account_screen.dart';
import 'capture_screen.dart';
import 'history_screen.dart';
import 'leaderboard_screen.dart';

/// Signed-in shell with bottom navigation: Play, History, Leaderboard, Account.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  int _index = 0;
  int _historyGen = 0;
  int _leaderboardGen = 0;

  @override
  void initState() {
    super.initState();
    FirebaseService().ensureCurrentUser();
  }

  void _onDestinationSelected(int index) {
    setState(() {
      if (index == 1) _historyGen++;
      if (index == 2) _leaderboardGen++;
      _index = index;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _index,
        children: [
          const CaptureScreen(),
          HistoryScreen(key: ValueKey(_historyGen)),
          LeaderboardScreen(key: ValueKey(_leaderboardGen)),
          const AccountScreen(),
        ],
      ),
      bottomNavigationBar: NavigationBar(
        selectedIndex: _index,
        onDestinationSelected: _onDestinationSelected,
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.play_arrow_outlined),
            selectedIcon: Icon(Icons.play_arrow),
            label: 'Play',
          ),
          NavigationDestination(
            icon: Icon(Icons.history_outlined),
            selectedIcon: Icon(Icons.history),
            label: 'History',
          ),
          NavigationDestination(
            icon: Icon(Icons.leaderboard_outlined),
            selectedIcon: Icon(Icons.leaderboard),
            label: 'Leaderboard',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outline),
            selectedIcon: Icon(Icons.person),
            label: 'Account',
          ),
        ],
      ),
    );
  }
}
