import 'package:flutter/material.dart';

import '../services/firebase_service.dart';
import 'capture_screen.dart';

/// Signed-in home: the Play screen. History, Leaderboard, Account and the
/// rest of the secondary actions open from its side drawer.
class HomeShell extends StatefulWidget {
  const HomeShell({super.key});

  @override
  State<HomeShell> createState() => _HomeShellState();
}

class _HomeShellState extends State<HomeShell> {
  @override
  void initState() {
    super.initState();
    FirebaseService().ensureCurrentUser();
  }

  @override
  Widget build(BuildContext context) => const CaptureScreen();
}
