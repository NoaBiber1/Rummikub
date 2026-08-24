import 'package:flutter/foundation.dart';

/// Solver URL helpers.
///
/// Local debug (Chrome / desktop / iOS simulator): `http://127.0.0.1:8000/solve`.
/// Android emulator: `http://10.0.2.2:8000/solve`.
/// Hosted web (Firebase Hosting, iPhone, checkers): same-origin `/solve`.
///
/// Override with `--dart-define=SOLVER_URL=https://example.com/solve`
/// or `--dart-define=SOLVER_HOST=192.168.1.10`.
class SolverConfig {
  static const urlOverride = String.fromEnvironment('SOLVER_URL');
  static const hostOverride = String.fromEnvironment('SOLVER_HOST');
  static const port = 8000;

  static String get host {
    if (hostOverride.isNotEmpty) return hostOverride;
    if (!kIsWeb && defaultTargetPlatform == TargetPlatform.android) {
      return '10.0.2.2';
    }
    return '127.0.0.1';
  }

  static String get solveUrl {
    if (urlOverride.isNotEmpty) {
      final trimmed = urlOverride.replaceAll(RegExp(r'/$'), '');
      return trimmed.endsWith('/solve') ? trimmed : '$trimmed/solve';
    }

    if (kIsWeb && _isHostedWeb) {
      return Uri.base.resolve('solve').toString();
    }

    return 'http://$host:$port/solve';
  }

  static bool get _isHostedWeb {
    final host = Uri.base.host;
    return host != 'localhost' &&
        host != '127.0.0.1' &&
        host != '0.0.0.0' &&
        !host.endsWith('.local');
  }
}
