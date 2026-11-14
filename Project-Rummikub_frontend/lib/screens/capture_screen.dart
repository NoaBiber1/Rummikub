import 'dart:math' as math;

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../config/roboflow_config.dart';
import '../models/models.dart';
import '../providers/game_provider.dart';
import '../services/vision_image.dart';
import '../services/vision_service.dart';
import '../widgets/auth_brand.dart';
import '../widgets/brand_buttons.dart';
import '../widgets/hover_drawer.dart';
import '../widgets/photo_zones.dart';
import '../widgets/table_scene.dart';
import 'account_screen.dart';
import 'community_challenge_screen.dart';
import 'correction_screen.dart';
import 'history_screen.dart';
import 'leaderboard_screen.dart';

/// Turns picked bytes into the JPEG that is both previewed and recognized.
typedef PrepareVisionImage = Future<PreparedVisionImage> Function(
  Uint8List bytes,
);

/// Capture screen: table photo, then rack photo, then correction.
///
/// Laid out as a first-person tabletop: the board in perspective, a rack along
/// each far side (decoration), the player's rack in the foreground, and the
/// Recognize button under it. Hovering the board or the rack reveals Camera
/// and Gallery; a photo then rests on it as a small card with Replace and
/// Delete. Everything else lives in the hover/pin drawer at the top left.
///
/// Data flow, one source of truth per photo:
/// picker `XFile` -> bytes -> [prepareVisionJpeg] -> `_Photo.bytes`, which is
/// both the preview and exactly what `VisionService` receives -> `GameState`
/// -> `GameProvider.updateGameState(fromVision: true)`, once per successful
/// recognition -> `CorrectionScreen`, which renders the provider's state.
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({
    super.key,
    this.picker,
    this.visionService,
    this.prepareImage,
  });

  /// Test seams. Defaults: the platform picker, the Roboflow client, and
  /// [prepareVisionJpegInBackground].
  final ImagePicker? picker;
  final VisionService? visionService;
  final PrepareVisionImage? prepareImage;

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

enum _Slot { table, rack }

/// What the screen is doing. Anything but [idle] blocks every action, so a
/// double tap can never open two pickers, send two requests or push twice.
enum _Activity { idle, picking, preparing, recognizing, away }

/// One accepted photo. [bytes] is previewed and sent without re-encoding.
class _Photo {
  const _Photo({required this.name, required this.image});

  final String name;
  final PreparedVisionImage image;

  Uint8List get bytes => image.bytes;
}

class _CaptureScreenState extends State<CaptureScreen> {
  late final ImagePicker _picker = widget.picker ?? ImagePicker();
  late final VisionService _visionService =
      widget.visionService ?? VisionService();
  late final PrepareVisionImage _prepare =
      widget.prepareImage ?? prepareVisionJpegInBackground;
  late final bool _cameraSupported =
      _picker.supportsImageSource(ImageSource.camera);

  late final GameProvider _game;
  late int _seenEndedGames;

  _Photo? _table;
  _Photo? _rack;
  _Activity _activity = _Activity.idle;
  _Slot? _activeSlot;

  /// Last failure, shown until the next successful action.
  String? _error;

  /// Non-error guidance, e.g. after coming back from the correction screen.
  String? _notice;

  /// The missing-API-key banner floats over the scene: let it be put away.
  bool _keyWarningDismissed = false;

  bool get _busy => _activity != _Activity.idle;

  @override
  void initState() {
    super.initState();
    _game = context.read<GameProvider>();
    _seenEndedGames = _game.endedGames;
    _game.addListener(_onGameChanged);
  }

  @override
  void dispose() {
    _game.removeListener(_onGameChanged);
    super.dispose();
  }

  /// A game was ended (here, on the correction screen, or by signing out):
  /// back to fresh tile recognition.
  void _onGameChanged() {
    if (_game.endedGames == _seenEndedGames) return;
    _seenEndedGames = _game.endedGames;
    if (!mounted) return;
    setState(() {
      _table = null;
      _rack = null;
      _error = null;
      _notice = null;
    });
  }

  void _setPhoto(_Slot slot, _Photo? photo) {
    if (slot == _Slot.table) {
      _table = photo;
    } else {
      _rack = photo;
    }
  }

  static String _name(_Slot slot) => slot == _Slot.table ? 'table' : 'rack';

  Future<void> _pick(_Slot slot, ImageSource source) async {
    if (_busy) return;
    if (slot == _Slot.rack && _table == null) return;
    setState(() {
      _activity = _Activity.picking;
      _activeSlot = slot;
    });
    try {
      final XFile? file;
      try {
        // Flutter web's image_picker resizes through a canvas sized from
        // element.width (not naturalWidth). When that is 0, toBlob writes a
        // ~651-byte empty JPEG and Roboflow sees no tiles. So on web the
        // picker gets no size or quality options and prepareVisionJpeg does
        // the resizing instead.
        file = await _picker.pickImage(
          source: source,
          maxWidth: kIsWeb ? null : 1920,
          maxHeight: kIsWeb ? null : 1920,
          imageQuality: kIsWeb ? null : 85,
          preferredCameraDevice: CameraDevice.rear,
        );
      } on PlatformException catch (error) {
        _fail(_pickerMessage(error.code, source));
        return;
      } on StateError {
        // Desktop image_picker throws this for the camera without a delegate.
        _fail(_pickerMessage('no_available_camera', source));
        return;
      }
      // Cancelled: nothing about the screen changes.
      if (file == null) return;

      final raw = await file.readAsBytes();
      if (!mounted) return;
      if (raw.length < minVisionPhotoBytes) {
        _fail(
          'The ${_name(slot)} photo is empty or too small to recognize '
          '(${raw.length} bytes). Take it again, or pick another photo.',
        );
        return;
      }

      setState(() => _activity = _Activity.preparing);
      // Let the progress overlay paint before the decode starts (on web the
      // decode runs on the UI thread).
      await WidgetsBinding.instance.endOfFrame;
      final prepared = await _prepare(raw);
      if (!mounted) return;
      if (!prepared.decoded) {
        _fail(
          'The ${_name(slot)} photo is not a readable image. '
          'Use a JPEG or PNG photo.',
        );
        return;
      }
      final name = file.name;
      setState(() {
        _setPhoto(slot, _Photo(name: name, image: prepared));
        _error = null;
        _notice = null;
      });
    } catch (error) {
      debugPrint('Could not load the ${_name(slot)} photo: $error');
      _fail('The ${_name(slot)} photo could not be loaded. Try again.');
    } finally {
      if (mounted) {
        setState(() {
          _activity = _Activity.idle;
          _activeSlot = null;
        });
      }
    }
  }

  static String _pickerMessage(String code, ImageSource source) {
    switch (code) {
      case 'camera_access_denied':
        return 'Camera access is turned off for this app. Allow it in your '
            'device settings, or use Gallery instead.';
      case 'photo_access_denied':
        return 'Photo access is turned off for this app. Allow it in your '
            'device settings, or use Camera instead.';
      case 'no_available_camera':
        return 'No camera is available on this device. Use Gallery instead.';
      default:
        return source == ImageSource.camera
            ? 'The camera could not be opened. Try again, or use Gallery.'
            : 'The gallery could not be opened. Try again, or use Camera.';
    }
  }

  Future<void> _recognize() async {
    final table = _table;
    final rack = _rack;
    if (_busy || table == null || rack == null) return;
    setState(() {
      _activity = _Activity.recognizing;
      _error = null;
      _notice = null;
    });
    try {
      final GameState gameState;
      try {
        gameState = await _visionService.processBoardAndRackFromBytes(
          boardImage: table.bytes,
          rackImage: rack.bytes,
        );
      } on VisionException catch (error) {
        _fail(error.message);
        return;
      } catch (error) {
        debugPrint('Recognition failed: $error');
        _fail(
          'Recognition failed. Check your connection and try again, '
          'or enter the tiles manually.',
        );
        return;
      }
      if (!mounted) return;

      if (gameState.board.isEmpty && gameState.rack.isEmpty) {
        // Nothing to correct: stay here with both photos so one tap retakes.
        _fail(
          'No tiles were recognized in either photo. Make sure the tiles are '
          'sharp and well lit, then retake the photos, or enter them manually.',
        );
        return;
      }

      final dump = _visionService.lastDump;
      context.read<GameProvider>().updateGameState(
            gameState,
            fromVision: true,
            visionDebug: RoboflowConfig.debugDump
                ? '${table.image.debugLine} | ${rack.image.debugLine} || '
                    '${dump?.unfilteredSummary ?? _visionService.lastSummaries.join(' | ')}'
                : null,
            roboflowDump: RoboflowConfig.debugDump ? dump : null,
          );
      final emptySide = gameState.rack.isEmpty
          ? 'No tiles were found in the rack photo. Add them on this screen, '
              'or go back and retake it.'
          : gameState.board.isEmpty
              ? 'No sets were found in the table photo. If the table has '
                  'tiles, go back and retake it.'
              : null;
      if (emptySide != null) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(emptySide)),
        );
      }

      setState(() => _activity = _Activity.away);
      final endedGames = _seenEndedGames;
      await Navigator.push<void>(
        context,
        MaterialPageRoute(builder: (_) => const CorrectionScreen()),
      );
      if (!mounted) return;
      // The game was ended over there: nothing to come back to.
      if (_seenEndedGames != endedGames) return;
      // Decision: coming back from the correction screen keeps both photos,
      // so a bad recognition costs one retake instead of two. "Start over"
      // clears them for the next turn.
      setState(() {
        _notice = 'Your photos are still here. Retake either one and tap '
            'Recognize tiles again, or tap Start over for a new turn.';
      });
    } finally {
      if (mounted) setState(() => _activity = _Activity.idle);
    }
  }

  void _fail(String message) {
    if (!mounted) return;
    setState(() => _error = message);
  }

  void _removePhoto(_Slot slot) {
    if (_busy) return;
    setState(() {
      _setPhoto(slot, null);
      _error = null;
      _notice = null;
    });
  }

  void _startOver() {
    if (_busy) return;
    setState(() {
      _table = null;
      _rack = null;
      _error = null;
      _notice = null;
    });
  }

  void _endGame() {
    if (_busy) return;
    _game.endGame();
  }

  Future<void> _openScreen(WidgetBuilder builder) async {
    if (_busy) return;
    setState(() => _activity = _Activity.away);
    try {
      await Navigator.push<void>(context, MaterialPageRoute(builder: builder));
    } finally {
      if (mounted) setState(() => _activity = _Activity.idle);
    }
  }

  void _enterManually() {
    if (_busy) return;
    context.read<GameProvider>().beginManualEntry();
    _openScreen((_) => const CorrectionScreen());
  }

  void _openArena() => _openScreen((_) => const CommunityChallengeScreen());

  void _openHistory() => _openScreen((_) => const HistoryScreen());

  void _openLeaderboard() => _openScreen((_) => const LeaderboardScreen());

  void _openAccount() => _openScreen((_) => const AccountScreen());

  String get _nextStep {
    if (_table == null && _rack != null) {
      return 'Add the table photo to continue.';
    }
    if (_table == null) {
      return 'Next: photograph the table (the sets already played).';
    }
    if (_rack == null) return 'Next: photograph your rack.';
    return 'Both photos are ready. Tap Recognize Tiles.';
  }

  String? get _progressLabel {
    switch (_activity) {
      case _Activity.preparing:
        return 'Preparing the ${_name(_activeSlot ?? _Slot.table)} photo...';
      case _Activity.recognizing:
        return 'Recognizing tiles in both photos...';
      case _Activity.idle:
      case _Activity.picking:
      case _Activity.away:
        return null;
    }
  }

  /// Hover does not exist on a phone: the hints say tap there.
  static bool get _touchFirst => switch (defaultTargetPlatform) {
        TargetPlatform.android ||
        TargetPlatform.iOS ||
        TargetPlatform.fuchsia =>
          true,
        _ => false,
      };

  ZonePhoto? _zonePhoto(_Photo? photo) {
    if (photo == null) return null;
    final debug = RoboflowConfig.debugDump ? '\n${photo.image.debugLine}' : '';
    return ZonePhoto(
      bytes: photo.bytes,
      width: photo.image.width,
      height: photo.image.height,
      label: '${photo.name} · ${photo.image.width}×${photo.image.height}$debug',
    );
  }

  List<DrawerEntry> get _drawerEntries => [
        DrawerEntry(
          entryKey: const ValueKey('end-game'),
          icon: Icons.stop_circle_outlined,
          label: 'End game',
          onTap: _busy ? null : _endGame,
        ),
        const DrawerEntry.divider(),
        DrawerEntry(
          entryKey: const ValueKey('manual-entry'),
          icon: Icons.grid_on_rounded,
          label: 'Enter tiles manually',
          onTap: _busy ? null : _enterManually,
        ),
        DrawerEntry(
          entryKey: const ValueKey('community-challenge'),
          icon: Icons.emoji_events_outlined,
          label: 'Community challenge',
          onTap: _busy ? null : _openArena,
        ),
        const DrawerEntry.divider(),
        DrawerEntry(
          entryKey: const ValueKey('nav-history'),
          icon: Icons.history_rounded,
          label: 'History',
          onTap: _busy ? null : _openHistory,
        ),
        DrawerEntry(
          entryKey: const ValueKey('nav-leaderboard'),
          icon: Icons.leaderboard_outlined,
          label: 'Leaderboard',
          onTap: _busy ? null : _openLeaderboard,
        ),
        DrawerEntry(
          entryKey: const ValueKey('nav-account'),
          icon: Icons.person_outline_rounded,
          label: 'Account',
          onTap: _busy ? null : _openAccount,
        ),
      ];

  @override
  Widget build(BuildContext context) {
    final progress = _progressLabel;
    return Theme(
      data: authTheme(),
      child: Scaffold(
        body: AuthBackdrop(
          child: Stack(
            children: [
              Positioned.fill(
                child: SafeArea(
                  child: LayoutBuilder(
                    builder: (context, constraints) {
                      final layout = SceneLayout.of(
                        Size(
                          math.min(constraints.maxWidth, SceneLayout.maxWidth),
                          constraints.maxHeight,
                        ),
                      );
                      return SingleChildScrollView(
                        child: Center(
                          child: SizedBox.fromSize(
                            size: layout.size,
                            child: _buildScene(context, layout),
                          ),
                        ),
                      );
                    },
                  ),
                ),
              ),
              Positioned.fill(child: HoverDrawer(entries: _drawerEntries)),
              if (progress != null)
                Positioned.fill(
                  child: AbsorbPointer(
                    child: ColoredBox(
                      color: const Color(0x99000000),
                      child: Center(
                        child: Card(
                          child: Padding(
                            padding: const EdgeInsets.all(24),
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                const CircularProgressIndicator(),
                                const SizedBox(height: 16),
                                Text(
                                  progress,
                                  key: const ValueKey('capture-progress'),
                                  style: const TextStyle(
                                    color: AuthColors.ink,
                                    fontWeight: FontWeight.w600,
                                  ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ),
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildScene(BuildContext context, SceneLayout layout) {
    final theme = Theme.of(context);
    final compact = layout.compact;
    final tableReady = _table != null;
    final touch = _touchFirst;
    final hintVerb = touch ? 'Tap' : 'Hover';

    final rackPreview = compact ? const Size(120, 90) : const Size(156, 117);
    // Cards rest half on the rack: this much of one stands above it.
    final rackHeadroom = rackPreview.height * 0.6;
    final boardPreview = Size(
      math.min(layout.board.width * 0.3, 230),
      math.min(layout.board.width * 0.3, 230) * 0.75,
    );
    final messagePad = compact ? 10.0 : 20.0;

    return Stack(
      children: [
        // The three decorative racks. Their ledges face away from the board,
        // as the far players' would.
        Positioned.fromRect(
          rect: layout.topRack,
          child: RotatedBox(
            quarterTurns: 2,
            child: const CustomPaint(painter: RackPainter()),
          ),
        ),
        Positioned.fromRect(
          rect: layout.leftRack,
          child: Transform.rotate(
            angle: layout.sideAngle,
            child: RotatedBox(
              quarterTurns: 1,
              child: const CustomPaint(painter: RackPainter()),
            ),
          ),
        ),
        Positioned.fromRect(
          rect: layout.rightRack,
          child: Transform.rotate(
            angle: -layout.sideAngle,
            child: RotatedBox(
              quarterTurns: 3,
              child: const CustomPaint(painter: RackPainter()),
            ),
          ),
        ),

        // Header.
        Positioned.fromRect(
          rect: layout.header,
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 72),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                RummikubHelperWordmark(compact: compact),
                const SizedBox(height: 8),
                Flexible(
                  child: Text(
                    _nextStep,
                    key: const ValueKey('capture-next-step'),
                    textAlign: TextAlign.center,
                    maxLines: 2,
                    overflow: TextOverflow.ellipsis,
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),

        // The board.
        Positioned.fromRect(
          rect: layout.board,
          child: Stack(
            children: [
              Positioned.fill(
                child: PhotoZone(
                  slot: 'table',
                  semanticsLabel: _table == null
                      ? 'Table photo, not added. $hintVerb to choose Camera or Gallery.'
                      : 'Table photo, added.',
                  surface: (glow) => CustomPaint(
                    painter: BoardPainter(
                      inset: layout.boardInset,
                      glow: glow,
                    ),
                  ),
                  hint: Center(
                    child: _ZoneLabel(
                      title: 'Table photo',
                      subtitle: '$hintVerb to add',
                    ),
                  ),
                  hasPhoto: _table != null,
                  active: _table == null,
                  canCapture: !_busy,
                  cameraSupported: _cameraSupported,
                  onCamera: () => _pick(_Slot.table, ImageSource.camera),
                  onGallery: () => _pick(_Slot.table, ImageSource.gallery),
                  hitClipper: BoardClipper(layout.boardInset),
                  controlsAlignment: const Alignment(0, 0.1),
                ),
              ),
              if (_table != null)
                Align(
                  alignment: const Alignment(0, 0.06),
                  child: PhotoPreviewCard(
                    slot: 'table',
                    photo: _zonePhoto(_table)!,
                    size: boardPreview,
                    tilt: -0.025,
                    cameraSupported: _cameraSupported,
                    canReplace: !_busy,
                    canRemove: !_busy,
                    onCamera: () => _pick(_Slot.table, ImageSource.camera),
                    onGallery: () => _pick(_Slot.table, ImageSource.gallery),
                    onRemove: () => _removePhoto(_Slot.table),
                  ),
                ),
            ],
          ),
        ),

        // The player's rack, with its photo card resting on it.
        Positioned.fromRect(
          rect: Rect.fromLTRB(
            layout.playerRack.left,
            layout.playerRack.top - rackHeadroom,
            layout.playerRack.right,
            layout.playerRack.bottom,
          ),
          child: Stack(
            children: [
              Positioned(
                left: 0,
                right: 0,
                bottom: 0,
                height: layout.playerRack.height,
                child: PhotoZone(
                  slot: 'rack',
                  semanticsLabel: !tableReady
                      ? 'Rack photo, available after the table photo.'
                      : _rack == null
                          ? 'Rack photo, not added. $hintVerb to choose Camera or Gallery.'
                          : 'Rack photo, added.',
                  surface: (glow) => CustomPaint(
                    painter: RackPainter(glow: glow, rows: 2),
                  ),
                  hint: Center(
                    child: _ZoneLabel(
                      title: 'Your rack',
                      subtitle:
                          tableReady ? '$hintVerb to add' : 'table photo first',
                      inline: true,
                    ),
                  ),
                  note: tableReady ? null : 'Photograph the table first',
                  hasPhoto: _rack != null,
                  active: tableReady && _rack == null,
                  canCapture: !_busy && tableReady,
                  cameraSupported: _cameraSupported,
                  onCamera: () => _pick(_Slot.rack, ImageSource.camera),
                  onGallery: () => _pick(_Slot.rack, ImageSource.gallery),
                  compactControls: true,
                ),
              ),
              if (_rack != null)
                Positioned(
                  top: 0,
                  left: 0,
                  right: 0,
                  child: Center(
                    child: PhotoPreviewCard(
                      slot: 'rack',
                      photo: _zonePhoto(_rack)!,
                      size: rackPreview,
                      tilt: 0.03,
                      cameraSupported: _cameraSupported,
                      canReplace: !_busy && tableReady,
                      canRemove: !_busy,
                      onCamera: () => _pick(_Slot.rack, ImageSource.camera),
                      onGallery: () => _pick(_Slot.rack, ImageSource.gallery),
                      onRemove: () => _removePhoto(_Slot.rack),
                    ),
                  ),
                ),
            ],
          ),
        ),

        // Primary action, and the small way back to a clean slate.
        Positioned.fromRect(
          rect: layout.recognize,
          child: FilledButton.icon(
            key: const ValueKey('recognize'),
            onPressed:
                _busy || _table == null || _rack == null ? null : _recognize,
            style: brandButtonStyle(),
            icon: const Icon(Icons.search_rounded),
            label: const Text('Recognize Tiles'),
          ),
        ),
        if (_table != null || _rack != null)
          Positioned.fromRect(
            rect: layout.startOver,
            child: Center(
              child: TextButton.icon(
                key: const ValueKey('start-over'),
                onPressed: _busy ? null : _startOver,
                style: TextButton.styleFrom(
                  foregroundColor: Colors.white70,
                  disabledForegroundColor: Colors.white30,
                  textStyle: const TextStyle(fontWeight: FontWeight.w600),
                ).copyWith(
                  foregroundColor: WidgetStateProperty.resolveWith(
                    (states) => states.contains(WidgetState.disabled)
                        ? Colors.white30
                        : states.contains(WidgetState.hovered)
                            ? Colors.white
                            : Colors.white70,
                  ),
                ),
                icon: const Icon(Icons.restart_alt_rounded, size: 18),
                label: const Text('Start over'),
              ),
            ),
          ),

        // Messages float under the header, over the far decoration.
        Positioned(
          left: messagePad,
          right: messagePad,
          top: layout.header.bottom,
          child: Center(
            child: ConstrainedBox(
              constraints: const BoxConstraints(maxWidth: 560),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  if (!_visionService.isConfigured && !_keyWarningDismissed)
                    _MessageCard(
                      icon: Icons.key_off,
                      message: 'Tile recognition is not set up in this build '
                          '(missing Roboflow API key). Enter the tiles manually, '
                          'or rebuild with --dart-define=ROBOFLOW_API_KEY=your_key.',
                      color: theme.colorScheme.errorContainer,
                      onDismiss: () =>
                          setState(() => _keyWarningDismissed = true),
                    ),
                  if (_error != null)
                    _MessageCard(
                      key: const ValueKey('capture-error'),
                      icon: Icons.error_outline,
                      message: _error!,
                      color: theme.colorScheme.errorContainer,
                      onDismiss: () => setState(() => _error = null),
                    ),
                  if (_notice != null)
                    _MessageCard(
                      key: const ValueKey('capture-notice'),
                      icon: Icons.info_outline,
                      message: _notice!,
                      color: theme.colorScheme.secondaryContainer,
                      onDismiss: () => setState(() => _notice = null),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

/// The label stuck on an empty zone: icon, title, and how to add a photo.
class _ZoneLabel extends StatelessWidget {
  const _ZoneLabel({
    required this.title,
    required this.subtitle,
    this.inline = false,
  });

  final String title;
  final String subtitle;

  /// One line, for the narrow rack.
  final bool inline;

  @override
  Widget build(BuildContext context) {
    final titleStyle = const TextStyle(
      color: Colors.white,
      fontSize: 16,
      fontWeight: FontWeight.w800,
    );
    const subtitleStyle = TextStyle(color: Colors.white70, fontSize: 13);
    return FittedBox(
      fit: BoxFit.scaleDown,
      child: DecoratedBox(
        decoration: BoxDecoration(
          color: const Color(0x99071A45),
          borderRadius: BorderRadius.circular(inline ? 20 : 18),
        ),
        child: Padding(
          padding: EdgeInsets.symmetric(
            horizontal: 16,
            vertical: inline ? 6 : 10,
          ),
          child: inline
              ? Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.add_a_photo_outlined,
                        color: Colors.white, size: 18),
                    const SizedBox(width: 8),
                    Text(title, style: titleStyle),
                    const SizedBox(width: 8),
                    Text(subtitle, style: subtitleStyle),
                  ],
                )
              : Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.add_a_photo_outlined,
                        color: Colors.white, size: 28),
                    const SizedBox(height: 4),
                    Text(title, style: titleStyle),
                    Text(subtitle, style: subtitleStyle),
                  ],
                ),
        ),
      ),
    );
  }
}

class _MessageCard extends StatelessWidget {
  const _MessageCard({
    super.key,
    required this.icon,
    required this.message,
    required this.color,
    this.onDismiss,
  });

  final IconData icon;
  final String message;
  final Color color;
  final VoidCallback? onDismiss;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Material(
        color: color,
        elevation: 4,
        borderRadius: BorderRadius.circular(16),
        child: Padding(
          padding: const EdgeInsets.fromLTRB(12, 8, 4, 8),
          child: Row(
            children: [
              Icon(icon, color: AuthColors.ink),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  message,
                  style: const TextStyle(color: AuthColors.ink),
                ),
              ),
              if (onDismiss != null)
                IconButton(
                  tooltip: 'Dismiss',
                  onPressed: onDismiss,
                  color: AuthColors.ink,
                  icon: const Icon(Icons.close),
                ),
            ],
          ),
        ),
      ),
    );
  }
}
