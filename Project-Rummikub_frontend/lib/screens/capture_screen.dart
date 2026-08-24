import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:provider/provider.dart';

import '../providers/game_provider.dart';
import '../services/vision_service.dart';
import 'arena_list_screen.dart';
import 'correction_screen.dart';

/// Capture screen where users take photos or enter tiles manually.
class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  final ImagePicker _picker = ImagePicker();
  final VisionService _visionService = VisionService();
  bool _isProcessing = false;

  Future<void> _pickAndProcess(ImageSource source) async {
    if (_isProcessing) return;

    final image = await _picker.pickImage(source: source);
    if (image == null) return;

    setState(() => _isProcessing = true);
    try {
      final gameState = await _visionService.processImage(image);
      if (!mounted) return;
      context.read<GameProvider>().updateGameState(gameState, fromVision: true);
      Navigator.push(
        context,
        MaterialPageRoute(builder: (_) => const CorrectionScreen()),
      );
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not process image: $e')),
      );
    } finally {
      if (mounted) {
        setState(() => _isProcessing = false);
      }
    }
  }

  void _enterManually() {
    context.read<GameProvider>().beginManualEntry();
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const CorrectionScreen()),
    );
  }

  void _openArena() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (_) => const ArenaListScreen()),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Play'),
      ),
      body: Center(
        child: _isProcessing
            ? const Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  CircularProgressIndicator(),
                  SizedBox(height: 16),
                  Text('Recognizing tiles...'),
                ],
              )
            : Padding(
                padding: const EdgeInsets.all(24),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      'Take a photo of the board and rack, or enter tiles by hand.',
                      style: TextStyle(fontSize: 18),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 24),
                    FilledButton.icon(
                      onPressed: () => _pickAndProcess(ImageSource.camera),
                      icon: const Icon(Icons.camera_alt),
                      label: const Text('Camera'),
                    ),
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      onPressed: () => _pickAndProcess(ImageSource.gallery),
                      icon: const Icon(Icons.photo_library),
                      label: const Text('Gallery'),
                    ),
                    const SizedBox(height: 12),
                    OutlinedButton.icon(
                      onPressed: _enterManually,
                      icon: const Icon(Icons.grid_on),
                      label: const Text('Enter tiles manually'),
                    ),
                    const SizedBox(height: 24),
                    FilledButton.tonalIcon(
                      onPressed: _openArena,
                      icon: const Icon(Icons.emoji_events_outlined),
                      label: const Text('Community challenge'),
                    ),
                  ],
                ),
              ),
      ),
    );
  }
}
