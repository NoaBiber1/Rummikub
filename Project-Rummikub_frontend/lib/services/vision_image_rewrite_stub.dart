import 'dart:typed_data';

import 'vision_image_header.dart';

/// Result of a browser-side decode, resize and JPEG encode.
typedef BrowserRewrite = ({Uint8List bytes, int width, int height});

/// Only the web build has a native decoder to offload to; elsewhere the
/// caller uses a background isolate.
Future<BrowserRewrite?> rewriteInBrowser(
  Uint8List bytes, {
  required int maxSide,
  required int quality,
  ImageHeader? header,
}) async =>
    null;
