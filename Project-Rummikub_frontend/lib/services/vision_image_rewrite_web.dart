import 'dart:async';
import 'dart:js_interop';
import 'dart:math' as math;
import 'dart:typed_data';

import 'package:web/web.dart' as web;

import 'vision_image_header.dart';

/// Result of a browser-side decode, resize and JPEG encode.
typedef BrowserRewrite = ({Uint8List bytes, int width, int height});

/// Decodes, downscales and JPEG-encodes [bytes] with the browser's own codecs.
///
/// The `image` package decodes in a Dart loop on the UI thread when compiled
/// to JavaScript: seconds of frozen UI for a 12 MP photo. The browser decodes
/// natively and encodes asynchronously. Returns null whenever it cannot be
/// trusted to match the Dart path (decode failure, EXIF not applied, no
/// canvas), so the caller falls back to [prepareVisionJpeg].
Future<BrowserRewrite?> rewriteInBrowser(
  Uint8List bytes, {
  required int maxSide,
  required int quality,
  ImageHeader? header,
}) async {
  web.ImageBitmap? bitmap;
  try {
    final blob = web.Blob([bytes.toJS].toJS);
    // Browsers apply the EXIF orientation while decoding.
    bitmap = await web.window.createImageBitmap(blob).toDart;
    final width = bitmap.width;
    final height = bitmap.height;
    if (width == 0 || height == 0) return null;

    // Orientations 5-8 swap the sides. If the sides still match the file
    // header, this browser ignored EXIF: let the Dart path bake it instead of
    // sending a sideways photo.
    final orientation = header?.orientation;
    if (header != null &&
        orientation != null &&
        orientation >= 5 &&
        orientation <= 8 &&
        width != height &&
        width == header.width &&
        height == header.height) {
      return null;
    }

    final longest = math.max(width, height);
    final scale = longest > maxSide ? maxSide / longest : 1.0;
    final outWidth = math.max(1, (width * scale).round());
    final outHeight = math.max(1, (height * scale).round());

    final canvas = web.HTMLCanvasElement()
      ..width = outWidth
      ..height = outHeight;
    final context = canvas.getContext('2d') as web.CanvasRenderingContext2D?;
    if (context == null) return null;
    // JPEG has no alpha: paint under transparent PNGs instead of getting black.
    context.fillStyle = '#fff'.toJS;
    context.fillRect(0, 0, outWidth, outHeight);
    context.imageSmoothingEnabled = true;
    context.imageSmoothingQuality = 'high';
    context.drawImage(bitmap, 0, 0, outWidth, outHeight);

    final encoded = Completer<web.Blob?>();
    canvas.toBlob(
      (web.Blob? blob) {
        encoded.complete(blob);
      }.toJS,
      'image/jpeg',
      (quality / 100).toJS,
    );
    final jpeg = await encoded.future;
    if (jpeg == null) return null;
    final buffer = await jpeg.arrayBuffer().toDart;
    return (
      bytes: buffer.toDart.asUint8List(),
      width: outWidth,
      height: outHeight,
    );
  } catch (_) {
    return null;
  } finally {
    bitmap?.close();
  }
}
