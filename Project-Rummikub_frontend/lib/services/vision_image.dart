import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;

import 'vision_image_header.dart';
import 'vision_image_rewrite_stub.dart'
    if (dart.library.js_interop) 'vision_image_rewrite_web.dart';

export 'vision_image_header.dart' show ImageHeader, readImageHeader;

/// Smallest photo worth recognizing. Anything below is an empty or failed
/// capture (Flutter web's canvas resize used to write ~651-byte JPEGs).
const minVisionPhotoBytes = 8000;

/// Longest side sent to recognition. The detector runs at 1024 px, but the
/// classifier reads tile crops from the original, so the extra pixels matter.
const visionMaxSide = 1920;

/// JPEG quality of any re-encode. 4:4:4 chroma is kept on purpose: the tile
/// colors are what the classifier separates.
const visionJpegQuality = 85;

/// JPEG ready for Roboflow: EXIF baked into pixels, longest side <= maxSide.
///
/// The capture screen previews [bytes] and hands the same instance to
/// `VisionService`, so the user sees exactly the image that is recognized.
class PreparedVisionImage {
  const PreparedVisionImage({
    required this.bytes,
    required this.width,
    required this.height,
    required this.exifOrientation,
    required this.bakedExif,
    required this.decoded,
  });

  final Uint8List bytes;
  final int width;
  final int height;
  final int? exifOrientation;
  final bool bakedExif;

  /// False when the input is not a readable image; [bytes] is then the input.
  final bool decoded;

  String get debugLine {
    final orient = exifOrientation == null ? 'none' : '$exifOrientation';
    return '${bytes.length}B ${width}x$height EXIF=$orient baked=$bakedExif';
  }
}

PreparedVisionImage _unreadable(Uint8List bytes) => PreparedVisionImage(
      bytes: bytes,
      width: 0,
      height: 0,
      exifOrientation: null,
      bakedExif: false,
      decoded: false,
    );

/// The input as it is, when its header shows it needs no pixel work: upright
/// and already within [maxSide]. Reads a few hundred bytes, never decodes.
/// Null when the header is unknown (other formats) or the image needs work.
PreparedVisionImage? _asIs(
  Uint8List bytes,
  ImageHeader? header,
  int maxSide,
) {
  if (header == null || header.needsBake || header.longestSide > maxSide) {
    return null;
  }
  return PreparedVisionImage(
    bytes: bytes,
    width: header.width,
    height: header.height,
    exifOrientation: header.orientation,
    bakedExif: false,
    decoded: true,
  );
}

/// Synchronous preparation: full decode only when the photo must change.
///
/// A photo that is upright and within [maxSide] keeps its original bytes
/// (the native pickers already deliver that), so it costs a header read.
/// Otherwise: decode, bake EXIF, downscale, one JPEG encode.
PreparedVisionImage prepareVisionJpeg(
  Uint8List bytes, {
  int maxSide = visionMaxSide,
  int quality = visionJpegQuality,
}) {
  final header = readImageHeader(bytes);
  final asIs = _asIs(bytes, header, maxSide);
  if (asIs != null) return asIs;

  img.Image? decoded;
  try {
    decoded = img.decodeImage(bytes);
  } catch (_) {
    // Truncated or corrupt files can throw inside a decoder instead of
    // returning null. Both mean the same thing: not a readable photo.
    decoded = null;
  }
  if (decoded == null) return _unreadable(bytes);

  // From the file header, not `decoded.exif`: image 4.9.2 drops the Exif
  // block when it decodes a JPEG, so a rotated phone photo would be resized
  // and sent sideways.
  final orientation = header?.orientation;
  final needsBake = orientationNeedsBake(orientation);
  var image = needsBake ? img.bakeOrientation(decoded) : decoded;

  final longest = image.width > image.height ? image.width : image.height;
  if (!needsBake && longest <= maxSide) {
    // Formats the header reader does not know (WebP, BMP...) land here.
    return PreparedVisionImage(
      bytes: bytes,
      width: image.width,
      height: image.height,
      exifOrientation: orientation,
      bakedExif: false,
      decoded: true,
    );
  }
  if (longest > maxSide) {
    image = image.width >= image.height
        ? img.copyResize(
            image,
            width: maxSide,
            interpolation: img.Interpolation.linear,
          )
        : img.copyResize(
            image,
            height: maxSide,
            interpolation: img.Interpolation.linear,
          );
  }

  final jpeg = Uint8List.fromList(img.encodeJpg(image, quality: quality));
  return PreparedVisionImage(
    bytes: jpeg,
    width: image.width,
    height: image.height,
    exifOrientation: orientation,
    bakedExif: needsBake,
    decoded: true,
  );
}

/// [prepareVisionJpeg] without freezing the UI.
///
/// - Upright and small enough: answered from the header, no isolate hop, no
///   decode.
/// - Web: the browser decodes, resizes and encodes natively (off the main
///   thread, no Dart pixel loop). `compute` cannot help there because it runs
///   on the same thread. A browser that cannot do it falls through.
/// - Everywhere else (and as the web fallback): a background isolate.
Future<PreparedVisionImage> prepareVisionJpegInBackground(
  Uint8List bytes, {
  int maxSide = visionMaxSide,
  int quality = visionJpegQuality,
}) async {
  final header = readImageHeader(bytes);
  final asIs = _asIs(bytes, header, maxSide);
  if (asIs != null) return asIs;

  if (kIsWeb) {
    final rewritten = await rewriteInBrowser(
      bytes,
      maxSide: maxSide,
      quality: quality,
      header: header,
    );
    if (rewritten != null) {
      return PreparedVisionImage(
        bytes: rewritten.bytes,
        width: rewritten.width,
        height: rewritten.height,
        exifOrientation: header?.orientation,
        bakedExif: header?.needsBake ?? false,
        decoded: true,
      );
    }
  }
  return compute(_prepare, (bytes, maxSide, quality));
}

PreparedVisionImage _prepare((Uint8List, int, int) args) =>
    prepareVisionJpeg(args.$1, maxSide: args.$2, quality: args.$3);
