import 'dart:typed_data';

/// Size and EXIF orientation read from a file header, with no pixel decode.
class ImageHeader {
  const ImageHeader({
    required this.width,
    required this.height,
    this.orientation,
  });

  /// As stored in the file, before any EXIF rotation.
  final int width;
  final int height;

  /// EXIF orientation 1-8, or null when the file has none.
  final int? orientation;

  int get longestSide => width > height ? width : height;

  bool get needsBake => orientationNeedsBake(orientation);
}

/// Orientations 2-8 move pixels; 1 is upright and anything else is invalid.
bool orientationNeedsBake(int? orientation) =>
    orientation != null && orientation >= 2 && orientation <= 8;

/// Reads the header of a JPEG or PNG in a few hundred bytes. Null for any
/// other format and for damaged headers; the caller then decodes in full.
ImageHeader? readImageHeader(Uint8List bytes) {
  if (bytes.length >= 4 && bytes[0] == 0xFF && bytes[1] == 0xD8) {
    return _jpegHeader(bytes);
  }
  if (bytes.length >= 24 && _isPng(bytes)) return _pngHeader(bytes);
  return null;
}

bool _isPng(Uint8List b) =>
    b[0] == 0x89 &&
    b[1] == 0x50 &&
    b[2] == 0x4E &&
    b[3] == 0x47 &&
    b[4] == 0x0D &&
    b[5] == 0x0A &&
    b[6] == 0x1A &&
    b[7] == 0x0A;

ImageHeader? _pngHeader(Uint8List b) {
  // 8-byte signature, then the IHDR chunk: length, "IHDR", width, height.
  if (b[12] != 0x49 || b[13] != 0x48 || b[14] != 0x44 || b[15] != 0x52) {
    return null;
  }
  final data = ByteData.sublistView(b);
  final width = data.getUint32(16);
  final height = data.getUint32(20);
  if (width == 0 || height == 0) return null;
  return ImageHeader(width: width, height: height);
}

/// Walks the marker segments up to the frame header. The Exif block comes
/// before it, so both are found without touching the entropy-coded data.
ImageHeader? _jpegHeader(Uint8List b) {
  final data = ByteData.sublistView(b);
  int? orientation;
  var i = 2;
  while (i + 4 <= b.length) {
    if (b[i] != 0xFF) return null;
    final marker = b[i + 1];
    if (marker == 0xFF) {
      i++; // fill byte
      continue;
    }
    // Markers without a length: TEM, RSTn.
    if (marker == 0x01 || (marker >= 0xD0 && marker <= 0xD8)) {
      i += 2;
      continue;
    }
    // End of image / start of scan before any frame header.
    if (marker == 0xD9 || marker == 0xDA) return null;

    final length = data.getUint16(i + 2);
    final end = i + 2 + length;
    if (length < 2 || end > b.length) return null;

    if (marker == 0xE1) orientation ??= _exifOrientation(data, i + 4, end);
    if (_isFrameHeader(marker)) {
      if (length < 8) return null;
      final height = data.getUint16(i + 5);
      final width = data.getUint16(i + 7);
      if (width == 0 || height == 0) return null;
      return ImageHeader(
        width: width,
        height: height,
        orientation: orientation,
      );
    }
    i = end;
  }
  return null;
}

/// SOF0-SOF15, except DHT (C4), JPG (C8) and DAC (CC).
bool _isFrameHeader(int marker) =>
    marker >= 0xC0 &&
    marker <= 0xCF &&
    marker != 0xC4 &&
    marker != 0xC8 &&
    marker != 0xCC;

/// Orientation tag (0x0112) of the first IFD in an APP1 "Exif" segment.
int? _exifOrientation(ByteData data, int start, int end) {
  // "Exif\0\0", then a TIFF header: byte order, 0x002A, offset to IFD0.
  if (end - start < 14 ||
      data.getUint32(start) != 0x45786966 ||
      data.getUint16(start + 4) != 0) {
    return null;
  }
  final tiff = start + 6;
  final Endian endian;
  switch (data.getUint16(tiff)) {
    case 0x4949:
      endian = Endian.little;
    case 0x4D4D:
      endian = Endian.big;
    default:
      return null;
  }
  if (data.getUint16(tiff + 2, endian) != 0x002A) return null;
  final ifd = tiff + data.getUint32(tiff + 4, endian);
  if (ifd + 2 > end) return null;
  final entries = data.getUint16(ifd, endian);
  for (var n = 0; n < entries; n++) {
    final entry = ifd + 2 + n * 12;
    if (entry + 12 > end) return null;
    if (data.getUint16(entry, endian) == 0x0112) {
      // SHORT, one value, stored in the first two bytes of the value field.
      return data.getUint16(entry + 8, endian);
    }
  }
  return null;
}
