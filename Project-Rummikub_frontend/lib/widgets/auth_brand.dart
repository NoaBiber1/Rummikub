import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Palette sampled from the original Rummikub box art.
abstract final class AuthColors {
  static const navy = Color(0xFF071A45);
  static const royal = Color(0xFF0C2F78);
  static const glow = Color(0xFF1A4FA3);
  static const red = Color(0xFFC4161C);
  static const ivory = Color(0xFFF7F3EA);
  static const ink = Color(0xFF141414);
  static const field = Color(0xFFFFFFFF);
  static const fieldBorder = Color(0xFFD9D3C7);
}

/// Tests turn this off: Google Fonts cannot load there (no network, no
/// bundled files) and reports the failed load as an uncaught async error.
/// The lettering then falls back to the default font; the layout is the same.
@visibleForTesting
bool brandFontsEnabled = true;

/// The red display face: Playfair Display Black Italic.
TextStyle brandDisplayStyle({
  required double fontSize,
  required double height,
  required double letterSpacing,
}) {
  const weight = FontWeight.w900;
  return brandFontsEnabled
      ? GoogleFonts.playfairDisplay(
          fontSize: fontSize,
          fontWeight: weight,
          fontStyle: FontStyle.italic,
          height: height,
          letterSpacing: letterSpacing,
        )
      : TextStyle(
          fontSize: fontSize,
          fontWeight: weight,
          fontStyle: FontStyle.italic,
          height: height,
          letterSpacing: letterSpacing,
        );
}

ThemeData authTheme() {
  const radius = 16.0;
  final rounded = RoundedRectangleBorder(
    borderRadius: BorderRadius.circular(radius),
  );
  final buttonStyle = ButtonStyle(
    backgroundColor: const WidgetStatePropertyAll(AuthColors.red),
    foregroundColor: const WidgetStatePropertyAll(Colors.white),
    minimumSize: const WidgetStatePropertyAll(Size.fromHeight(52)),
    elevation: const WidgetStatePropertyAll(0),
    shape: WidgetStatePropertyAll(rounded),
    textStyle: const WidgetStatePropertyAll(
      TextStyle(fontWeight: FontWeight.w700, fontSize: 16),
    ),
  );

  return ThemeData(
    useMaterial3: true,
    brightness: Brightness.light,
    colorScheme: ColorScheme.fromSeed(
      seedColor: AuthColors.red,
      primary: AuthColors.red,
      surface: AuthColors.ivory,
    ),
    scaffoldBackgroundColor: Colors.transparent,
    textTheme: const TextTheme(
      headlineSmall: TextStyle(
        color: Colors.white,
        fontSize: 24,
        fontWeight: FontWeight.w600,
      ),
      bodyMedium: TextStyle(color: Colors.white),
      bodySmall: TextStyle(color: Colors.white70),
      bodyLarge: TextStyle(color: Color(0xFF1A1A1A)),
    ),
    inputDecorationTheme: InputDecorationTheme(
      filled: true,
      fillColor: AuthColors.field,
      contentPadding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
      labelStyle: const TextStyle(color: Color(0xFF6B6570)),
      floatingLabelStyle: const TextStyle(color: AuthColors.red),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radius),
        borderSide: const BorderSide(color: AuthColors.fieldBorder),
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radius),
        borderSide: const BorderSide(color: AuthColors.fieldBorder),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radius),
        borderSide: const BorderSide(color: AuthColors.red, width: 1.6),
      ),
      errorBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(radius),
        borderSide: const BorderSide(color: AuthColors.red),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(style: buttonStyle),
    filledButtonTheme: FilledButtonThemeData(style: buttonStyle),
    textButtonTheme: TextButtonThemeData(
      style: TextButton.styleFrom(
        foregroundColor: AuthColors.red,
        textStyle: const TextStyle(fontWeight: FontWeight.w600),
      ),
    ),
  );
}

/// Deep blue box-art backdrop. Intentionally empty at the bottom.
class AuthBackdrop extends StatelessWidget {
  const AuthBackdrop({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        const Positioned.fill(
          child: DecoratedBox(
            decoration: BoxDecoration(
              gradient: RadialGradient(
                center: Alignment(0, -0.62),
                radius: 1.2,
                colors: [
                  AuthColors.glow,
                  AuthColors.royal,
                  AuthColors.navy,
                ],
                stops: [0.0, 0.42, 1.0],
              ),
            ),
          ),
        ),
        child,
      ],
    );
  }
}

/// "Rummikub" in the original red display face, with "Helper" just to its right.
class RummikubHelperWordmark extends StatelessWidget {
  const RummikubHelperWordmark({super.key, this.compact = false});

  final bool compact;

  @override
  Widget build(BuildContext context) {
    final rummikubStyle = brandDisplayStyle(
      fontSize: compact ? 44 : 56,
      height: 0.86,
      letterSpacing: -1.2,
    );
    const shadows = [
      Shadow(color: Color(0xF2FFFFFF), blurRadius: 8),
      Shadow(color: Color(0x99FFFFFF), blurRadius: 16),
    ];
    final helperStyle = brandFontsEnabled
        ? GoogleFonts.allura(
            fontSize: compact ? 40 : 48,
            fontWeight: FontWeight.w400,
            height: 0.7,
            color: AuthColors.ink,
            shadows: shadows,
          )
        : TextStyle(
            fontSize: compact ? 40 : 48,
            fontStyle: FontStyle.italic,
            height: 0.7,
            color: AuthColors.ink,
            shadows: shadows,
          );

    return FittedBox(
      fit: BoxFit.scaleDown,
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          OutlinedText(
            'Rummikub',
            style: rummikubStyle,
            fill: AuthColors.red,
            strokeWidth: compact ? 5 : 6.5,
          ),
          Transform.translate(
            offset: Offset(compact ? -8 : -10, compact ? 2 : 4),
            child: Text('Helper', style: helperStyle),
          ),
        ],
      ),
    );
  }
}

/// A section or drawer title in the wordmark's red display face.
class BrandHeading extends StatelessWidget {
  const BrandHeading(this.text, {super.key, this.fontSize = 26});

  final String text;
  final double fontSize;

  @override
  Widget build(BuildContext context) {
    return OutlinedText(
      text,
      style: brandDisplayStyle(
        fontSize: fontSize,
        height: 0.95,
        letterSpacing: -0.6,
      ),
      fill: AuthColors.red,
      strokeWidth: fontSize * 0.14,
    );
  }
}

class AuthFormCard extends StatelessWidget {
  const AuthFormCard({super.key, required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.fromLTRB(8, 8, 8, 12),
      child: child,
    );
  }
}

/// [text] filled with [fill] under a white outline and a hard drop shadow:
/// the Rummikub box-art lettering.
class OutlinedText extends StatelessWidget {
  const OutlinedText(
    this.text, {
    super.key,
    required this.style,
    required this.fill,
    this.strokeWidth = 6,
  });

  final String text;
  final TextStyle style;
  final Color fill;
  final double strokeWidth;

  @override
  Widget build(BuildContext context) {
    return Stack(
      alignment: Alignment.center,
      children: [
        Text(
          text,
          textAlign: TextAlign.center,
          style: style.copyWith(
            foreground: Paint()
              ..style = PaintingStyle.stroke
              ..strokeJoin = StrokeJoin.round
              ..strokeCap = StrokeCap.round
              ..strokeWidth = strokeWidth
              ..color = Colors.white,
          ),
        ),
        Text(
          text,
          textAlign: TextAlign.center,
          style: style.copyWith(
            color: fill,
            shadows: const [
              Shadow(
                color: Color(0x66000000),
                offset: Offset(1.5, 2.5),
                blurRadius: 0,
              ),
            ],
          ),
        ),
      ],
    );
  }
}
