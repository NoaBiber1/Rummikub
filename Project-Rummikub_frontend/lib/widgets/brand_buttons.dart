import 'package:flutter/material.dart';

import 'auth_brand.dart';

/// The login screen's button, with the states a hover-driven screen needs.
///
/// Same red, 16 px corners and bold white label as the sign-in button. On top
/// of that: a lighter red and a lift on hover, a darker red when pressed, and a
/// washed-out red when disabled (the login theme paints disabled buttons the
/// same red as enabled ones). Material animates every change below over 200 ms.
ButtonStyle brandButtonStyle({bool compact = false}) {
  return ButtonStyle(
    backgroundColor: WidgetStateProperty.resolveWith((states) {
      if (states.contains(WidgetState.disabled)) {
        return const Color(0xFF3A4F85);
      }
      if (states.contains(WidgetState.pressed)) return const Color(0xFF9E1015);
      if (states.contains(WidgetState.hovered) ||
          states.contains(WidgetState.focused)) {
        return const Color(0xFFE02A31);
      }
      return AuthColors.red;
    }),
    foregroundColor: WidgetStateProperty.resolveWith(
      (states) =>
          states.contains(WidgetState.disabled) ? Colors.white54 : Colors.white,
    ),
    elevation: WidgetStateProperty.resolveWith((states) {
      if (states.contains(WidgetState.disabled)) return 0;
      return states.contains(WidgetState.hovered) ? 6 : 1;
    }),
    shadowColor: const WidgetStatePropertyAll(Color(0x99000000)),
    minimumSize: WidgetStatePropertyAll(Size(0, compact ? 40 : 56)),
    padding: WidgetStatePropertyAll(
      EdgeInsets.symmetric(horizontal: compact ? 14 : 28),
    ),
    shape: WidgetStatePropertyAll(
      RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
    ),
    textStyle: WidgetStatePropertyAll(
      TextStyle(fontWeight: FontWeight.w700, fontSize: compact ? 14 : 17),
    ),
    animationDuration: const Duration(milliseconds: 200),
  );
}

/// The quiet partner of [brandButtonStyle]: ivory with red lettering.
ButtonStyle brandLightButtonStyle({bool compact = false}) {
  return brandButtonStyle(compact: compact).copyWith(
    backgroundColor: WidgetStateProperty.resolveWith((states) {
      if (states.contains(WidgetState.disabled)) return const Color(0x66F7F3EA);
      if (states.contains(WidgetState.pressed)) return const Color(0xFFE4DDCB);
      if (states.contains(WidgetState.hovered) ||
          states.contains(WidgetState.focused)) {
        return Colors.white;
      }
      return AuthColors.ivory;
    }),
    foregroundColor: WidgetStateProperty.resolveWith(
      (states) => states.contains(WidgetState.disabled)
          ? const Color(0x99141414)
          : AuthColors.red,
    ),
  );
}
