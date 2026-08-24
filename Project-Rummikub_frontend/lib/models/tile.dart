import 'package:uuid/uuid.dart';

/// Represents a Rummikub tile with color and value.
/// 
/// Valid colors: RED, BLUE, BLACK, YELLOW, JOKER
/// Valid values: 1-13 for regular tiles, 0 for JOKER
class Tile {
  final String id;
  final String color;
  final int value;

  Tile({
    String? id,
    required this.color,
    required this.value,
  }) : id = id ?? const Uuid().v4();

  /// Creates a Tile from JSON map
  factory Tile.fromJson(Map<String, dynamic> json) {
    if (json['color'] == null || json['value'] == null) {
      throw ArgumentError('Tile JSON must contain color and value fields');
    }

    final color = json['color'] as String;
    final rawValue = json['value'];
    final value = rawValue is num
        ? rawValue.toInt()
        : int.parse(rawValue.toString());

    if (!_isValidColor(color)) {
      throw ArgumentError('Invalid color: $color. Must be RED, BLUE, BLACK, YELLOW, or JOKER');
    }

    if (!_isValidValue(value, color)) {
      throw ArgumentError('Invalid value: $value for color: $color');
    }

    return Tile(
      id: json['id'] as String?,
      color: color,
      value: value,
    );
  }

  /// Converts Tile to JSON map
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'color': color,
      'value': value,
    };
  }

  /// Validates if the color is one of the allowed values
  static bool _isValidColor(String color) {
    return ['RED', 'BLUE', 'BLACK', 'YELLOW', 'JOKER'].contains(color);
  }

  /// Validates if the value is appropriate for the given color
  static bool _isValidValue(int value, String color) {
    if (color == 'JOKER') {
      return value == 0;
    }
    return value >= 1 && value <= 13;
  }

  /// Creates a copy of this Tile with optional field updates
  Tile copyWith({
    String? id,
    String? color,
    int? value,
  }) {
    return Tile(
      id: id ?? this.id,
      color: color ?? this.color,
      value: value ?? this.value,
    );
  }

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is Tile &&
        other.id == id &&
        other.color == color &&
        other.value == value;
  }

  @override
  int get hashCode => Object.hash(id, color, value);

  @override
  String toString() => 'Tile(id: $id, color: $color, value: $value)';
}
