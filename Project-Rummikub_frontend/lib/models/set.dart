import 'package:uuid/uuid.dart';
import 'json_map.dart';
import 'tile.dart';

/// Represents a set of tiles in Rummikub (a valid run, group, or invalid combination).
/// 
/// Types: RUN, GROUP, INVALID
/// A valid set must have at least 3 tiles
class TileSet {
  final String id;
  final String type;
  final List<Tile> tiles;

  TileSet({
    String? id,
    required this.type,
    required this.tiles,
  }) : id = id ?? const Uuid().v4();

  /// Creates a TileSet from JSON map
  factory TileSet.fromJson(Map<String, dynamic> json) {
    if (json['type'] == null || json['tiles'] == null) {
      throw ArgumentError('TileSet JSON must contain type and tiles fields');
    }

    final type = json['type'] as String;
    if (!_isValidType(type)) {
      throw ArgumentError('Invalid type: $type. Must be RUN, GROUP, or INVALID');
    }

    final tiles = asMapList(json['tiles']).map(Tile.fromJson).toList();

    return TileSet(
      id: json['id'] as String?,
      type: type,
      tiles: tiles,
    );
  }

  /// Converts TileSet to JSON map
  Map<String, dynamic> toJson() {
    return {
      'id': id,
      'type': type,
      'tiles': tiles.map((tile) => tile.toJson()).toList(),
    };
  }

  /// Validates if the type is one of the allowed values
  static bool _isValidType(String type) {
    return ['RUN', 'GROUP', 'INVALID'].contains(type);
  }

  /// Creates a copy of this TileSet with optional field updates
  TileSet copyWith({
    String? id,
    String? type,
    List<Tile>? tiles,
  }) {
    return TileSet(
      id: id ?? this.id,
      type: type ?? this.type,
      tiles: tiles ?? List.from(this.tiles),
    );
  }

  /// Returns the number of tiles in this set
  int get length => tiles.length;

  /// Checks if this set is empty
  bool get isEmpty => tiles.isEmpty;

  /// Checks if this set is not empty
  bool get isNotEmpty => tiles.isNotEmpty;

  /// Checks if this set is valid (RUN or GROUP)
  bool get isValid => type == 'RUN' || type == 'GROUP';

  /// Checks if this set is invalid
  bool get isInvalid => type == 'INVALID';

  @override
  bool operator ==(Object other) {
    if (identical(this, other)) return true;
    return other is TileSet &&
        other.id == id &&
        other.type == type &&
        _listEquals(other.tiles, tiles);
  }

  @override
  int get hashCode => Object.hash(id, type, Object.hashAll(tiles));

  @override
  String toString() => 'TileSet(id: $id, type: $type, tiles: ${tiles.length})';

  /// Helper method to compare two lists of tiles
  static bool _listEquals(List<Tile> a, List<Tile> b) {
    if (a.length != b.length) return false;
    for (int i = 0; i < a.length; i++) {
      if (a[i] != b[i]) return false;
    }
    return true;
  }
}
