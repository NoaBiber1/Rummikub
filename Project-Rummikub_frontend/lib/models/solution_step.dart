import 'tile.dart';

/// Mirrors the backend `StepType` enum.
enum SolutionStepType {
  placeFromRack('PLACE_FROM_RACK'),
  moveBetweenSets('MOVE_BETWEEN_SETS'),
  formNewSet('FORM_NEW_SET'),
  disbandSet('DISBAND_SET');

  const SolutionStepType(this.wire);

  final String wire;

  static SolutionStepType parse(String value) {
    return SolutionStepType.values.firstWhere(
      (t) => t.wire == value,
      orElse: () => throw ArgumentError('Invalid step type: $value'),
    );
  }
}

/// One transition in `optimal_state.solution_steps`.
///
/// A null [sourceSetId] means the rack (or nothing, for a new set). A null
/// [targetSetId] means the tiles are released (a disband).
class SolutionStep {
  const SolutionStep({
    required this.stepIndex,
    required this.type,
    this.sourceSetId,
    this.targetSetId,
    required this.tiles,
    required this.description,
  });

  final int stepIndex;
  final SolutionStepType type;
  final String? sourceSetId;
  final String? targetSetId;
  final List<Tile> tiles;
  final String description;

  factory SolutionStep.fromJson(Map<String, dynamic> json) {
    return SolutionStep(
      stepIndex: (json['stepIndex'] as num).toInt(),
      type: SolutionStepType.parse(json['type'] as String),
      sourceSetId: json['sourceSetId'] as String?,
      targetSetId: json['targetSetId'] as String?,
      tiles: (json['tiles'] as List<dynamic>)
          .map((t) => Tile.fromJson(t as Map<String, dynamic>))
          .toList(),
      description: json['description'] as String? ?? '',
    );
  }

  Map<String, dynamic> toJson() => {
        'stepIndex': stepIndex,
        'type': type.wire,
        'sourceSetId': sourceSetId,
        'targetSetId': targetSetId,
        'tiles': tiles.map((t) => t.toJson()).toList(),
        'description': description,
      };
}
