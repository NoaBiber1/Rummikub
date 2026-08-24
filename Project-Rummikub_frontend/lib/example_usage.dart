// ignore_for_file: avoid_print

import 'models/models.dart';
void main() {
  // Create sample tiles
  final tile1 = Tile(color: 'RED', value: 7);
  final tile2 = Tile(color: 'BLUE', value: 8);
  final tile3 = Tile(color: 'BLACK', value: 9);
  final joker = Tile(color: 'JOKER', value: 0);

  print('=== Tile Examples ===');
  print('Tile 1: ${tile1.toString()}');
  print('Tile 1 JSON: ${tile1.toJson()}');
  print('');

  // Create a tile from JSON
  final tileFromJson = Tile.fromJson({
    'id': 'test-id-123',
    'color': 'YELLOW',
    'value': 5,
  });
  print('Tile from JSON: ${tileFromJson.toString()}');
  print('');

  // Create a TileSet (run)
  final runSet = TileSet(
    type: 'RUN',
    tiles: [tile1, tile2, tile3],
  );
  print('=== TileSet Example ===');
  print('Run Set: ${runSet.toString()}');
  print('Run Set JSON: ${runSet.toJson()}');
  print('Is valid: ${runSet.isValid}');
  print('Number of tiles: ${runSet.length}');
  print('');

  // Create a TileSet from JSON
  final setFromJson = TileSet.fromJson({
    'id': 'set-id-456',
    'type': 'GROUP',
    'tiles': [
      {'id': 'tile-1', 'color': 'RED', 'value': 5},
      {'id': 'tile-2', 'color': 'BLUE', 'value': 5},
      {'id': 'tile-3', 'color': 'BLACK', 'value': 5},
    ],
  });
  print('Set from JSON: ${setFromJson.toString()}');
  print('');

  // Create a GameState
  final gameState = GameState(
    board: [runSet, setFromJson],
    rack: [joker, Tile(color: 'YELLOW', value: 10)],
  );
  print('=== GameState Example ===');
  print('Game State: ${gameState.toString()}');
  print('Total tiles on board: ${gameState.totalTilesOnBoard}');
  print('Rack size: ${gameState.rackSize}');
  print('Number of sets: ${gameState.numberOfSets}');
  print('All sets valid: ${gameState.isAllSetsValid}');
  print('');
  print('Game State JSON:');
  print(gameState.toJson());
  print('');

  // Create a GameState from JSON
  final gameStateFromJson = GameState.fromJson({
    'board': [
      {
        'id': 'board-set-1',
        'type': 'RUN',
        'tiles': [
          {'id': 'tile-a', 'color': 'RED', 'value': 1},
          {'id': 'tile-b', 'color': 'RED', 'value': 2},
          {'id': 'tile-c', 'color': 'RED', 'value': 3},
        ],
      },
    ],
    'rack': [
      {'id': 'rack-tile-1', 'color': 'BLUE', 'value': 7},
      {'id': 'rack-tile-2', 'color': 'JOKER', 'value': 0},
    ],
  });
  print('Game State from JSON: ${gameStateFromJson.toString()}');
  print('');

  print('=== All tests completed successfully! ===');
}
