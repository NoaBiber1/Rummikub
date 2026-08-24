import 'package:flutter_test/flutter_test.dart';
import 'package:rummikub_assistant/providers/game_provider.dart';
import 'package:rummikub_assistant/models/models.dart';

void main() {
  Tile tile(String id, int value) =>
      Tile(id: id, color: 'RED', value: value);

  GameProvider runWithRackEnds() {
    final provider = GameProvider();
    provider.updateGameState(
      GameState(
        board: [
          TileSet(
            id: 'run',
            type: 'RUN',
            tiles: [tile('4', 4), tile('5', 5), tile('6', 6)],
          ),
        ],
        rack: [tile('3', 3), tile('7', 7)],
      ),
    );
    return provider;
  }

  test('inserts to the left of the first tile', () {
    final provider = runWithRackEnds();
    provider.moveTile(
      tileId: '3',
      destination: TileDestination.insertIntoSet('run', 0),
    );
    expect(
      provider.gameState.board.single.tiles.map((t) => t.value),
      [3, 4, 5, 6],
    );
    expect(provider.gameState.board.single.type, 'RUN');
  });

  test('inserts to the right of the last tile', () {
    final provider = runWithRackEnds();
    provider.moveTile(
      tileId: '7',
      destination: TileDestination.insertIntoSet('run', 3),
    );
    expect(
      provider.gameState.board.single.tiles.map((t) => t.value),
      [4, 5, 6, 7],
    );
  });

  test('reorders a tile to the left inside the same set', () {
    final provider = runWithRackEnds();
    provider.moveTile(
      tileId: '6',
      fromSetId: 'run',
      destination: TileDestination.insertIntoSet('run', 0),
    );
    expect(
      provider.gameState.board.single.tiles.map((t) => t.value),
      [6, 4, 5],
    );
  });
}
