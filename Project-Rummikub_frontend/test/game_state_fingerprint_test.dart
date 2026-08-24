import 'package:flutter_test/flutter_test.dart';
import 'package:rummikub_assistant/logic/game_state_fingerprint.dart';
import 'package:rummikub_assistant/models/models.dart';

void main() {
  test('same tiles with different ids share a fingerprint', () {
    final a = GameState(
      board: [
        TileSet(
          id: 'set-a',
          type: 'RUN',
          tiles: [
            Tile(id: '1', color: 'RED', value: 1),
            Tile(id: '2', color: 'RED', value: 2),
            Tile(id: '3', color: 'RED', value: 3),
          ],
        ),
      ],
      rack: [Tile(id: 'r1', color: 'BLUE', value: 8)],
    );
    final b = GameState(
      board: [
        TileSet(
          id: 'set-b',
          type: 'RUN',
          tiles: [
            Tile(id: 'x', color: 'RED', value: 1),
            Tile(id: 'y', color: 'RED', value: 2),
            Tile(id: 'z', color: 'RED', value: 3),
          ],
        ),
      ],
      rack: [Tile(id: 'r2', color: 'BLUE', value: 8)],
    );

    expect(GameStateFingerprint.canonical(a), GameStateFingerprint.canonical(b));
    expect(GameStateFingerprint.challengeId(a), GameStateFingerprint.challengeId(b));
  });

  test('different racks produce different ids', () {
    GameState withRack(int value) => GameState(
          board: [],
          rack: [Tile(color: 'RED', value: value)],
        );

    expect(
      GameStateFingerprint.challengeId(withRack(1)),
      isNot(GameStateFingerprint.challengeId(withRack(2))),
    );
  });
}
