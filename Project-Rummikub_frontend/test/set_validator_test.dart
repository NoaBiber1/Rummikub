import 'package:flutter_test/flutter_test.dart';
import 'package:rummikub_assistant/logic/set_validator.dart';
import 'package:rummikub_assistant/models/models.dart';

void main() {
  TileSet setOf(String type, List<Tile> tiles) =>
      TileSet(type: type, tiles: tiles);

  test('valid run is classified RUN', () {
    final classified = SetValidator.classifySet(
      setOf('INVALID', [
        Tile(color: 'RED', value: 1),
        Tile(color: 'RED', value: 2),
        Tile(color: 'RED', value: 3),
      ]),
    );
    expect(classified.type, 'RUN');
    expect(SetValidator.isBoardValid([classified]), isTrue);
  });

  test('valid group is classified GROUP', () {
    final classified = SetValidator.classifySet(
      setOf('INVALID', [
        Tile(color: 'RED', value: 5),
        Tile(color: 'BLUE', value: 5),
        Tile(color: 'BLACK', value: 5),
      ]),
    );
    expect(classified.type, 'GROUP');
  });

  test('short set is invalid', () {
    final classified = SetValidator.classifySet(
      setOf('RUN', [
        Tile(color: 'RED', value: 1),
        Tile(color: 'RED', value: 2),
      ]),
    );
    expect(classified.type, 'INVALID');
    expect(SetValidator.isBoardValid([classified]), isFalse);
  });

  test('empty board is valid', () {
    expect(SetValidator.isBoardValid([]), isTrue);
  });
}
