# Rummikub Move API

`api.py` is the single entry point to the Rummikub move policies. One call, `get_move`, gives the same position to every policy and returns their moves together in one dict.

```python
from api import get_move

moves = get_move(hand, board, first_meld=False)
```

| Key | Policy | Current value |
|---|---|---|
| `"greedy_result"` | Greedy solver: plays as many hand tiles as the rules allow | A move (see [Output](#output)) |
| `"max_q_result"` | Max-Q policy, reserved for the learned model | Always `''` until that model is integrated |

---

## Signature

```python
get_move(hand: list, board: list, first_meld: bool = False) -> dict
```

---

## Input

### Tile format

Every tile, in both `hand` and `board`, is **exactly** one of:

| Tile | Format | Example |
|---|---|---|
| Joker | the string `"J"` | `"J"` |
| Number tile | a 2-tuple `(number, color)`, in that order | `(5, "red")` |

- `number` is an `int` from `1` to `13`.
- `color` is one of `"red"`, `"yellow"`, `"blue"`, `"black"` (lower case).

Anything else raises `ValueError`:

| Rejected | Why |
|---|---|
| `("red", 5)` | wrong order |
| `[5, "red"]` | list, not tuple |
| `"5 red"`, `"red5"` | string, not tuple |
| `(5, "Red")`, `(5, "green")` | colour not one of the four lower-case names |
| `(14, "red")`, `(0, "red")` | number out of range |
| `(5.0, "red")`, `(True, "red")` | number not an `int` |
| `"j"`, `"joker"` | joker must be exactly `"J"` |

### `hand` — `list`

The player's tiles, in any order.

```python
hand = [(4, "red"), (9, "blue"), "J"]
```

### `board` — `list` (flat)

Every tile on the table as **one flat list**, with no set structure.

```python
board = [(3, "red"), (4, "red"), (5, "red"), (9, "blue"), (9, "black"), "J"]
```

Nested sets are **not** accepted:

```python
board = [[(3, "red"), (4, "red"), (5, "red")]]   # ValueError
```

The board must split **completely and disjointly** into legal sets, with every tile in exactly one set:

- **Group:** 3 or 4 tiles of the same number in different colours.
- **Run:** 3 to 13 consecutive numbers in one colour. 1 is always low; runs do not wrap past 13.
- **Joker:** may stand in for any tile.

This is checked before anything else, whatever the hand holds. For example, `[(5, "red"), (6, "red")]` raises even if the hand holds the red 7.

**Copy limit:** across `hand` and `board` together, no tile may appear more than twice, and there are at most two jokers.

### `first_meld` — `bool`, default `False`

| Value | Turn type | Rule |
|---|---|---|
| `False` | Ordinary turn | The whole board may be rearranged, as long as every board tile ends up in a legal set. |
| `True` | Initial meld | The move must be **new sets from the hand alone**, worth **at least 30 points**, with the board left untouched. A joker counts as the number it represents. |

---

## Output

`get_move` returns a dict with exactly two keys:

```python
{"greedy_result": <move>, "max_q_result": <move>}
```

A `<move>` is one of two things.

### 1. Tiles can be played: a 4-tuple

```python
(hand, action, sBoard, numOfTails)
```

| Position | Name | Type | Meaning |
|---|---|---|---|
| 0 | `hand` | `list` of tiles | Tiles left in the hand, in their original order |
| 1 | `action` | `list` of tiles | Tiles moved from the hand to the table, in hand order |
| 2 | `sBoard` | `list` of `list`s of tiles | Every set on the table after the move (see below) |
| 3 | `numOfTails` | `int` | Number of tiles played; always `len(action)` |

About `sBoard`:

- Each inner list is one legal group or run, in display order. Runs are ascending, and a joker sits in the position whose number it takes.
- Together the sets hold each original board tile and each played tile **exactly once**.
- On an initial-meld turn, the board's own sets come first and the new sets follow.

The tuple is a named tuple, so its parts are also available as `.hand`, `.action`, `.sBoard` and `.numOfTails`.

### 2. No tile can be played: a string

```python
"Draw a tile"
```

This is the exact string.

### JSON structure

The result is returned as a Python `dict`. Serialised with `json.dumps(result)`, it has the following JSON structure.

**When tiles can be played:**

```json
{
  "greedy_result": [
    [[1, "black"]],
    [[4, "red"], [9, "blue"]],
    [
      [[9, "red"], [9, "yellow"], [9, "blue"], [9, "black"]],
      [[4, "red"], [5, "red"], [6, "red"], [7, "red"]]
    ],
    2
  ],
  "max_q_result": ""
}
```

Here `"greedy_result"` is an array of exactly four elements, in this order:

| Index | Field | JSON type | Content |
|---|---|---|---|
| `0` | `hand` | array of tiles | Tiles left in the hand |
| `1` | `action` | array of tiles | Tiles played from the hand |
| `2` | `sBoard` | array of arrays of tiles | Every set on the table after the move |
| `3` | `numOfTails` | integer | Number of tiles played |

**When no tile can be played:**

```json
{
  "greedy_result": "Draw a tile",
  "max_q_result": ""
}
```

**Tiles in JSON:**

| Tile | Python | JSON |
|---|---|---|
| Number tile | `(5, "red")` | `[5, "red"]` |
| Joker | `"J"` | `"J"` |

**Overall shape:**

```text
{
  "greedy_result": [hand, action, sBoard, numOfTails]  |  "Draw a tile",
  "max_q_result":  ""
}

hand, action  : [tile, ...]
sBoard        : [[tile, tile, tile, ...], ...]
numOfTails    : integer
tile          : [number, "color"]  |  "J"
number        : 1..13
color         : "red" | "yellow" | "blue" | "black"
```

JSON has no tuples or named tuples. The move tuple and each tile tuple therefore become arrays, and the field names `hand`, `action`, `sBoard` and `numOfTails` exist only as positions `0`–`3`. To pass a JSON tile back into `get_move`, convert it to a tuple again: `[5, "red"]` → `(5, "red")`.

### Per key

| Key | Value |
|---|---|
| `"greedy_result"` | Always a `<move>`: the play placing the most hand tiles. Ties are broken toward shedding the most penalty points (number value; joker = 30). |
| `"max_q_result"` | Currently always `''`. Once the max-Q model is integrated it will hold a `<move>` in the same format. |

---

## Errors

| Exception | Raised when |
|---|---|
| `TypeError` | `hand` or `board` is not a `list` |
| `ValueError` | A tile is not in the [tile format](#tile-format); a tile or the joker appears more than twice across hand and board; or the board cannot be split into legal sets |
| `RuntimeError` | The MILP solver behind the greedy policy fails |

---

## Examples

**Ordinary turn: extending a run**

```python
>>> moves = get_move([(4, "red"), (9, "blue")], [(5, "red"), (6, "red"), (7, "red")])
>>> hand, action, sBoard, numOfTails = moves["greedy_result"]
>>> hand
[(9, 'blue')]
>>> action
[(4, 'red')]
>>> sBoard
[[(4, 'red'), (5, 'red'), (6, 'red'), (7, 'red')]]
>>> numOfTails
1
>>> moves["max_q_result"]
''
```

**Initial meld: a board tile cannot help**

```python
>>> get_move([(13, "red")], [(10, "red"), (11, "red"), (12, "red")], first_meld=True)
{'greedy_result': 'Draw a tile', 'max_q_result': ''}
```

**Illegal board**

```python
>>> get_move([(7, "red")], [(5, "red"), (6, "red")])
Traceback (most recent call last):
  ...
ValueError: board tiles cannot be partitioned into valid sets
```

---

## Setup

- **Requirements:** Python 3, PuLP 4.2 or newer, and `highspy` (the HiGHS solver).

  ```bash
  pip install "pulp>=4.2" highspy
  ```
