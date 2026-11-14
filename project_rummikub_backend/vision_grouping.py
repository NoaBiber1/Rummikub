"""Geometry-first grouping for Roboflow Rummikub tile detections.

The module has no FastAPI dependency.  ``group_vision_detections`` accepts
either a prediction list or a complete Roboflow workflow response and returns
JSON-serialisable data.  It deliberately retains low-confidence, unknown, and
malformed detections; detections without usable geometry are reported as
``unassigned``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from models import Set, SetType, Tile, TileColor
from validator import validate_board

_LABEL_RE = re.compile(
    r"^(red|blue|black|yellow|orange)(\d{1,2})$", re.IGNORECASE
)
_JOKER_RE = re.compile(r"^jokers?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class TileIdentity:
    color: TileColor
    value: int


@dataclass(frozen=True, slots=True)
class Detection:
    detection_id: str
    source_index: int
    label: str
    confidence: float
    x: float | None
    y: float | None
    width: float | None
    height: float | None
    tile: TileIdentity | None
    parse_warnings: tuple[str, ...] = ()

    @property
    def has_geometry(self) -> bool:
        return (
            self.x is not None
            and self.y is not None
            and self.width is not None
            and self.height is not None
            and self.width > 0
            and self.height > 0
        )

    @property
    def left(self) -> float:
        assert self.x is not None and self.width is not None
        return self.x - self.width / 2

    @property
    def right(self) -> float:
        assert self.x is not None and self.width is not None
        return self.x + self.width / 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.detection_id,
            "source_index": self.source_index,
            "label": self.label,
            "confidence": self.confidence,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "color": self.tile.color.value if self.tile else None,
            "value": self.tile.value if self.tile else None,
            "recognized": self.tile is not None,
            "parse_warnings": list(self.parse_warnings),
        }


@dataclass(frozen=True, slots=True)
class GroupingOptions:
    """Tuning inputs in source-image coordinates.

    ``rack_boundary_y`` is the most reliable way to separate rack and board.
    When absent, rows in the bottom 28% of a known image are rack rows.  If the
    image height is also unknown, a large vertical gap may infer the split.
    """

    image_height: float | None = None
    rack_boundary_y: float | None = None
    rack_bottom_fraction: float = 0.28
    row_tolerance: float = 0.70
    # Minimum board link tolerance above the tile pitch (see _group_board).
    pitch_tolerance: float = 0.15


@dataclass(slots=True)
class _Row:
    detections: list[Detection] = field(default_factory=list)

    @property
    def center_y(self) -> float:
        return median(d.y for d in self.detections if d.y is not None)

    @property
    def height(self) -> float:
        return median(d.height for d in self.detections if d.height is not None)


def parse_tile_label(raw_label: str) -> TileIdentity | None:
    compact = re.sub(r"[\s_-]+", "", raw_label.strip())
    if _JOKER_RE.fullmatch(compact):
        return TileIdentity(TileColor.JOKER, 0)
    match = _LABEL_RE.fullmatch(compact)
    if not match:
        return None
    value = int(match.group(2))
    if not 1 <= value <= 13:
        return None
    color_name = match.group(1).upper()
    if color_name == "ORANGE":
        color_name = "YELLOW"
    return TileIdentity(TileColor(color_name), value)


def parse_detections(payload: Any) -> list[Detection]:
    """Find and type tile detections in a list or nested workflow response."""
    raw_predictions = _find_best_prediction_list(payload)
    parsed: list[Detection] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(raw_predictions):
        detection = _parse_detection(raw, index)
        if detection.detection_id in used_ids:
            detection = replace(
                detection,
                detection_id=f"{detection.detection_id}-{index}",
                parse_warnings=(
                    *detection.parse_warnings,
                    "duplicate detection id; appended source index",
                ),
            )
        used_ids.add(detection.detection_id)
        parsed.append(detection)
    return parsed


def group_vision_detections(
    payload: Any,
    *,
    options: GroupingOptions | None = None,
) -> dict[str, Any]:
    """Group all detected boxes into board meld candidates and rack rows."""
    options = options or GroupingOptions()
    detections = parse_detections(payload)
    geometric = [d for d in detections if d.has_geometry]
    unassigned = [d for d in detections if not d.has_geometry]
    rows = _cluster_rows(geometric, options.row_tolerance)
    board_rows, rack_rows, rack_method = _separate_rack_rows(rows, options)

    # Only the rack/board separation is row based; board tiles are grouped
    # geometrically so melds may lie at any angle.
    board_detections = [d for row in board_rows for d in row.detections]
    board_melds, board_component_count = _group_board(board_detections, options)

    return {
        "detections": [d.to_dict() for d in detections],
        "board_melds": board_melds,
        "rack_rows": [
            {
                "row_index": index,
                "detection_ids": [
                    d.detection_id for d in sorted(row.detections, key=_x_key)
                ],
                "center_y": round(row.center_y, 4),
            }
            for index, row in enumerate(rack_rows)
        ],
        "unassigned_detection_ids": [d.detection_id for d in unassigned],
        "metadata": {
            "detection_count": len(detections),
            # Number of connected board components (candidate melds before any
            # perpendicular split).  The legacy key name is kept; it no longer
            # counts image rows.
            "board_row_count": board_component_count,
            "rack_row_count": len(rack_rows),
            "rack_classification": rack_method,
            "all_detections_retained": True,
        },
    }


def _parse_detection(raw: Mapping[str, Any], index: int) -> Detection:
    warnings: list[str] = []
    label = str(raw.get("class") or raw.get("class_name") or raw.get("className") or "")
    confidence = _number(raw.get("confidence", raw.get("confidence_score", 1.0)))
    if confidence is None:
        confidence = 1.0
        warnings.append("invalid confidence; defaulted to 1.0")
    if confidence > 1:
        confidence /= 100
    confidence = max(0.0, min(1.0, confidence))

    x, y, width, height = _geometry(raw)
    if x is None or y is None or width is None or height is None:
        warnings.append("missing geometry")
    elif width <= 0 or height <= 0:
        warnings.append("non-positive box dimensions")

    raw_id = raw.get("detection_id") or raw.get("id")
    detection_id = str(raw_id) if raw_id not in (None, "") else f"d{index}"
    return Detection(
        detection_id=detection_id,
        source_index=index,
        label=label,
        confidence=confidence,
        x=x,
        y=y,
        width=width,
        height=height,
        tile=parse_tile_label(label),
        parse_warnings=tuple(warnings),
    )


def _geometry(
    raw: Mapping[str, Any],
) -> tuple[float | None, float | None, float | None, float | None]:
    x = _number(raw.get("x"))
    y = _number(raw.get("y"))
    width = _number(raw.get("width"))
    height = _number(raw.get("height"))
    bbox = raw.get("bbox")
    if all(value is not None for value in (x, y, width, height)):
        return x, y, width, height
    if isinstance(bbox, Mapping):
        x = _number(bbox.get("x"))
        y = _number(bbox.get("y"))
        width = _number(bbox.get("width", bbox.get("w")))
        height = _number(bbox.get("height", bbox.get("h")))
        if all(value is not None for value in (x, y, width, height)):
            return x, y, width, height
        x1 = _number(bbox.get("x1", bbox.get("left")))
        y1 = _number(bbox.get("y1", bbox.get("top")))
        x2 = _number(bbox.get("x2", bbox.get("right")))
        y2 = _number(bbox.get("y2", bbox.get("bottom")))
        if all(value is not None for value in (x1, y1, x2, y2)):
            assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
            return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    return x, y, width, height


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _looks_like_prediction(raw: Mapping[str, Any]) -> bool:
    # Geometry can be absent in a partially failed workflow output. Retain the
    # item so callers can correct it rather than silently losing a detection.
    return any(key in raw for key in ("class", "class_name", "className", "class_id"))


def _prediction_lists(node: Any) -> Iterable[list[Mapping[str, Any]]]:
    if isinstance(node, list):
        mappings = [item for item in node if isinstance(item, Mapping)]
        predictions = [item for item in mappings if _looks_like_prediction(item)]
        if predictions:
            yield predictions
        for item in node:
            if isinstance(item, (Mapping, list)):
                yield from _prediction_lists(item)
    elif isinstance(node, Mapping):
        for key, value in node.items():
            if str(key).lower() in {"visualization", "image", "output_image"}:
                continue
            yield from _prediction_lists(value)


def _find_best_prediction_list(payload: Any) -> list[Mapping[str, Any]]:
    candidates = list(_prediction_lists(payload))
    if not candidates:
        return []
    # Prefer recognized tile labels, then the most complete prediction list.
    return max(
        candidates,
        key=lambda items: (
            sum(
                parse_tile_label(
                    str(item.get("class") or item.get("class_name") or item.get("className") or "")
                )
                is not None
                for item in items
            ),
            len(items),
        ),
    )


def _cluster_rows(detections: Sequence[Detection], tolerance: float) -> list[_Row]:
    rows: list[_Row] = []
    for detection in sorted(detections, key=lambda d: (d.y or 0, d.x or 0)):
        best: _Row | None = None
        best_distance = math.inf
        for row in rows:
            assert detection.y is not None and detection.height is not None
            scale = max(row.height, detection.height)
            distance = abs(detection.y - row.center_y) / scale
            if distance <= tolerance and distance < best_distance:
                best, best_distance = row, distance
        if best is None:
            rows.append(_Row([detection]))
        else:
            best.detections.append(detection)
    return sorted(rows, key=lambda row: row.center_y)


def _separate_rack_rows(
    rows: list[_Row], options: GroupingOptions
) -> tuple[list[_Row], list[_Row], str]:
    if not rows:
        return [], [], "none"
    boundary = options.rack_boundary_y
    method = "explicit_boundary"
    if boundary is None and options.image_height and options.image_height > 0:
        boundary = options.image_height * (1 - options.rack_bottom_fraction)
        method = "image_bottom_region"
    if boundary is not None:
        board = [row for row in rows if row.center_y < boundary]
        rack = [row for row in rows if row.center_y >= boundary]
        return board, rack, method

    if len(rows) >= 2:
        gaps = [
            rows[index + 1].center_y - rows[index].center_y
            for index in range(len(rows) - 1)
        ]
        split_index = max(range(len(gaps)), key=gaps.__getitem__)
        local_height = median(row.height for row in rows)
        # Keep inference conservative: a rack must be visibly detached and can
        # occupy at most the lower two clustered rows.
        lower_count = len(rows) - split_index - 1
        if gaps[split_index] >= 1.5 * local_height and lower_count <= 2:
            return (
                rows[: split_index + 1],
                rows[split_index + 1 :],
                "inferred_vertical_gap",
            )
    return rows, [], "not_inferred"


_MELD_SPREAD_RATIO = 0.6
_MELD_CLUSTER_RATIO = 0.5
# Upper clamp of the link tolerance; stacked rows sit ~1.5 pitches apart.
_MAX_PITCH_TOLERANCE = 0.30
# Fragments whose facing ends are this many pitches apart may be rejoined.
_REJOIN_RATIO = 1.6


def _center_dist(a: Detection, b: Detection) -> float:
    """Distance between two box centers."""
    return math.hypot((a.x or 0.0) - (b.x or 0.0), (a.y or 0.0) - (b.y or 0.0))


def _box_gap(a: Detection, b: Detection) -> float:
    """Euclidean distance between two axis-aligned boxes (0 if they touch)."""
    assert a.x is not None and a.y is not None
    assert a.width is not None and a.height is not None
    assert b.x is not None and b.y is not None
    assert b.width is not None and b.height is not None
    dx = max(0.0, abs(a.x - b.x) - (a.width + b.width) / 2)
    dy = max(0.0, abs(a.y - b.y) - (a.height + b.height) / 2)
    return math.hypot(dx, dy)


def _link_components(
    items: Sequence[Detection], threshold: float
) -> list[list[Detection]]:
    """Connected components of boxes whose centers are <= threshold apart."""
    parent = list(range(len(items)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if _center_dist(items[i], items[j]) <= threshold:
                parent[find(j)] = find(i)
    groups: dict[int, list[Detection]] = {}
    for index, item in enumerate(items):
        groups.setdefault(find(index), []).append(item)
    return list(groups.values())


def _principal_axis(items: Sequence[Detection]) -> tuple[float, float]:
    """Unit axis of the box centers with canonical sign (x > 0, else y > 0).

    Two tiles use the vector between them; three or more use the dominant
    eigenvector of the 2x2 covariance matrix.
    """
    xs = [d.x or 0.0 for d in items]
    ys = [d.y or 0.0 for d in items]
    if len(items) == 2:
        ax, ay = xs[1] - xs[0], ys[1] - ys[0]
        norm = math.hypot(ax, ay)
        if norm < 1e-12:
            return 1.0, 0.0
        ax, ay = ax / norm, ay / norm
    else:
        cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - cx) ** 2 for x in xs)
        syy = sum((y - cy) ** 2 for y in ys)
        sxy = sum((x - cx) * (y - cy) for x, y in zip(xs, ys))
        theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
        ax, ay = math.cos(theta), math.sin(theta)
    if abs(ax) < 1e-6:
        ax, ay = 0.0, abs(ay)
    elif ax < 0:
        ax, ay = -ax, -ay
    return ax, ay


def _perpendicular_clusters(
    items: Sequence[Detection], tile_long: float
) -> list[list[Detection]]:
    """Split a component of >=3 tiles into parallel lines if it is not one line."""
    ax, ay = _principal_axis(items)
    cx = sum(d.x or 0.0 for d in items) / len(items)
    cy = sum(d.y or 0.0 for d in items) / len(items)
    keyed = sorted(
        (
            (-ay * ((d.x or 0.0) - cx) + ax * ((d.y or 0.0) - cy), d.source_index, d)
            for d in items
        ),
        key=lambda t: (t[0], t[1]),
    )
    # The full perpendicular range (max - min) is compared with the threshold
    # rather than max |offset|: two touching rows of upright tiles are only one
    # tile height apart, i.e. +-0.5 * tile_long from the centroid, which
    # max |offset| <= 0.6 * tile_long would never split.
    if keyed[-1][0] - keyed[0][0] <= _MELD_SPREAD_RATIO * tile_long:
        return [list(items)]
    clusters: list[list[Detection]] = [[keyed[0][2]]]
    for previous, current in zip(keyed, keyed[1:]):
        if current[0] - previous[0] > _MELD_CLUSTER_RATIO * tile_long:
            clusters.append([])
        clusters[-1].append(current[2])
    return clusters


def _order_along_axis(items: Sequence[Detection]) -> tuple[list[Detection], float]:
    """Order tiles by projection on the principal axis; return (order, angle)."""
    if len(items) < 2:
        return list(items), 0.0
    ax, ay = _principal_axis(items)
    ordered = sorted(
        items,
        key=lambda d: ((d.x or 0.0) * ax + (d.y or 0.0) * ay, d.source_index),
    )
    angle = math.degrees(math.atan2(ay, ax)) % 180.0
    if angle >= 180.0:
        angle -= 180.0
    return ordered, angle


def _split_at_gap_peaks(
    ordered: Sequence[Detection], threshold: float
) -> list[list[Detection]]:
    """Cut an axis-ordered line where neighbouring centers exceed threshold.

    These are local gap peaks; the cut also breaks L-shaped chains.
    """
    if not ordered or threshold <= 0:
        return [list(ordered)]
    pieces: list[list[Detection]] = [[ordered[0]]]
    for previous, current in zip(ordered, ordered[1:]):
        if _center_dist(previous, current) > threshold:
            pieces.append([])
        pieces[-1].append(current)
    return pieces


def _is_valid_meld(items: Sequence[Detection]) -> bool:
    """True when the tiles form a valid set read forward or reversed."""
    if len(items) < 3 or any(d.tile is None for d in items):
        return False
    return any(
        _evaluate_order(order, candidate_type, "probe")[0]
        for candidate_type in (SetType.RUN, SetType.GROUP)
        for order in (list(items), list(reversed(items)))
    )


def _split_into_valid_sets(line: Sequence[Detection]) -> list[list[Detection]]:
    """Guardrail: cut an invalid line into the fewest valid sets, if possible.

    Mirrors ``SetValidator.splitIntoValidSets`` in the app (a run and a group
    touching on one line).
    """
    n = len(line)
    if n < 6 or _is_valid_meld(line):
        return [list(line)]
    # best[end]: fewest valid sets covering line[:end]; cut_at[end]: where the
    # last of those sets starts.
    best: list[int | None] = [0] + [None] * n
    cut_at = [0] * (n + 1)
    for end in range(3, n + 1):
        for start in range(end - 2):
            before = best[start]
            if before is None or not _is_valid_meld(line[start:end]):
                continue
            current = best[end]
            if current is None or before + 1 < current:
                best[end] = before + 1
                cut_at[end] = start
    if best[n] is None:
        return [list(line)]
    pieces: list[list[Detection]] = []
    end = n
    while end > 0:
        pieces.insert(0, list(line[cut_at[end] : end]))
        end = cut_at[end]
    return pieces


def _rejoin_fragments(
    parts: list[tuple[int, list[Detection]]], reach: float
) -> None:
    """Guardrail: rejoin end-to-end fragments when only the union is valid.

    Catches a sloppy gap inside a real meld.  Edits ``parts`` in place; a
    merged part keeps the first fragment's component index.
    """
    merged = True
    while merged:
        merged = False
        for a, (component_index, first) in enumerate(parts):
            for b, (_other_index, second) in enumerate(parts):
                if a == b or _center_dist(first[-1], second[0]) > reach:
                    continue
                if _is_valid_meld(first) and _is_valid_meld(second):
                    continue
                union, _angle = _order_along_axis([*first, *second])
                if not _is_valid_meld(union):
                    continue
                parts[a] = (component_index, union)
                del parts[b]
                merged = True
                break
            if merged:
                break


def _group_board(
    detections: Sequence[Detection], options: GroupingOptions
) -> tuple[list[dict[str, Any]], int]:
    """Rotation-aware grouping of board boxes into meld candidates.

    Tiles are linked when their centers are at most about one tile pitch apart
    (the median nearest-neighbour center distance, which does not depend on
    rotation or box padding); components that contain several parallel lines
    are split along the perpendicular axis, and each line is cut where
    neighbouring centers are farther apart than the link threshold.  Two
    rule-based guardrails follow: an invalid line that cuts into valid sets is
    cut, and end-to-end fragments whose union is a valid set are rejoined.
    Every meld is ordered along its principal axis and validated forward and
    reversed.

    Returns ``(melds, component_count)``.
    """
    if not detections:
        return [], 0
    pitch = 0.0
    if len(detections) == 1:
        components = [list(detections)]
        tile_long = 0.0
        link_threshold = 0.0
    else:
        tile_short = median(min(d.width or 0.0, d.height or 0.0) for d in detections)
        tile_long = median(max(d.width or 0.0, d.height or 0.0) for d in detections)
        nearest = [
            min(_center_dist(d, other) for other in detections if other is not d)
            for d in detections
        ]
        # The floor guards against duplicate boxes sharing one center.
        pitch = max(median(nearest), 0.5 * tile_short)
        mad = median(abs(distance - pitch) for distance in nearest)
        tolerance = min(
            max(options.pitch_tolerance, 3 * mad / pitch),
            max(_MAX_PITCH_TOLERANCE, options.pitch_tolerance),
        )
        link_threshold = pitch * (1 + tolerance)
        components = _link_components(detections, link_threshold)

    # (component index, axis-ordered tiles) of every meld candidate.
    parts: list[tuple[int, list[Detection]]] = []
    for component_index, component in enumerate(components):
        clusters = (
            _perpendicular_clusters(component, tile_long)
            if len(component) >= 3
            else [component]
        )
        if len(clusters) == 1:
            lines = [component]
        else:
            lines = [
                line
                for cluster in clusters
                for line in _link_components(cluster, link_threshold)
            ]
        for line in lines:
            ordered, _angle = _order_along_axis(line)
            for piece in _split_at_gap_peaks(ordered, link_threshold):
                parts.extend(
                    (component_index, valid_piece)
                    for valid_piece in _split_into_valid_sets(piece)
                )
    if pitch > 0:
        _rejoin_fragments(parts, _REJOIN_RATIO * pitch)

    # (centroid_y, centroid_x, component index, ordered tiles, angle)
    entries: list[tuple[float, float, int, list[Detection], float]] = []
    for component_index, part in parts:
        ordered, angle = _order_along_axis(part)
        cx = sum(d.x or 0.0 for d in part) / len(part)
        cy = sum(d.y or 0.0 for d in part) / len(part)
        entries.append((cy, cx, component_index, ordered, angle))

    # Deterministic output order: centroid (y, x).  row_index is the rank of
    # the meld's connected component, meld_index its rank within the component.
    entries.sort(key=lambda e: (e[0], e[1]))
    component_rank: dict[int, int] = {}
    sub_counts: dict[int, int] = {}
    melds: list[dict[str, Any]] = []
    for _cy, _cx, component_index, ordered, angle in entries:
        row_index = component_rank.setdefault(component_index, len(component_rank))
        meld_index = sub_counts.get(component_index, 0)
        sub_counts[component_index] = meld_index + 1
        melds.append(
            _score_meld(
                ordered,
                row_index=row_index,
                meld_index=meld_index,
                angle_degrees=angle,
            )
        )
    return melds, len(components)


def _evaluate_order(
    detections: Sequence[Detection], candidate_type: SetType, set_id: str
) -> tuple[bool, str]:
    tiles = [
        Tile(id=d.detection_id, color=d.tile.color, value=d.tile.value)
        for d in detections
        if d.tile is not None
    ]
    return validate_board([Set(id=set_id, type=candidate_type, tiles=tiles)])


def _score_meld(
    detections: Sequence[Detection],
    *,
    row_index: int,
    meld_index: int,
    angle_degrees: float = 0.0,
) -> dict[str, Any]:
    """Validate a meld in axis order and reversed; keep the first valid order."""
    known = [d for d in detections if d.tile is not None]
    validation_applied = len(known) == len(detections)
    set_type = SetType.INVALID
    valid = False
    validation_message = "Validation unavailable: one or more labels are unknown."
    ordered = list(detections)
    if validation_applied:
        set_id = f"vision-{row_index}-{meld_index}"
        orders = [list(detections), list(reversed(detections))]
        validation_message = ""
        for candidate_type in (SetType.RUN, SetType.GROUP):
            for candidate_order in orders:
                candidate_valid, message = _evaluate_order(
                    candidate_order, candidate_type, set_id
                )
                if candidate_valid:
                    valid = True
                    set_type = candidate_type
                    ordered = candidate_order
                    validation_message = "Valid Rummikub set."
                    break
                validation_message = message
            if valid:
                break

    mean_confidence = (
        sum(d.confidence for d in ordered) / len(ordered) if ordered else 0
    )
    compactness = _compactness(ordered)
    score = 0.55 * mean_confidence + 0.20 * compactness + (0.25 if valid else 0)
    return {
        "row_index": row_index,
        "meld_index": meld_index,
        "detection_ids": [d.detection_id for d in ordered],
        "type": set_type.value,
        "is_valid": valid,
        "validation_applied": validation_applied,
        "validation_message": validation_message,
        "score": round(min(1.0, score), 4),
        "angle_degrees": round(angle_degrees, 4) if len(ordered) >= 2 else 0.0,
    }


def _compactness(detections: Sequence[Detection]) -> float:
    """1 / (1 + mean neighbour gap / tile width); rotation independent."""
    if len(detections) < 2:
        return 1.0
    typical_width = median(min(d.width or 0.0, d.height or 0.0) for d in detections)
    gaps = [
        _box_gap(detections[index], detections[index + 1])
        for index in range(len(detections) - 1)
    ]
    mean_gap = sum(gaps) / len(gaps)
    return 1 / (1 + mean_gap / max(typical_width, 1e-9))


def _x_key(detection: Detection) -> tuple[float, int]:
    return (detection.x or 0, detection.source_index)
