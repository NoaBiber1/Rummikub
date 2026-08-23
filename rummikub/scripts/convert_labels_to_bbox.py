from pathlib import Path

LABELS_DIR = Path("rummikub/dataset/labels")

converted = 0
unchanged = 0

for label_file in LABELS_DIR.rglob("*.txt"):
    new_lines = []

    for line in label_file.read_text().splitlines():
        if not line.strip():
            continue

        parts = line.split()

        # Already in YOLO bounding-box format:
        # class_id x_center y_center width height
        if len(parts) == 5:
            new_lines.append(line)
            unchanged += 1
            continue

        class_id = parts[0]
        coordinates = list(map(float, parts[1:]))

        # Polygon format: x1 y1 x2 y2 x3 y3 ...
        xs = coordinates[0::2]
        ys = coordinates[1::2]

        x_min = min(xs)
        x_max = max(xs)
        y_min = min(ys)
        y_max = max(ys)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min

        new_line = (
            f"{class_id} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{width:.6f} "
            f"{height:.6f}"
        )

        new_lines.append(new_line)
        converted += 1

    label_file.write_text("\n".join(new_lines) + "\n")

print("Conversion completed.")
print(f"Converted polygon annotations: {converted}")
print(f"Existing bounding boxes unchanged: {unchanged}")