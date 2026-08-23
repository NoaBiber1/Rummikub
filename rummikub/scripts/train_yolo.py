from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolo11n.pt")

model.train(
    data="rummikub/config/data.yaml",
    epochs=50,
    imgsz=640,

    # Data Augmentation
    degrees=10,       # small rotations
    translate=0.05,   # small position changes
    scale=0.15,       # zoom in/out

    hsv_h=0.0,        # do not change hue - tile color is important
    hsv_s=0.05,       # very small saturation change
    hsv_v=0.20,       # lighting / brightness variations

    fliplr=0.0,       # no horizontal flip
    flipud=0.0,       # no vertical flip

    mosaic=0.0,
    mixup=0.0
)