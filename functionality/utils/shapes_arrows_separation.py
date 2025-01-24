import numpy as np
from ultralytics import YOLO

model = YOLO("functionality/model.pt")

def separate_shapes_and_arrows(detections):
    shape_classes = ["terminal", "decision", "process", "input_output", "print"]
    arrow_classes = ["down arrow", "up arrow", "left arrow", "right arrow"]

    shapes = []
    arrows = []

    for box in detections.boxes:
        class_name = model.names[int(box.cls)]
        element = {
            "type": class_name,
            "coords": box.xyxy[0].cpu().numpy(),
            "center": (
                np.mean(box.xyxy[0].cpu().numpy()[0::2]),
                np.mean(box.xyxy[0].cpu().numpy()[1::2]),
            ),
            "text": None,
        }

        if class_name in shape_classes:
            shapes.append(element)
        elif class_name in arrow_classes:
            arrows.append(element)

    print("=== SHAPE/ARROW SEPARATION ===")
    print(f"Found {len(shapes)} shapes:")
    for shape in shapes:
        print(f"  {shape['type']} at {shape['center']}")

    print(f"\nFound {len(arrows)} arrows:")
    for arrow in arrows:
        print(f"  {arrow['type']} at {arrow['center']}")

    return shapes, arrows
