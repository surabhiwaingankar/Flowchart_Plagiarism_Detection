import os
import cv2
import numpy as np
from ultralytics import YOLO


def inspect_raw_detections(test_image_path):
    if not isinstance(test_image_path, str):
        raise ValueError("Expected a file path, but received a non-string input.")

    # Load YOLO model
    model = YOLO("functionality/model.pt")
    detections = model.predict(test_image_path, conf=0.3)[0]

    print("=== RAW DETECTION INSIGHTS ===")
    print(f"Detected {len(detections.boxes)} elements in image:")
    for i, box in enumerate(detections.boxes):
        coords = box.xyxy[0].cpu().numpy()
        class_name = model.names[int(box.cls)]
        conf = box.conf.item()
        print(f"Element {i+1}:")
        print(f"  Class: {class_name} ({conf:.2f} confidence)")
        print(f"  Coordinates: {coords}")
        print(f"  Center: ({np.mean(coords[0::2]):.1f}, {np.mean(coords[1::2]):.1f})")
        print("-" * 50)

    # Create a new file path for saving the result
    file_dir, file_name = os.path.split(test_image_path)
    file_name_wo_ext, ext = os.path.splitext(file_name)
    result_image_path = os.path.join(file_dir, f"{file_name_wo_ext}_predictions{ext}")

    # Save the annotated image
    annotated_image = detections.plot()
    cv2.imwrite(result_image_path, annotated_image)
    print(f"Predictions saved to {result_image_path}")

    return detections, result_image_path
