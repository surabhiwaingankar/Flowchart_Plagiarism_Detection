from ultralytics import YOLO

def inspect_raw_detections(dataset_path, test_image_path):
    model = YOLO(f'{dataset_path}/runs/detect/train/weights/best.pt')
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
        print("-"*50)

    return detections