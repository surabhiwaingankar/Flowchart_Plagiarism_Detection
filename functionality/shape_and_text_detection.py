import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ultralytics import YOLO
import easyocr
from collections import deque

def visualize_predictions(dataset_path):
    results_path = f'{dataset_path}/runs/detect/predict'
    images = [f for f in os.listdir(results_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    plt.figure(figsize=(20, 15))
    for idx, img_name in enumerate(images[:6]):
        img_path = os.path.join(results_path, img_name)
        img = mpimg.imread(img_path)
        plt.subplot(2, 3, idx+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Pred: {img_name}", fontsize=8)
    plt.tight_layout()
    plt.show()

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

def separate_shapes_and_arrows(detections, model):
    shape_classes = ['terminal', 'decision', 'process', 'input_output', 'print']
    arrow_classes = ['down arrow', 'up arrow', 'left arrow', 'right arrow']

    shapes = []
    arrows = []

    for box in detections.boxes:
        class_name = model.names[int(box.cls)]
        element = {
            'type': class_name,
            'coords': box.xyxy[0].cpu().numpy(),
            'center': (np.mean(box.xyxy[0].cpu().numpy()[0::2]),
                      np.mean(box.xyxy[0].cpu().numpy()[1::2])),
            'text': None
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

def context_aware_text_matching(dataset_path, test_image_path, shapes):
    reader = easyocr.Reader(['en'])
    image = cv2.imread(test_image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ocr_results = reader.readtext(gray_image, paragraph=False)

    print("\n=== CONTEXT-AWARE MATCHING ===")

    def is_inside(shape_coords, text_center):
        x_min, y_min, x_max, y_max = shape_coords
        return (x_min < text_center[0] < x_max) and (y_min < text_center[1] < y_max)

    filtered_ocr = []
    for bbox, text, conf in ocr_results:
        bbox_np = np.array(bbox)
        text_center = (np.mean(bbox_np[:,0]), np.mean(bbox_np[:,1]))

        if text.strip().lower() in {'yes', 'no', 'start', 'stop'}:
            min_conf = 0.3
        else:
            min_conf = 0.45

        if conf >= min_conf:
            filtered_ocr.append((bbox, text, conf, text_center))

    print(f"Using {len(filtered_ocr)}/{len(ocr_results)} texts after adaptive filtering")

    matched_texts = set()
    shapes_sorted = sorted(shapes, key=lambda x: x['center'][1])

    for shape in shapes_sorted:
        shape_coords = shape['coords']
        best_match = None
        min_distance = float('inf')

        for bbox, text, conf, text_center in filtered_ocr:
            if text in matched_texts:
                continue

            distance = np.linalg.norm(np.array(shape['center']) - text_center)
            inside_bonus = 0.8 if is_inside(shape_coords, text_center) else 1.0

            if shape['type'] == 'decision':
                if text_center[1] > shape['center'][1]:
                    distance *= 0.7
            elif shape['type'] == 'process':
                if abs(shape['center'][0] - text_center[0]) > 30:
                    continue

            weighted_distance = distance * inside_bonus

            if weighted_distance < min_distance:
                min_distance = weighted_distance
                best_match = (text, weighted_distance)

        if best_match and best_match[1] < 100:
            shape['text'] = best_match[0]
            matched_texts.add(best_match[0])
        else:
            for bbox, text, conf in ocr_results:
                bbox_np = np.array(bbox)
                text_center = (np.mean(bbox_np[:,0]), np.mean(bbox_np[:,1]))

                if (text not in matched_texts and
                    is_inside(shape_coords, text_center) and
                    conf >= 0.3):
                    shape['text'] = text
                    matched_texts.add(text)
                    break
            else:
                shape['text'] = "No text"

        print(f"{shape['type'].upper()}@{shape['center']}: {shape['text']}")

    print("\nValidated Matches:")
    for shape in shapes_sorted:
        print(f"- {shape['type']}: {shape['text']}")

    return shapes_sorted

def enhanced_arrow_connection_analysis(shapes, arrows):
    print("\n=== ENHANCED ARROW CONNECTION ANALYSIS ===")

    for shape in shapes:
        shape['center'] = (float(shape['center'][0]), float(shape['center'][1]))
    for arrow in arrows:
        arrow['center'] = (float(arrow['center'][0]), float(arrow['center'][1]))

    sorted_shapes = sorted(shapes, key=lambda x: (x['center'][1], -int(x['type'] == 'decision')))
    sorted_arrows = sorted(arrows, key=lambda x: x['center'][1])

    adjacency = {i: [] for i in range(len(sorted_shapes))}
    shape_details = []

    for idx, shape in enumerate(sorted_shapes):
        shape_details.append({
            'id': idx + 1,
            'type': shape['type'],
            'center': (round(shape['center'][0], 1), round(shape['center'][1], 1)),
            'text': shape['text']
        })

    print("\nTemporary ID Assignment:")
    for s in shape_details:
        print(f"ID {s['id']}: {s['type'].upper()} ({s['text']})")

    for arrow in sorted_arrows:
        arrow_center = np.array(arrow['center'])

        sources = [s for s in sorted_shapes if s['center'][1] < arrow_center[1]]
        if not sources:
            continue

        source = min(sources, key=lambda x: np.linalg.norm(np.array(x['center']) - arrow_center))

        try:
            src_idx = next(
                i for i, s in enumerate(sorted_shapes)
                if s['type'] == source['type']
                and np.allclose(np.array(s['center']), np.array(source['center']), atol=0.1)
                and s['text'] == source['text']
            )
        except StopIteration:
            print(f"⚠️ Source not found for arrow at {tuple(arrow_center)}")
            continue

        if source['type'] == 'decision':
            dec_x = source['center'][0]
            if arrow_center[0] < dec_x:
                targets = [s for s in sorted_shapes if s['center'][1] > arrow_center[1] and s['center'][0] < dec_x]
            else:
                targets = [s for s in sorted_shapes if s['center'][1] > arrow_center[1] and s['type'] in ['process', 'input_output']]
        else:
            targets = [s for s in sorted_shapes if s['center'][1] > arrow_center[1]]

        if targets:
            target = min(targets, key=lambda x: np.linalg.norm(np.array(x['center']) - arrow_center))
            try:
                tgt_idx = next(
                    i for i, s in enumerate(sorted_shapes)
                    if s['type'] == target['type']
                    and np.allclose(np.array(s['center']), np.array(target['center']), atol=0.1)
                    and s['text'] == target['text']
                )
            except StopIteration:
                print(f"⚠️ Target not found for arrow at {tuple(arrow_center)}")
                continue

            adjacency[src_idx].append(tgt_idx)
            print(f"\n{arrow['type']} at {tuple(np.round(arrow_center, 1))} connects:")
            print(f"  Source: {source['type']} ({source['text']})")
            print(f"  Target: {target['type']} ({target['text']})")

    return adjacency, shape_details, sorted_shapes

def find_start_node(sorted_shapes):
    for i, s in enumerate(sorted_shapes):
        if s['type'] == 'terminal' and 'start' in s['text'].lower():
            return i

    terminals = [i for i, s in enumerate(sorted_shapes) if s['type'] == 'terminal']
    if terminals:
        return min(terminals, key=lambda x: sorted_shapes[x]['center'][1])

    print("⚠️ No terminal nodes found, using first shape as start")
    return 0

def final_flowchart_structure(adjacency, shape_details, sorted_shapes):
    print("\n=== FINAL FLOWCHART STRUCTURE ===")

    start_node = find_start_node(sorted_shapes)

    final_ids = {}
    current_id = 1
    queue = deque([start_node])

    while queue:
        node = queue.popleft()
        if node not in final_ids:
            final_ids[node] = current_id
            current_id += 1

            if sorted_shapes[node]['type'] == 'decision':
                neighbors = sorted(adjacency[node], key=lambda x: sorted_shapes[x]['center'][0], reverse=True)
            else:
                neighbors = sorted(adjacency[node], key=lambda x: sorted_shapes[x]['center'][0])

            queue.extend(neighbors)

    for orig_id, new_id in final_ids.items():
        shape_details[orig_id]['id'] = new_id

    final_shapes = sorted(shape_details, key=lambda x: x['id'])

    edges = []
    for src, targets in adjacency.items():
        if src in final_ids:
            for tgt in targets:
                if tgt in final_ids:
                    edges.append({
                        'source': final_ids[src],
                        'target': final_ids[tgt]
                    })

    print("\nDetected Shapes:")
    for shape in final_shapes:
        print(f"ID {shape['id']}: {shape['type']} | Center: {shape['center']} | Text: {shape['text']}")

    print("\nGenerated Edges:")
    for edge in sorted(edges, key=lambda x: x['source']):
        src_text = next(s['text'] for s in final_shapes if s['id'] == edge['source'])
        tgt_text = next(s['text'] for s in final_shapes if s['id'] == edge['target'])
        print(f"Source: {edge['source']} ({src_text}) → Target: {edge['target']} ({tgt_text})")

# def main():
#     dataset_path = '/content/drive/MyDrive/dataset'
#     test_image_path = f'{dataset_path}/test/1.png'

#     visualize_predictions(dataset_path)
#     detections = inspect_raw_detections(dataset_path, test_image_path)
#     shapes, arrows = separate_shapes_and_arrows(detections, YOLO(f'{dataset_path}/runs/detect/train/weights/best.pt'))
#     shapes_sorted = context_aware_text_matching(dataset_path, test_image_path, shapes)
#     adjacency, shape_details, sorted_shapes = enhanced_arrow_connection_analysis(shapes_sorted, arrows)
#     final_flowchart_structure(adjacency, shape_details, sorted_shapes)

# if __name__ == "__main__":
#     main()