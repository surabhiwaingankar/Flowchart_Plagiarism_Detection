from collections import deque
from functionality.utils.find_start_node import find_start_node

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