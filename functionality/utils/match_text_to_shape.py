import cv2
import easyocr
import numpy as np

def shape_text_matching(test_image_path, shapes):
    reader = easyocr.Reader(["en"])
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
        text_center = (np.mean(bbox_np[:, 0]), np.mean(bbox_np[:, 1]))

        if text.strip().lower() in {"yes", "no", "start", "stop"}:
            min_conf = 0.3
        else:
            min_conf = 0.45

        if conf >= min_conf:
            filtered_ocr.append((bbox, text, conf, text_center))

    print(
        f"Using {len(filtered_ocr)}/{len(ocr_results)} texts after adaptive filtering"
    )

    matched_texts = set()
    shapes_sorted = sorted(shapes, key=lambda x: x["center"][1])

    for shape in shapes_sorted:
        shape_coords = shape["coords"]
        best_match = None
        min_distance = float("inf")

        for bbox, text, conf, text_center in filtered_ocr:
            if text in matched_texts:
                continue

            distance = np.linalg.norm(np.array(shape["center"]) - text_center)
            inside_bonus = 0.8 if is_inside(shape_coords, text_center) else 1.0

            if shape["type"] == "decision":
                if text_center[1] > shape["center"][1]:
                    distance *= 0.7
            elif shape["type"] == "process":
                if abs(shape["center"][0] - text_center[0]) > 30:
                    continue

            weighted_distance = distance * inside_bonus

            if weighted_distance < min_distance:
                min_distance = weighted_distance
                best_match = (text, weighted_distance)

        if best_match and best_match[1] < 100:
            shape["text"] = best_match[0]
            matched_texts.add(best_match[0])
        else:
            for bbox, text, conf in ocr_results:
                bbox_np = np.array(bbox)
                text_center = (np.mean(bbox_np[:, 0]), np.mean(bbox_np[:, 1]))

                if (
                    text not in matched_texts
                    and is_inside(shape_coords, text_center)
                    and conf >= 0.3
                ):
                    shape["text"] = text
                    matched_texts.add(text)
                    break
            else:
                shape["text"] = "No text"

        print(f"{shape['type'].upper()}@{shape['center']}: {shape['text']}")

    print("\nValidated Matches:")
    for shape in shapes_sorted:
        print(f"- {shape['type']}: {shape['text']}")

    return shapes_sorted
