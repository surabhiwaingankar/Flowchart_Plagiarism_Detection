import numpy as np

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