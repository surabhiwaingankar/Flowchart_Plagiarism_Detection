def find_start_node(sorted_shapes):
    for i, s in enumerate(sorted_shapes):
        if s['type'] == 'terminal' and 'start' in s['text'].lower():
            return i

    terminals = [i for i, s in enumerate(sorted_shapes) if s['type'] == 'terminal']
    if terminals:
        return min(terminals, key=lambda x: sorted_shapes[x]['center'][1])

    print("⚠️ No terminal nodes found, using first shape as start")
    return 0