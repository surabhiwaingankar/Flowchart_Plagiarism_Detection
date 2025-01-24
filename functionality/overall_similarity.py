from functionality.utils.graph_comparison import node_similarity, edge_similarity


def graph_similarity(graph1, graph2):
    """
    Compute structural similarity between two graphs.

    Args:
        graph1 (nx.DiGraph): First graph.
        graph2 (nx.DiGraph): Second graph.

    Returns:
        float: Structural similarity score.
    """
    # Node similarity
    node_sim = 0
    matched_nodes = 0
    for node1 in graph1.nodes:
        for node2 in graph2.nodes:
            sim = node_similarity(graph1.nodes[node1], graph2.nodes[node2])
            if sim > 0.7:  # Threshold for matching nodes
                node_sim += sim
                matched_nodes += 1
    node_sim = node_sim / matched_nodes if matched_nodes > 0 else 0

    # Edge similarity
    edge_sim = edge_similarity(graph1, graph2)

    # Weighted combination of node and edge similarity
    print("Node similarity:", node_sim)
    print("Edge similarity:", edge_sim)

    return 0.6 * node_sim + 0.4 * edge_sim


# # Main Workflow
# graph1 = build_graph(shapes1, edges1)
# graph2 = build_graph(shapes2, edges2)

# # Calculate similarity score
# similarity_score = graph_similarity(graph1, graph2)

# print("Similarity Score:", similarity_score)

# if(similarity_score>0.7):
#     print("Graphs are similar")
# else:
#     print("Graphs are not similar")
