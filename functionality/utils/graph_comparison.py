from functionality.utils.graph_builder import build_graph
from functionality.utils.textual_similarity import text_similarity


def node_similarity(node1, node2):
    """
    Compute similarity between two nodes based on their attributes.

    Args:
        node1 (dict): Attributes of the first node.
        node2 (dict): Attributes of the second node.

    Returns:
        float: Similarity score between the two nodes.
    """
    # Compare shape type (exact match)
    type_sim = 1 if node1["type"] == node2["type"] else 0

    # Compare text (cosine similarity)
    text_sim = text_similarity(node1["text"], node2["text"])

    # Compare spatial position (Euclidean distance between centers)
    center1, center2 = node1["center"], node2["center"]
    spatial_dist = (
        (center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2
    ) ** 0.5
    spatial_sim = 1 / (1 + spatial_dist)  # Normalize to [0, 1]

    # Weighted combination of similarities
    return 0.4 * type_sim + 0.5 * text_sim + 0.1 * spatial_sim


def edge_similarity(graph1, graph2):
    """
    Compute similarity between edges of two graphs.

    Args:
        graph1 (nx.DiGraph): First graph.
        graph2 (nx.DiGraph): Second graph.

    Returns:
        float: Edge similarity score.
    """
    common_edges = len(set(graph1.edges) & set(graph2.edges))
    total_edges = max(len(graph1.edges), len(graph2.edges))
    return common_edges / total_edges if total_edges > 0 else 0
