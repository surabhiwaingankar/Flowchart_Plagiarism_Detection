import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Initialize global TF-IDF Vectorizer to avoid redundant instantiation
vectorizer = TfidfVectorizer()

def build_graph(shapes, edges):
    """
    Build a directed graph from shapes and edges.

    Args:
        shapes (list): List of dictionaries containing shape attributes.
        edges (list): List of dictionaries containing edge connections.

    Returns:
        nx.DiGraph: A directed graph representing the flowchart.
    """
    G = nx.DiGraph()

    # Add nodes
    for shape in shapes:
        G.add_node(shape['id'], type=shape['type'], center=shape['center'], text=shape['text'])

    # Add edges
    for edge in edges:
        G.add_edge(edge['source'], edge['target'])

    # # Visualize the graph
    # pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1500, font_size=10)
    # plt.title("Graph Visualization")
    # plt.show()

    return G

def text_similarity(text1, text2):
    """
    Compute cosine similarity between two texts using TF-IDF.

    Args:
        text1 (str): First text.
        text2 (str): Second text.

    Returns:
        float: Cosine similarity between the two texts.
    """
    tfidf = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

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
    type_sim = 1 if node1['type'] == node2['type'] else 0

    # Compare text (cosine similarity)
    text_sim = text_similarity(node1['text'], node2['text'])

    # Compare spatial position (Euclidean distance between centers)
    center1, center2 = node1['center'], node2['center']
    spatial_dist = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
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
    # print("Node similarity:", node_sim)
    # print("Edge similarity:", edge_sim)

    return 0.6 * node_sim + 0.4 * edge_sim

def overall_similarity(graph1, graph2):
    """
    Compute overall similarity between two graphs, combining structural and textual similarity.

    Args:
        graph1 (nx.DiGraph): First graph.
        graph2 (nx.DiGraph): Second graph.

    Returns:
        float: Overall similarity score.
    """
    # Structural similarity (includes textual similarity in node_similarity)
    return graph_similarity(graph1, graph2)

# # Main Workflow
# graph1 = build_graph(shapes1, edges1)
# graph2 = build_graph(shapes2, edges2)

# # Calculate similarity score
# similarity_score = overall_similarity(graph1, graph2)

# print("Similarity Score:", similarity_score)

# if(similarity_score>0.7):
#     print("Graphs are similar")
# else:
#     print("Graphs are not similar")