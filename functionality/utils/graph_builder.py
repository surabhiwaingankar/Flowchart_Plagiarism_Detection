import networkx as nx
# import matplotlib.pyplot as plt

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