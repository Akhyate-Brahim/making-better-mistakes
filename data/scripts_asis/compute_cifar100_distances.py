import pickle
import lzma
import networkx as nx
from nltk import Tree

# Load the NLTK tree from the pickle file
with open("../cifar100_tree.pkl", "rb") as f:
    nltk_tree = pickle.load(f)

# Convert the NLTK tree to a NetworkX graph and calculate heights
def tree_to_graph_and_heights(tree):
    graph = nx.DiGraph()
    heights = {}

    def add_edges(node, height):
        if isinstance(node, Tree):
            for child in node:
                graph.add_edge(node.label(), child.label() if isinstance(child, Tree) else child)
                add_edges(child, height + 1)
        heights[node.label() if isinstance(node, Tree) else node] = height

    add_edges(tree, 0)
    return graph, heights

if isinstance(nltk_tree, Tree):
    graph, heights = tree_to_graph_and_heights(nltk_tree)
else:
    graph = nltk_tree

# Calculate the height of the LCA
def lca_height(graph, heights, node1, node2):
    ancestors1 = nx.ancestors(graph, node1) | {node1}
    ancestors2 = nx.ancestors(graph, node2) | {node2}
    common_ancestors = ancestors1 & ancestors2
    if not common_ancestors:
        return None
    return max(heights[ancestor] for ancestor in common_ancestors)

# Get all nodes
nodes = list(graph.nodes())

# Calculate distances
distances = {}
for i, node1 in enumerate(nodes):
    for node2 in nodes[i+1:]:
        distance = lca_height(graph, heights, node1, node2)
        if distance is not None:
            distances[(node1, node2)] = distance
            distances[(node2, node1)] = distance

# Add zero distances for self-loops
for node in nodes:
    distances[(node, node)] = heights[node]

# Save distances to a compressed pickle file
distances_path = '../cifar100_ilsvrc_distances.pkl.xz'
with lzma.open(distances_path, "wb") as f:
    pickle.dump(distances, f)

print(f"Distances saved to {distances_path}")
