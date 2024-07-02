import argparse
import pickle
from collections import defaultdict
from nltk import Tree
import networkx as nx

def load_parent_child(file):
    parent_child = []
    with open(file, 'r') as f:
        for line in f:
            parent, child = line.strip().split()
            parent_child.append((int(parent), int(child)))
    return parent_child

def build_hierarchy_tree(parent_child):
    # Build directed graph using NetworkX
    graph = nx.DiGraph()
    for parent, child in parent_child:
        graph.add_edge(parent, child)

    # Find root node(s)
    roots = [node for node in graph.nodes() if graph.in_degree(node) == 0]

    # Build NLTK tree recursively
    def build_tree(node):
        children = list(graph.successors(node))
        if children:
            return Tree(str(node), [build_tree(child) for child in children])
        else:
            return str(node)

    nltk_tree = Tree('root', [build_tree(root) for root in roots])
    return nltk_tree

def save_tree(nltk_tree, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(nltk_tree, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build hierarchical tree from parent-child pairs.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input .txt file')
    parser.add_argument('--output', type=str, required=True, help='Path to the output pickle file')
    args = parser.parse_args()

    parent_child = load_parent_child(args.input)
    nltk_tree = build_hierarchy_tree(parent_child)
    save_tree(nltk_tree, args.output)
    print('Tree generated and saved successfully.')