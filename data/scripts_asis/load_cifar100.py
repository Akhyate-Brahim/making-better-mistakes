import argparse
import pickle
from nltk import Tree

def load_tree(file):
    with open(file, 'rb') as f:
        tree = pickle.load(f)
    return tree

def display_tree_info(tree):
    print("Tree structure:")
    print_tree(tree, 0)
    print("\nNumber of leaf nodes:", len(tree.leaves()))
    print("Tree depth:", tree.height())
    print("Number of subtrees:", len(list(tree.subtrees())))

def print_tree(tree, level):
    if isinstance(tree, str):
        print("  " * level + "- " + tree)
    else:
        print("  " * level + "- " + tree.label())
        for child in tree:
            print_tree(child, level + 1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Display information about the hierarchical tree.')
    parser.add_argument('--input', type=str, required=True, help='Path to the input pickle file')
    args = parser.parse_args()

    tree = load_tree(args.input)
    display_tree_info(tree)