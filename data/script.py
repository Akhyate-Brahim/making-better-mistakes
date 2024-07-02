import lzma
import pickle

def load_distances(file_path):
    with lzma.open(file_path, "rb") as f:
        distances = pickle.load(f)
    return distances

def compare_distances(distances1, distances2):
    mismatches = []
    
    # Check distances from distances1 against distances2
    for key, dist1 in distances1.items():
        dist2 = distances2.get(key)
        if dist2 is None:
            mismatches.append((key, dist1, None))
        elif dist1 != dist2:
            mismatches.append((key, dist1, dist2))
    
    # Check for any pairs in distances2 not present in distances1
    for key, dist2 in distances2.items():
        if key not in distances1:
            mismatches.append((key, None, dist2))
    
    return mismatches

if __name__ == "__main__":
    file1 = "imagenet_ilsvrc_distances.pkl.xz"
    file2 = "imagenet2_ilsvrc_distances.pkl.xz"
    
    distances1 = load_distances(file1)
    distances2 = load_distances(file2)
    
    mismatches = compare_distances(distances1, distances2)
    
    output_file = "distance_comparison_results.txt"
    with open(output_file, "w") as f:
        if mismatches:
            f.write("Differences found:\n")
            for key, dist1, dist2 in mismatches:
                f.write(f"Pair {key}: {dist1} != {dist2}\n")
        else:
            f.write("No differences found. The distances are the same for all pairs of nodes.\n")
    
    print(f"Differences saved to {output_file}")
