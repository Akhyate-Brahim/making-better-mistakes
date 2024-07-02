import subprocess
import json
import os
from datetime import datetime

# Parameter
beta_values = [4, 8, 15, 20, 30]
base_command = "python model_train.py --loss soft-labels"
results_dir = f"hierarchical_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(results_dir, exist_ok=True)
all_results = {}

for beta in beta_values:
    print(f"\nRunning with beta = {beta}")
    
    out_folder = os.path.join(results_dir, f"beta_{beta}")
    os.makedirs(out_folder, exist_ok=True)
    command = f"{base_command} --beta {beta} --out_folder {out_folder}"
    subprocess.run(command, shell=True, check=True)
    with open(os.path.join(out_folder, 'test_results.json'), 'r') as f:
        results = json.load(f)
    
    all_results[beta] = results
    
    print(f"Results for beta = {beta}:")
    print(f"Test Loss: {results['test_loss']:.4f}")
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    print(f"Number of mistakes: {results['num_mistakes']}/{results['total_samples']}")
    
    for i, k in enumerate([1, 5, 10, 20, 100]):
        print(f"Top-{k} results:")
        print(f"  Hierarchical Distance Avg: {results['hierarchical_distance_avg'][i]:.4f}")
        print(f"  Hierarchical Distance Top: {results['hierarchical_distance_top'][i]:.4f}")
        print(f"  Hierarchical Distance Mistakes: {results['hierarchical_distance_mistakes'][i]:.4f}")
        print(f"  Hierarchical Precision: {results['hierarchical_precision'][i]:.4f}")
        print(f"  Hierarchical mAP: {results['hierarchical_mAP'][i]:.4f}")

# Save all results to a single JSON file
with open(os.path.join(results_dir, 'all_results.json'), 'w') as f:
    json.dump(all_results, f, indent=4)

print(f"\nAll results have been saved in {results_dir}")
