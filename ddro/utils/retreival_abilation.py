import matplotlib.pyplot as plt
import numpy as np
import os

def ensure_output_directory(directory):
    os.makedirs(directory, exist_ok=True)

def plot_combined_bar_charts(data, output_dir):
    models = ["DDRO (PQ)", "DDRO (PQ) w/o pairwise ranking", "DDRO (TU)", "DDRO (TU) w/o pairwise ranking"]
    metrics = ["R@1", "R@5", "R@10", "MRR@10"]
    datasets = ["MS MARCO", "Natural Questions"]
    
    colors = {
        "MS MARCO": ['#6495ED', '#FF6347', '#32CD32', '#FFD700'],
        "Natural Questions": ['#4169E1', '#FF4500', '#2E8B57', '#FFA500']
    }
    
    for dataset in datasets:
        values = np.array([
            data[dataset]["R@1"],
            data[dataset]["R@5"],
            data[dataset]["R@10"],
            data[dataset]["MRR@10"]
        ])
        
        x = np.arange(len(models))
        bar_width = 0.2
        
        plt.figure(figsize=(14, 7))
        for i in range(len(metrics)):
            plt.bar(x + i * bar_width, values[i], bar_width,
                    label=f"{metrics[i]}", color=colors[dataset][i], edgecolor='black', alpha=0.9)
        
        plt.xlabel("Models", fontsize=14)
        plt.ylabel("Metric Values", fontsize=14)
        plt.title(f"Comparison of DDRO Variants on {dataset}", fontsize=16, fontweight='bold')
        plt.xticks(x + bar_width, models, fontsize=12, rotation=15)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12, loc='upper left')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        filename = f"{dataset.lower().replace(' ', '_')}_comparison.pdf"
        plt.savefig(os.path.join(output_dir, filename), format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Saved chart: {filename}")

# Main execution
data = {
    "MS MARCO": {
        "R@1": [32.92, 32.18, 38.24, 38.12],
        "R@5": [64.36, 62.62, 66.46, 64.60],
        "R@10": [73.02, 71.29, 74.01, 72.90],
        "MRR@10": [45.76, 44.79, 50.07, 49.18]
    },
    "Natural Questions": {
        "R@1": [48.92, 44.19, 40.86, 39.58],
        "R@5": [64.10, 58.44, 53.12, 50.50],
        "R@10": [67.31, 62.23, 55.98, 53.53],
        "MRR@10": [55.51, 50.48, 45.99, 44.32]
    }
}

output_dir = "Images-ablation"
ensure_output_directory(output_dir)

plot_combined_bar_charts(data, output_dir)
