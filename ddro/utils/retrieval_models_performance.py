import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from your table
# data = {
#     "Model": [
#         "BM25", "DocT5Query", "DPR", "ANCE", "DSI (SI)", "DSI-QG(SI)", "NCI (SI)", "SEAL (NG)", "Ultron (TU)",
#         "Ultron (SI)", "GenRRL (TU)", "DDRO (SI)", "DDRO (TU)"
#     ],
#     "Hits@1_MS": [13.42, 23.27, 29.08, 29.65, 25.74, 28.82, 29.54, 27.58, 29.82, 31.55, 33.01, 32.92, 38.24],
#     "Hits@5_MS": [36.43, 49.38, 62.75, 63.43, 43.58, 50.74, 57.99, 52.47, 60.39, 63.98, 63.62, 64.36, 66.46],
#     "Hits@10_MS": [47.56, 63.61, 73.13, 74.28, 53.84, 62.26, 67.28, 61.01, 68.31, 73.14, 74.91, 73.02, 74.01],
#     "MRR@10_MS": [23.16, 34.81, 43.41, 44.09, 33.92, 38.45, 40.46, 37.68, 42.53, 45.35, 45.93, 45.76, 50.07],
#     "Hits@1_NQ": [14.06, 19.07, 22.78, 24.54, 27.42, 30.17, 32.69, 29.30, 33.78, 25.64, 35.79, 48.92, 40.86],
#     "Hits@5_NQ": [36.91, 43.88, 53.44, 54.21, 47.26, 53.20, 55.82, 54.12, 54.20, 53.09, 56.49, 64.10, 53.12],
#     "Hits@10_NQ": [47.93, 55.83, 68.58, 69.08, 56.58, 66.37, 69.20, 68.53, 67.05, 65.75, 70.96, 67.31, 55.98],
#     "MRR@10_NQ": [23.60, 29.55, 35.92, 36.88, 34.31, 38.85, 42.84, 40.34, 42.51, 37.12, 45.73, 55.51, 45.99]
# }

data = {
    "Model": [
        "BM25", "DocT5Query", "DPR", "ANCE", "RepBERT", "Sentence-T5", "DSI (SI)", "DSI-QG(SI)", "NCI (SI)", 
        "SEAL (NG)", "Ultron (TU)", "Ultron (PQ)", "ROGER-NCI (SI)", "ROGER-Ultron (TU)", "MINDER (SI)", 
        "LTRGR (SI)", "GenRRL (TU)", "DDRO (PQ)", "DDRO (TU)"
    ],
    "R@1_MS": [18.94, 23.27, 29.08, 29.65, 25.25, 27.27, 25.74, 28.82, 29.54, 27.58, 29.82, 31.55, 30.61, 33.07, 29.98, 32.69, 33.01, 32.92, 38.24],
    "R@5_MS": [42.82, 49.38, 62.75, 63.43, 58.41, 58.91, 43.58, 50.74, 57.99, 52.47, 60.39, 63.98, 59.02, 63.93, 58.37, 64.37, 63.62, 64.36, 66.46],
    "R@10_MS": [55.07, 63.61, 73.13, 74.28, 69.18, 72.15, 53.84, 62.26, 67.28, 61.01, 68.31, 73.14, 68.78, 75.13, 71.92, 72.43, 74.91, 73.02, 74.01],
    "MRR@10_MS": [29.24, 34.81, 43.41, 44.09, 38.48, 40.69, 33.92, 38.45, 40.46, 37.68, 42.53, 45.35, 42.02, 46.35, 42.51, 47.85, 45.93, 45.76, 50.07],
    "R@1_NQ": [14.06, 19.07, 22.78, 24.54, 22.57, 22.51, 27.42, 30.17, 32.69, 29.30, 33.78, 25.64, 33.20, 35.90, 40.61, 42.68, 35.79, 48.92, 40.86],
    "R@5_NQ": [36.91, 43.88, 53.44, 54.21, 52.20, 52.00, 47.26, 53.20, 55.82, 54.12, 54.20, 53.09, 56.34, 55.59, 65.37, 68.35, 56.49, 64.10, 53.12],
    "R@10_NQ": [47.93, 55.83, 68.58, 69.08, 65.65, 65.12, 56.58, 66.37, 69.20, 68.53, 67.05, 65.75, 69.80, 69.86, 78.45, 79.26, 70.96, 67.31, 55.98],
    "MRR@10_NQ": [23.60, 29.55, 35.92, 36.88, 35.13, 34.95, 34.31, 38.85, 42.84, 40.34, 42.51, 37.12, 43.45, 44.92, 51.44, 53.96, 45.73, 55.51, 45.99]
}


# Convert to DataFrame
df = pd.DataFrame(data)

# Define categories for color coding
# categories = ["Index-based retrieval"] * 2 + ["Dense retrieval"] * 2 + ["End-to-end retrieval"] * 9
# categories[-2:] = ["Ours", "Ours"]  # Highlight "DDRO (SI)" and "DDRO (TU)" as "Ours"
# df['Category'] = categories

categories = (
    ["Index-based retrieval"] * 2 +  # First 2 models
    ["Dense retrieval"] * 4 +       # Next 4 models
    ["End-to-end retrieval"] * 11 +  # Next 11 models
    ["Ours"] * 2                   # Last 2 models
)
df['Category'] = categories

# Assign custom colors for each category
palette = {
    "Index-based retrieval": "lightcoral",
    "Dense retrieval": "lightblue",
    "End-to-end retrieval": "lightgreen",
    "Ours": "gold",
}

# Set style
sns.set(style="whitegrid")

# Ensure the Images directory exists
output_dir = "Images"
os.makedirs(output_dir, exist_ok=True)

# Metrics to plot
metrics = ["R@1_MS", "R@1_NQ", "MRR@10_MS", "MRR@10_NQ"]

# # Generate plots
# handles, labels = None, None  # To store legend elements for shared legend
# for i, metric in enumerate(metrics):
#     plt.figure(figsize=(14, 6))  # Increased height for better readability
#     barplot = sns.barplot(
#         x="Model", y=metric, hue="Category", data=df, palette=palette, dodge=False
#     )
#     plt.ylabel(metric.split("_")[0], fontsize=22)  # Larger Y-axis label
#     plt.xlabel("Model", fontsize=22)  # Larger X-axis label
#     plt.xticks(rotation=45, fontsize=18)  # Larger X-ticks
#     plt.yticks(fontsize=18)  # Larger Y-ticks
    
#     if i == 0:  # Save legend for shared plot
#         handles, labels = barplot.get_legend_handles_labels()
#     barplot.get_legend().remove()  # Remove individual legends

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{metric}_individual_plot.pdf"), format="pdf", bbox_inches="tight")
#     plt.show()

# Generate plots
handles, labels = None, None  # To store legend elements for shared legend
for i, metric in enumerate(metrics):
    plt.figure(figsize=(20, 10))  # Increased width for better readability
    barplot = sns.barplot(
        x="Model", y=metric, hue="Category", data=df, palette=palette, dodge=False
    )
    plt.ylabel(metric.split("_")[0], fontsize=24)  # Adjust Y-axis label font size
    plt.xlabel("Model", fontsize=26)  # Adjust X-axis label font size
    plt.xticks(rotation=90, fontsize=24)  # Rotate X-ticks and set smaller font size
    plt.yticks(fontsize=20)  # Adjust Y-ticks font size
    
    if i == 0:  # Save legend for shared plot
        handles, labels = barplot.get_legend_handles_labels()
    barplot.get_legend().remove()  # Remove individual legends

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric}_individual_plot.pdf"), format="pdf", bbox_inches="tight")
    plt.show()

# Create shared legend
if handles and labels:
    plt.figure(figsize=(10, 2))
    plt.legend(
        handles, labels, loc="center", ncol=4, fontsize=22,
        title="Type of model", title_fontsize=24, frameon=True, framealpha=0.9
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shared_legend.pdf"), format="pdf", bbox_inches="tight")
    plt.show()
