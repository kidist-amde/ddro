import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
mpl.rc('font',family='Times New Roman')
data = {
    "Model": [
        "BM25", "DocT5Query", "DPR", "ANCE", "RepBERT", "Sentence-T5", "DSI (SI)", "DSI-QG(SI)", "NCI (SI)", 
        "SEAL (NG)", "Ultron (TU)", "Ultron (PQ)", "ROGER-NCI (SI)", "ROGER-Ultron (TU)", "MINDER (SI)", 
        "LTRGR (SI)", "DDRO (PQ)", "DDRO (TU)"
    ],
    "R@1_MS": [18.94, 23.27, 29.08, 29.65, 25.25, 27.27, 25.74, 28.82, 29.54, 27.58, 29.82, 31.55, 30.61, 33.07, 29.98, 32.69, 32.92, 38.24],
    "R@5_MS": [42.82, 49.38, 62.75, 63.43, 58.41, 58.91, 43.58, 50.74, 57.99, 52.47, 60.39, 63.98, 59.02, 63.93, 58.37, 64.37, 64.36, 66.46],
    "R@10_MS": [55.07, 63.61, 73.13, 74.28, 69.18, 72.15, 53.84, 62.26, 67.28, 61.01, 68.31, 73.14, 68.78, 75.13, 71.92, 72.43, 73.02, 74.01],
    "MRR@10_MS": [29.24, 34.81, 43.41, 44.09, 38.48, 40.69, 33.92, 38.45, 40.46, 37.68, 42.53, 45.35, 42.02, 46.35, 42.51, 47.85, 45.76, 50.07],
    "R@1_NQ": [14.06, 19.07, 22.78, 24.54, 22.57, 22.51, 27.42, 30.17, 32.69, 29.30, 33.78, 25.64, 33.20, 35.90, 31.00, 32.80, 48.92, 40.86],
    "R@5_NQ": [36.91, 43.88, 53.44, 54.21, 52.20, 52.00, 47.26, 53.20, 55.82, 54.12, 54.20, 53.09, 56.34, 55.59, 55.50, 56.20, 64.10, 53.12],
    "R@10_NQ": [47.93, 55.83, 68.58, 69.08, 65.65, 65.12, 56.58, 66.37, 69.20, 68.53, 67.05, 65.75, 69.80, 69.86, 65.79, 68.74, 67.31, 55.98],
    "MRR@10_NQ": [23.60, 29.55, 35.92, 36.88, 35.13, 34.95, 34.31, 38.85, 42.84, 40.34, 42.51, 37.12, 43.45, 44.92, 43.50, 44.80, 55.51, 45.99]
}

# Convert data to DataFrame
df = pd.DataFrame(data)

# Define categories for color coding
categories = (
    ["Term-based retrieval"] * 2 +  # First 2 models
    ["Dense retrieval"] * 4 +        # Next 4 models
    ["Generative retrieval"] * 10 +  # Next 10 models (instead of 11)
    ["Ours"] * 2                      # Last 2 models
)

df['Category'] = categories

# # Assign custom colors for each category
# palette_ms = {
#     "Index-based retrieval": "lightcoral",
#     "Dense retrieval": "lightblue",
#     "End-to-end retrieval": "lightgreen",
#     "Ours": "gold",
# }

# palette_nq = {
#     "Index-based retrieval": "indianred",
#     "Dense retrieval": "steelblue",
#     "End-to-end retrieval": "mediumseagreen",
#     "Ours": "darkgoldenrod",
# }

# Set style
sns.set(style="whitegrid")

# Ensure the Images directory exists
output_dir = "Images"
os.makedirs(output_dir, exist_ok=True)

# Metrics to plot for MS and NQ separately
ms_metrics = ["R@1_MS", "R@5_MS", "R@10_MS", "MRR@10_MS"]
nq_metrics = ["R@1_NQ", "R@5_NQ", "R@10_NQ", "MRR@10_NQ"]

# # Generate plots for MS dataset
# for metric in ms_metrics:
#     plt.figure(figsize=(20, 11))  # Increased width for better readability
#     barplot = sns.barplot(
#         x="Model", y=metric, hue="Category", data=df, palette=palette_ms, dodge=False
#     )
#     plt.ylabel(metric.split("_")[0], fontsize=24)  # Adjust Y-axis label font size
#     plt.xlabel("Model", fontsize=26)  # Adjust X-axis label font size
#     plt.xticks(rotation=90, fontsize=24)  # Rotate X-ticks and set font size
#     plt.yticks(fontsize=20)  # Adjust Y-ticks font size
#     plt.title(f"{metric.replace('_', ' ')} - MS MARCO", fontsize=28)
#     barplot.get_legend().remove()  # Remove individual legends

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{metric}_individual_plot_MS.pdf"), format="pdf", bbox_inches="tight")
#     plt.show()

# # Generate plots for NQ dataset with different colors
# for metric in nq_metrics:
#     plt.figure(figsize=(20, 10))  # Increased width for better readability
#     barplot = sns.barplot(
#         x="Model", y=metric, hue="Category", data=df, palette=palette_nq, dodge=False
#     )
#     plt.ylabel(metric.split("_")[0], fontsize=24)  # Adjust Y-axis label font size
#     plt.xlabel("Model", fontsize=26)  # Adjust X-axis label font size
#     plt.xticks(rotation=90, fontsize=24)  # Rotate X-ticks and set font size
#     plt.yticks(fontsize=20)  # Adjust Y-ticks font size
#     plt.title(f"{metric.replace('_', ' ')} - Natural Questions", fontsize=28)
#     barplot.get_legend().remove()  # Remove individual legends

#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f"{metric}_individual_plot_NQ.pdf"), format="pdf", bbox_inches="tight")
#     plt.show()

# # Create shared legend
# handles, labels = barplot.get_legend_handles_labels()
# plt.figure(figsize=(10, 2))
# plt.legend(
#     handles, labels, loc="center", ncol=4, fontsize=22,
#     title="Type of model", title_fontsize=24, frameon=True, framealpha=0.9
# )
# plt.axis("off")
# plt.tight_layout()
# plt.savefig(os.path.join(output_dir, "shared_legend_NQ.pdf"), format="pdf", bbox_inches="tight")
# plt.show()

# Use the same color palette for both MS MARCO and NQ
shared_palette = {
    "Term-based retrieval": "indianred",
    "Dense retrieval": "steelblue",
    "Generative retrieval": "mediumseagreen",
    "Ours": "darkgoldenrod",
}

# Assign custom colors for each category
# shared_palette = {
#     "Index-based retrieval": "lightcoral",
#     "Dense retrieval": "royalblue",
#     "End-to-end retrieval": "lightgreen",
#     "Ours": "gold",
# }

# shared_palette = {
#     "Index-based retrieval": "lightpink",
#     "Dense retrieval": "lightskyblue",
#     "End-to-end retrieval": "palegreen",
#     "Ours": "peachpuff",
# }

# Function to plot the metrics without any title
def plot_metrics(metrics, dataset_name):
    for metric in metrics:
        plt.figure(figsize=(20, 11))  # Increased width for better readability
        sns.barplot(
            x="Model", y=metric, hue="Category", data=df, palette=shared_palette, dodge=False
        )
        plt.ylabel("", fontsize=24)  # Remove Y-axis label
        plt.xlabel("", fontsize=26)  # Remove X-axis label
        plt.xticks(rotation=90, fontsize=24)  # Rotate X-ticks and set font size
        plt.yticks(fontsize=20)  # Adjust Y-ticks font size
        
        # Remove legend for individual plots
        plt.legend([],[], frameon=False)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{metric}_individual_plot_{dataset_name}.pdf"), format="pdf", bbox_inches="tight")
        plt.show()

# Plot for both MS MARCO and NQ using the same colors
plot_metrics(ms_metrics, "MS_MARCO")
plot_metrics(nq_metrics, "Natural_Questions")

# Create a shared legend
legend_labels = {
    "Term-based retrieval": "indianred",
    "Dense retrieval": "steelblue",
    "Generative retrieval": "mediumseagreen",
    "Ours": "darkgoldenrod",
}
# legend_labels = {
#     "Index-based retrieval": "lightcoral",
#     "Dense retrieval": "royalblue",
#     "Generative retrieval": "lightgreen",
#     "Ours": "gold",
# }

plt.figure(figsize=(10, 2))
for label, color in legend_labels.items():
    plt.bar(0, 0, color=color, label=label)

plt.legend(
    loc="center", ncol=4, fontsize=22,
    title="Type of Model", title_fontsize=24, frameon=True, framealpha=0.9
)
plt.axis("off")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shared_legend.pdf"), format="pdf", bbox_inches="tight")
plt.show()
