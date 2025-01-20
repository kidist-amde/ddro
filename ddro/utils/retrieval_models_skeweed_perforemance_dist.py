import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from your table
data = {
    "Model": [
        "BM25", "DocT5Query", "DPR", "ANCE", "DSI (SI)", "DSI-QG(SI)", "NCI (SI)", "SEAL (NG)", "Ultron (TU)",
        "Ultron (SI)", "GenRRL (TU)", "DDRO (SI)", "DDRO (TU)"
    ],
    "Hits@1_MS": [13.42, 23.27, 29.08, 29.65, 25.74, 28.82, 29.54, 27.58, 29.82, 31.55, 33.01, 32.92, 38.24],
    "Hits@5_MS": [36.43, 49.38, 62.75, 63.43, 43.58, 50.74, 57.99, 52.47, 60.39, 63.98, 63.62, 64.36, 66.46],
    "Hits@10_MS": [47.56, 63.61, 73.13, 74.28, 53.84, 62.26, 67.28, 61.01, 68.31, 73.14, 74.91, 73.02, 74.01],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Normalize Hits metrics to calculate distributions
df["Hits@1_norm"] = df["Hits@1_MS"] / df["Hits@10_MS"]
df["Hits@5_norm"] = (df["Hits@5_MS"] - df["Hits@1_MS"]) / df["Hits@10_MS"]
df["Hits@10_norm"] = 1 - df["Hits@1_norm"] - df["Hits@5_norm"]

# Melt DataFrame for visualization
distribution_df = df.melt(
    id_vars=["Model"], 
    value_vars=["Hits@1_norm", "Hits@5_norm", "Hits@10_norm"], 
    var_name="Metric", 
    value_name="Proportion"
)

# Define categories for color coding
categories = ["Index-based retrieval"] * 2 + ["Dense retrieval"] * 2 + ["End-to-end retrieval"] * 9
categories[-2:] = ["OURS", "OURS"]  # Highlight "DDRO (SI)" and "DDRO (TU)" as "OURS"
df['Category'] = categories
distribution_df["Category"] = distribution_df["Model"].map(dict(zip(df["Model"], df["Category"])))

# Set style and figure
sns.set(style="whitegrid")
plt.figure(figsize=(16, 8))
sns.barplot(
    x="Model", 
    y="Proportion", 
    hue="Metric", 
    data=distribution_df, 
    palette=["gold", "lightblue", "lightgreen"]
)

# Customize plot
plt.title("Distribution of Hits@1, Hits@5, and Hits@10 on MS MARCO", fontsize=20)
plt.ylabel("Proportion", fontsize=16)
plt.xlabel("Model", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.yticks(fontsize=12)
plt.legend(title="Rank Contribution", fontsize=12, title_fontsize=14)
plt.tight_layout()

# Save the plot
plt.savefig("distribution_hits_msmarco.pdf", format="pdf", bbox_inches="tight")
plt.show()
