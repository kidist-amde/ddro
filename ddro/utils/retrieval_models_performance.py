import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Data from your table
data = {
    "Model": ["BM25", "DocT5Query", "DPR", "ANCE", "DSI", "DSI-QG", "NCI", "SEAL", "Ultron_SI", "Ultron_TU", "SFT_SI", "SFT_TU", "GenRRL", "DDRO_SI", "DDRO_TU"],
    "Hits@1_MS": [18.94, 23.27, 29.08, 29.65, 25.74, 28.82, 29.54, 27.58, 32.18, 38.49, 32.18, 38.12, 33.01, 32.92, 38.24],
    "Hits@5_MS": [42.28, 49.38, 62.75, 63.43, 43.58, 50.74, 57.99, 52.47, 62.00, 65.22, 62.62, 64.60, 63.62, 64.36, 66.46],
    "Hits@10_MS": [55.07, 63.61, 73.13, 74.28, 53.84, 62.26, 67.28, 61.01, 69.55, 72.15, 71.29, 72.90, 74.91, 73.02, 74.01],
    "MRR@10_MS": [29.24, 34.81, 43.41, 44.09, 33.92, 38.45, 40.46, 37.68, 44.57, 49.50, 44.79, 49.18, 45.93, 45.76, 50.07],
    "Hits@1_NQ": [14.06, 19.07, 22.78, 24.54, 27.42, 30.17, 32.69, 29.30, 31.12, 33.63, 44.19, 39.58, 35.79, 48.92, 40.86],
    "Hits@5_NQ": [36.91, 43.88, 53.44, 54.21, 47.26, 53.20, 55.82, 54.12, 43.90, 46.06, 58.44, 50.50, 56.49, 64.10, 53.12],
    "Hits@10_NQ": [47.93, 55.83, 68.58, 69.08, 56.58, 66.37, 69.20, 68.53, 48.16, 49.62, 62.23, 53.53, 70.96, 67.31, 55.98],
    "MRR@10_NQ": [23.60, 29.55, 35.92, 36.88, 34.31, 38.85, 42.84, 40.34, 36.65, 38.98, 50.48, 44.32, 45.73, 55.51, 45.99]
}

# Converting to a DataFrame
df = pd.DataFrame(data)

# Set style and figure
sns.set(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Performance Comparison of Retrieval Models on MS MARCO and Natural Questions")

# Custom light colors
light_pink = sns.light_palette("pink", as_cmap=False, n_colors=15)
light_blue = sns.light_palette("skyblue", as_cmap=False, n_colors=15)

# Plot Hits@1
sns.barplot(x="Model", y="Hits@1_MS", hue="Model", data=df, ax=axes[0, 0], palette=light_blue, legend=False)
axes[0, 0].set_title("Hits@1 - MS MARCO")
axes[0, 0].tick_params(axis='x', rotation=90)

sns.barplot(x="Model", y="Hits@1_NQ", hue="Model", data=df, ax=axes[0, 1], palette=light_pink, legend=False)
axes[0, 1].set_title("Hits@1 - Natural Questions")
axes[0, 1].tick_params(axis='x', rotation=90)

# Plot MRR@10
sns.barplot(x="Model", y="MRR@10_MS", hue="Model", data=df, ax=axes[1, 0], palette=light_blue, legend=False)
axes[1, 0].set_title("MRR@10 - MS MARCO")
axes[1, 0].tick_params(axis='x', rotation=90)

sns.barplot(x="Model", y="MRR@10_NQ", hue="Model", data=df, ax=axes[1, 1], palette=light_pink, legend=False)
axes[1, 1].set_title("MRR@10 - Natural Questions")
axes[1, 1].tick_params(axis='x', rotation=90)

# Save the figure as a PDF
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("retrieval_models_performance_new.pdf", format="pdf", bbox_inches="tight")
plt.show()
