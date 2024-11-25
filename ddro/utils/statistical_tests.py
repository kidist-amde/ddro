import os
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import numpy as np



# Ensure the log folder exists
def ensure_log_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


# Step 1: Load Metrics from Two Files
def load_metrics(file_path1, file_path2, selected_metrics):
    """
    Load metrics from two CSV files and filter selected metrics.
    Handles missing values by filling them with zeros.
    """
    try:
        df1 = pd.read_csv(file_path1).fillna(0)
        df2 = pd.read_csv(file_path2).fillna(0)
        print(f"Columns in {file_path1}: {df1.columns.tolist()}")
        print(f"Columns in {file_path2}: {df2.columns.tolist()}")
        df1 = df1[selected_metrics]
        df2 = df2[selected_metrics]
        print(f"Metrics loaded successfully from {file_path1} and {file_path2}")
        return df1, df2
    except FileNotFoundError as e:
        raise FileNotFoundError(f"File not found: {e}")
    except Exception as e:
        raise ValueError(f"Error while loading metrics: {e}")


# Step 2: Perform Statistical Tests
def perform_stat_tests(df1, df2, metrics_to_test):
    """
    Perform paired t-tests to compare metrics between two datasets.
    Args:
        df1: DataFrame of metrics for the first dataset (e.g., Ultron).
        df2: DataFrame of metrics for the second dataset (e.g., DPO).
        metrics_to_test: List of metric names to compare.
    Returns:
        A DataFrame summarizing the statistical test results.
    """
    # Ensure metrics are present in both DataFrames
    missing_metrics = [m for m in metrics_to_test if m not in df1.columns or m not in df2.columns]
    if missing_metrics:
        raise ValueError(f"Metrics {missing_metrics} not found in one of the DataFrames.")
    
    # Perform vectorized paired t-tests
    means1 = df1[metrics_to_test].mean(axis=0)
    means2 = df2[metrics_to_test].mean(axis=0)
    improvements = ((means2 - means1) / means1) * 100

    # Vectorized T-Test
    t_stats, p_values = stats.ttest_rel(df1[metrics_to_test], df2[metrics_to_test], axis=0)

    # Construct the result DataFrame
    results_df = pd.DataFrame({
        "Metric": metrics_to_test,
        "T-Statistic": t_stats,
        "P-Value": p_values,
        "Significant": ["Yes" if p < 0.05 else "No" for p in p_values],
        "Ultron Mean": means1.values,
        "DDRO Mean": means2.values,
        "Improvement (%)": improvements.values
    })

    return results_df



# Step 3: Visualize and Save Metrics
def visualize_metrics(df1, df2, metrics_to_test, output_image_path):
    """
    Generate and save a bar plot to compare metrics between two datasets.
    """
    # Preprocess data for plotting
    means_df = pd.DataFrame({
        "Metric": metrics_to_test,
        "Ultron": df1[metrics_to_test].mean(axis=0).values,
        "DPO": df2[metrics_to_test].mean(axis=0).values
    }).set_index("Metric")

    # Generate the bar plot
    ax = means_df.plot(kind="bar", figsize=(10, 6), color=["skyblue", "orange"], edgecolor="black")
    plt.title("Comparison of Metrics: Ultron vs DPO", fontsize=14)
    plt.ylabel("Mean Value", fontsize=12)
    plt.xlabel("Metrics", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(rotation=45)

    # Annotate bars with values
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_image_path)
    print(f"Visualization saved to {output_image_path}")
    plt.close()



# Step 4: Visualize Metric Distributions
def visualize_distributions(df1, df2, metrics_to_test, output_image_path):
    """
    Generate and save distribution plots for metrics between two datasets.
    """
    num_metrics = len(metrics_to_test)

    # Prepare plots efficiently
    fig, axes = plt.subplots(num_metrics, 1, figsize=(12, 6 * num_metrics), constrained_layout=True)

    if num_metrics == 1:  # Single metric case
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_test):
        sns.kdeplot(df1[metric], label="Ultron", fill=True, alpha=0.5, color="blue", ax=ax)
        sns.kdeplot(df2[metric], label="DPO", fill=True, alpha=0.5, color="orange", ax=ax)
        ax.set_title(f"Distribution of {metric}", fontsize=14)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_ylabel("Density", fontsize=12)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    plt.savefig(output_image_path)
    print(f"Distribution plots saved to {output_image_path}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare metrics between two models and assess improvements.")
    parser.add_argument("--metrics_file1", type=str, required=True, help="Path to the first metrics CSV file (Ultron).")
    parser.add_argument("--metrics_file2", type=str, required=True, help="Path to the second metrics CSV file (DPO).")
    parser.add_argument("--metrics_to_test", nargs='+', type=str, required=True,
                        help="List of metrics to compare, e.g., 'MRR@10 Hit@1 Hit@5 Hit@10'.")
    parser.add_argument("--log_path", type=str, required=True, help="Path to save logs and outputs.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the statistical test results.")
    parser.add_argument("--comparison_image", type=str, required=True, help="Path to save the comparison bar plot.")
    parser.add_argument("--distribution_image", type=str, required=True, help="Path to save the metric distribution plots.")

    args = parser.parse_args()

    # Ensure log folder exists
    log_folder = ensure_log_folder(args.log_path)

    # Load metrics
    print("\nLoading metrics from files:")
    ultron_df, dpo_df = load_metrics(args.metrics_file1, args.metrics_file2, args.metrics_to_test)

    # Perform statistical tests
    print("\nPerforming statistical tests:")
    stat_results = perform_stat_tests(ultron_df, dpo_df, args.metrics_to_test)
    print("\nStatistical Test Results:")
    print(stat_results)

    # Save results to a file
    stat_results.to_csv(args.output_file, index=False)
    print(f"\nStatistical test results saved to {args.output_file}")

    # Visualize and save improvements
    print("\nVisualizing and saving metrics comparison:")
    visualize_metrics(ultron_df, dpo_df, args.metrics_to_test, args.comparison_image)

    # Visualize and save distributions
    print("\nVisualizing and saving metric distributions:")
    visualize_distributions(ultron_df, dpo_df, args.metrics_to_test, args.distribution_image)
