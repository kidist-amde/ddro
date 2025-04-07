import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

def load_metrics(file_path):
    """
    Load metrics from a CSV file.
    Each row should correspond to a query, and columns contain the metrics.
    """
    return pd.read_csv(file_path)

def compare_metrics(metrics_df_1, metrics_df_2, metrics_list, test='ttest'):
    """
    Compare metrics between two models using statistical tests.
    
    :param metrics_df_1: DataFrame for Model 1 metrics (per-query results)
    :param metrics_df_2: DataFrame for Model 2 metrics (per-query results)
    :param metrics_list: List of metrics to compare
    :param test: Statistical test to use ('ttest' or 'wilcoxon')
    :return: DataFrame with p-values and significance for each metric
    """
    results = []

    for metric in metrics_list:
        if metric in metrics_df_1.columns and metric in metrics_df_2.columns:
            data1 = metrics_df_1[metric]
            data2 = metrics_df_2[metric]

            if test == 'ttest':
                # Perform paired t-test
                t_stat, p_value = ttest_rel(data1, data2)
            elif test == 'wilcoxon':
                # Perform Wilcoxon signed-rank test
                _, p_value = wilcoxon(data1, data2)
            else:
                raise ValueError("Invalid test. Choose 'ttest' or 'wilcoxon'.")

            # Append results
            results.append({
                "Metric": metric,
                "P-Value": p_value,
                "Significant": p_value < 0.05  # True if p-value is less than 0.05
            })

    results_df = pd.DataFrame(results)
    return results_df

if __name__ == "__main__":
    # Load CSV files for both models
    model1_path = "model1_metrics.csv"
    model2_path = "model2_metrics.csv"

    model1_metrics = load_metrics(model1_path)
    model2_metrics = load_metrics(model2_path)

    # List of metrics to compare
    metrics_to_compare = [
        'MRR@10', 'MRR', 'NDCG@10', 'NDCG@20', 'NDCG@100', 
        'MAP@20', 'P@1', 'P@10', 'P@20', 'P@100',
        'R@1', 'R@10', 'R@100', 'R@1000', 'Hit@1', 'Hit@5', 'Hit@10', 'Hit@100'
    ]

    # Perform comparison
    comparison_results = compare_metrics(model1_metrics, model2_metrics, metrics_to_compare, test='ttest')

    # Save results to CSV
    comparison_results.to_csv("comparison_results.csv", index=False)

    print("Comparison Results:")
    print(comparison_results)
