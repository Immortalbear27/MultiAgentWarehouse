"""
Enhanced analysis program for multi-agent warehouse simulation results.

This script loads batch_results.csv, computes derived metrics,
aggregates results by strategy and task complexity,
produces descriptive and inferential analyses, and
exports summary tables and plots for report inclusion:
 - distribution plots (boxplots)
 - means ± 95% confidence intervals
 - ANOVA + Tukey HSD post-hoc
 - effect sizes (eta-squared, Cohen's d) with bar chart
 - regression with interaction and tabular export
 - slopes vs complexity
 - correlation analysis
 - composite efficiency index
 - collision analysis by warehouse size
 - error-bar plots for path efficiency and collision rate
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import sem, t, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.iolib.summary2 import summary_col


def load_and_clean(file_path):
    """
    Load simulation results from CSV, ensure data validity, and cast types.

    Args: Path to the input CSV file containing batch results.

    Returns: Cleaned DataFrame with numeric columns cast and NaN iterations removed.
    """
    
    # Read CSV into DataFrame:
    df = pd.read_csv(file_path)
    
    # Drop rows missing iteration identifiers:
    df = df.dropna(subset=["iteration"])
    
    # Cast key columns to float for subsequent calculations:
    cols = ["total_deliveries","ticks","pendingtasks","collisions","total_task_steps","iteration"]
    df[cols] = df[cols].astype(float)
    return df


def compute_initial_tasks(row):
    """
    Compute the total number of initial pickup tasks based on shelf layout.

    Uses shelf_edge_gap, width, height, shelf_rows, and aisle_interval to estimate
    how many shelf cells exist initially.

    Args: A row from the DataFrame with layout parameters.

    Returns: Estimated count of initial tasks (one per shelf cell).
    """
    
    # Determine horizontal span excluding edge gaps:
    x0, x1 = row.shelf_edge_gap, row.width - row.shelf_edge_gap
    total_positions = x1 - x0
    
    # Count aisles as two-column gaps every aisle_interval:
    aisles = total_positions // row.aisle_interval
    
    # Effective shelf cells per row:
    shelf_cells_per_row = total_positions - 2 * aisles
    return int(row.shelf_rows * shelf_cells_per_row)


def augment(df):
    """
    Add derived performance metrics to the results DataFrame.

    Calculates:
      - throughput (deliveries per tick)
      - average steps per delivery
      - collisions per delivery
      - initial_tasks via compute_initial_tasks
      - warehouse_area (width × height)

    Args: Cleaned results DataFrame.

    Returns: A new DataFrame with additional metric columns.
    """
    
    df = df.copy()
    df["throughput"] = df["total_deliveries"] / df["ticks"]
    df["avg_steps"] = df["total_task_steps"] / df["total_deliveries"]
    df["collisions_per_delivery"] = df["collisions"] / df["total_deliveries"]
    
    # Estimate how many tasks were created initially:
    df["initial_tasks"] = df.apply(compute_initial_tasks, axis=1)
    
    # Compute grid area:
    df["warehouse_area"] = df["width"] * df["height"]
    return df


def aggregate(df):
    """
    Aggregate metrics by strategy and task complexity (initial_tasks).

    Computes mean throughput, std deviation, mean steps, collisions, and composite.

    Args: DataFrame with augmented metrics.

    Returns: Grouped summary with mean and std metrics per strategy and complexity.
    """
    return (
        df.groupby(["strategy","initial_tasks"]) \
          .agg(mean_throughput=("throughput","mean"),
               std_throughput=("throughput","std"),
               mean_steps=("avg_steps","mean"),
               mean_collisions=("collisions_per_delivery","mean"),
               mean_composite=("composite_efficiency","mean"))\
          .reset_index()
    )


def plot_distributions(df, output_dir):
    """
    Generate and save boxplots of throughput by strategy and initial task count.

    Args:
    df: DataFrame with throughput and grouping columns.
    output_dir: Directory to save the plot.
    """
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="initial_tasks", y="throughput", hue="strategy")
    plt.title("Throughput distributions by strategy & complexity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_throughput.png"))


def compute_confidence_intervals(df, metric="throughput", alpha=0.05):
    """
    Compute 95% confidence intervals for a given metric by group.

    Args:
    df: Data with column `metric` and grouping columns.
    metric: Column name for which to compute CIs.
    alpha: Significance level (default 0.05 for 95% CI).

    Returns: Columns ['strategy','initial_tasks', '<metric>_ci_lower', '<metric>_ci_upper'].
    """
    
    # Group by strategy and complexity, extract metric series:
    groups = df.groupby(["strategy","initial_tasks"]).apply(lambda g: g[metric])
    
    # Compute lower/upper bounds per group:
    cis = groups.groupby(level=[0,1]).apply(lambda x: (
        x.mean() - sem(x) * t.ppf(1 - alpha/2, len(x)-1),
        x.mean() + sem(x) * t.ppf(1 - alpha/2, len(x)-1)
    ))
    ci_df = cis.apply(pd.Series).rename(columns={0:metric+"_ci_lower",1:metric+"_ci_upper"}).reset_index()
    return ci_df


def plot_error_bars(metric, metric_col, ylabel, filename, agg, ci_df, output_dir):
    """
    Plot and save error-bar charts for a given aggregated metric.

    Args:
    metric: Base metric name (e.g. 'throughput').
    metric_col: Column in `agg` for the mean values.
    ylabel: Label for the Y-axis.
    filename: Output filename for the plot.
    agg: Aggregated metrics by strategy and complexity.
    ci_df: Confidence interval DataFrame from compute_confidence_intervals.
    output_dir: Directory to save the plot.
    """
    
    plt.figure(figsize=(8, 5))
    
    # Merge mean and CI data:
    merged = agg.merge(ci_df, on=["strategy","initial_tasks"])
    
    # plot each strategy's line with error bars:
    for strat, g in merged.groupby("strategy"):
        plt.errorbar(
            g.initial_tasks,
            g[metric_col],
            yerr=[g[metric_col] - g[f"{metric}_ci_lower"], g[f"{metric}_ci_upper"] - g[metric_col]],
            marker='o', label=strat
        )
    plt.xlabel("Initial tasks")
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} vs Task Complexity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename))


def run_anova(df):
    """
    Perform one-way ANOVA on throughput by strategy.

    Fits an OLS model throughput ~ C(strategy) and returns the model,
    the ANOVA table, and eta-squared effect size.

    Args: Data containing 'throughput' and 'strategy'.

    Returns: (fitted model, ANOVA DataFrame, eta_squared float)
    """
    model = smf.ols('throughput ~ C(strategy)', data=df).fit()
    anova_res = anova_lm(model)
    eta2 = anova_res.loc['C(strategy)','sum_sq'] / anova_res['sum_sq'].sum()
    return model, anova_res, eta2


def run_posthoc_tukey(df, alpha=0.05):
    """
    Conduct Tukey HSD post-hoc comparisons on throughput by strategy.

    Args:
    df: Data containing 'throughput' and 'strategy'.
    alpha: Significance level.

    Returns: Object with summary and group comparisons.
    """
    return pairwise_tukeyhsd(endog=df.throughput, groups=df.strategy, alpha=alpha)


def compute_cohens_d(df):
    """
    Compute Cohen's d effect size between all pairs of strategies.

    Args: Data with 'throughput' and 'strategy' columns.

    Returns: Rows with 'pair' and 'cohens_d'.
    """
    strategies = df.strategy.unique()
    records = []
    
    # Compare each pair of strategies:
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            x1 = df[df.strategy==strategies[i]].throughput
            x2 = df[df.strategy==strategies[j]].throughput
            
            # Pooled standard deviation:
            pooled_sd = np.sqrt(((len(x1)-1)*x1.std()**2 + (len(x2)-1)*x2.std()**2)/(len(x1)+len(x2)-2))
            d = (x1.mean() - x2.mean())/pooled_sd
            records.append({"pair":f"{strategies[i]} vs {strategies[j]}", "cohens_d":d})
    return pd.DataFrame.from_records(records)


def plot_effect_sizes(df_d, output_dir):
    """
    Create and save a bar plot of Cohen's d for each strategy pair.

    Args:
    df_d: DataFrame with 'pair' and 'cohens_d'.
    output_dir: Directory to save the plot.
    """
    plt.figure(figsize=(6,4))
    sns.barplot(data=df_d, x="pair", y="cohens_d")
    plt.title("Cohen's d by Strategy Pair")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cohens_d_bar.png"))


def fit_regression(df):
    """
    Fit a regression model: throughput ~ initial_tasks * strategy.

    Captures interaction effects between complexity and strategy.

    Args: Data with 'throughput', 'initial_tasks', and 'strategy'.

    Returns: Fitted statsmodels OLS model.
    """
    return smf.ols('throughput ~ initial_tasks * C(strategy)', data=df).fit()


def save_regression_summary(model, output_dir):
    """
    Export regression model summary to text file.

    Args:
    model: statsmodels RegressionResultsWrapper.
    output_dir: Directory to save the summary.
    """
    df_summary = summary_col([model], stars=True, float_format='%0.3f', model_names=['Regression'])
    with open(os.path.join(output_dir, "regression_summary.txt"), 'w') as f:
        f.write(df_summary.as_text())


def compute_slopes(agg):
    """
    Compute linear slopes of mean throughput vs initial_tasks per strategy.

    Args: Aggregated summary with mean_throughput.

    Returns: Mapping strategy -> slope coefficient.
    """
    return {strat:np.polyfit(g.initial_tasks, g.mean_throughput,1)[0] for strat,g in agg.groupby("strategy")}


def compute_correlations(df):
    """
    Compute Pearson correlations between key metrics.

    Returns:
      - throughput_vs_steps
      - throughput_vs_collisions
      - collisions_vs_area

    Args: Data containing relevant columns.

    Returns: Mapping metric pair -> (correlation, p-value).
    """
    return {
        "throughput_vs_steps": pearsonr(df.throughput, df.avg_steps),
        "throughput_vs_collisions": pearsonr(df.throughput, df.collisions_per_delivery),
        "collisions_vs_area": pearsonr(df.collisions, df.warehouse_area)
    }


def plot_collisions_by_area(df, output_dir):
    """
    Generate scatter plot of collisions vs warehouse area, colored by strategy.

    Args:
    df: Data with 'warehouse_area', 'collisions', and 'strategy'.
    output_dir: Directory to save the figure.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="warehouse_area", y="collisions", hue="strategy")
    plt.xlabel("Warehouse area (units^2)")
    plt.ylabel("Total collisions")
    plt.title("Collisions vs Warehouse Size by Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "collisions_vs_area.png"))


def compute_composite(df):
    """
    Compute a composite efficiency index combining normalized metrics.

    Normalizes throughput positively and steps/collisions negatively,
    then subtracts to form a single composite score.

    Args: Data with throughput, avg_steps, collisions_per_delivery.

    Returns: Copy of df with added 'composite_efficiency' column.
    """
    df = df.copy()
    for col in ["throughput","avg_steps","collisions_per_delivery"]:
        minv, maxv = df[col].min(), df[col].max()
        df[col+"_norm"] = (df[col] - minv) / (maxv - minv)
    df["composite_efficiency"] = df.throughput_norm - df.avg_steps_norm - df.collisions_per_delivery_norm
    return df


def main():
    """
    Entry point: parse arguments, run analyses, and save all outputs.

    Steps:
      1. Load and clean data
      2. Augment with derived metrics and composite index
      3. Generate descriptive plots and summaries
      4. Run inferential statistics (ANOVA, Tukey, effect sizes)
      5. Fit regression and export summary
      6. Compute slopes, correlations, and save tables
      7. Save aggregated metrics
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--output_dir", default="analysis_plots")
    args = p.parse_args()
    
    # Ensure output directory exists:
    os.makedirs(args.output_dir, exist_ok=True)

    # Data preparation:
    df = load_and_clean(args.input_csv)
    df = augment(df)
    df = compute_composite(df)

    # Descriptive plots:
    plot_distributions(df, args.output_dir)
    agg = aggregate(df)

    # Error-bar plots for key metrics:
    mapping = {
        "throughput": ("mean_throughput", "Mean throughput (deliveries/tick)", "throughput_CI.png"),
        "avg_steps": ("mean_steps", "Mean steps per delivery", "steps_CI.png"),
        "collisions_per_delivery": ("mean_collisions", "Mean collisions per delivery", "collisions_CI.png")
    }
    for metric, (col_name, ylabel, fname) in mapping.items():
        ci = compute_confidence_intervals(df, metric)
        plot_error_bars(metric, col_name, ylabel, fname, agg, ci, args.output_dir)

    plot_collisions_by_area(df, args.output_dir)

    # Inferential statistics:
    model, anova_res, eta2 = run_anova(df)
    tukey = run_posthoc_tukey(df)

    # Effect sizes and Regression:
    df_d = compute_cohens_d(df)
    df_d.to_csv(os.path.join(args.output_dir, "cohens_d.csv"), index=False)
    plot_effect_sizes(df_d, args.output_dir)

    # Regression summary:
    save_regression_summary(model, args.output_dir)
    
    # Export regression table as markdown:
    df_reg = pd.read_csv(os.path.join(args.output_dir, "regression_summary.txt"), sep="	", engine="python", comment="#") if False else None
    
    # Alternatively write model summary to markdown directly:
    with open(os.path.join(args.output_dir, "regression_summary.md"), 'w') as f:
        f.write(model.summary().as_text())

    # Slopes and correlations:
    slopes = compute_slopes(agg)
    pd.DataFrame.from_dict(slopes, orient='index', columns=['slope']).to_csv(
        os.path.join(args.output_dir, "throughput_slopes.csv"))
    corrs = compute_correlations(df)
    pd.DataFrame.from_dict(corrs, orient='index', columns=['correlation','p_value']).to_csv(
        os.path.join(args.output_dir, "correlations.csv"))

    # ANOVA and Tukey as text (and markdown):
    anova_df = anova_res.reset_index().rename(columns={'index':'source'})
    
    # Save as CSV and Markdown:
    anova_df.to_csv(os.path.join(args.output_dir, "anova_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "anova_results.md"), 'w') as f:
        f.write(anova_df.to_markdown(index=False))
    with open(os.path.join(args.output_dir, "anova_results.txt"), 'w') as f:
        f.write(anova_res.to_string())
        f.write(f"Eta-squared: {eta2:.3f}")
        
    # Tukey post-hoc:
    tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
    tukey_df.to_csv(os.path.join(args.output_dir, "tukey_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "tukey_results.md"), 'w') as f:
        f.write(tukey_df.to_markdown(index=False))
    with open(os.path.join(args.output_dir, "tukey_results.txt"), 'w') as f:
        f.write(tukey.summary().as_text())
    with open(os.path.join(args.output_dir, "anova_results.txt"), 'w') as f:
        f.write(anova_res.to_string())
        f.write(f"\nEta-squared: {eta2:.3f}\n")
    with open(os.path.join(args.output_dir, "tukey_results.txt"), 'w') as f:
        f.write(tukey.summary().as_text())

    # Save aggregated metrics:
    agg.to_csv(os.path.join(args.output_dir, "aggregated_metrics.csv"), index=False)

if __name__ == '__main__':
    main()
