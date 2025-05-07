#!/usr/bin/env python3
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

References:
  McKinney, W. (2010) Data Structures for Statistical Computing in Python.
  Hunter, J.D. (2007) Matplotlib: A 2D Graphics Environment.
  Field, A. (2013) Discovering Statistics Using IBM SPSS Statistics.
  Cohen, J. (1988) Statistical Power Analysis for the Behavioral Sciences.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, sem, t, pearsonr
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.iolib.summary2 import summary_col


def load_and_clean(file_path):
    df = pd.read_csv(file_path)
    df = df.dropna(subset=["iteration"])
    cols = ["total_deliveries","ticks","pendingtasks","collisions","total_task_steps","iteration"]
    df[cols] = df[cols].astype(float)
    return df


def compute_initial_tasks(row):
    x0, x1 = row.shelf_edge_gap, row.width - row.shelf_edge_gap
    total_positions = x1 - x0
    aisles = total_positions // row.aisle_interval
    shelf_cells_per_row = total_positions - 2 * aisles
    return int(row.shelf_rows * shelf_cells_per_row)


def augment(df):
    df = df.copy()
    df["throughput"] = df["total_deliveries"] / df["ticks"]
    df["avg_steps"] = df["total_task_steps"] / df["total_deliveries"]
    df["collisions_per_delivery"] = df["collisions"] / df["total_deliveries"]
    df["initial_tasks"] = df.apply(compute_initial_tasks, axis=1)
    df["warehouse_area"] = df["width"] * df["height"]
    return df


def aggregate(df):
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
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x="initial_tasks", y="throughput", hue="strategy")
    plt.title("Throughput distributions by strategy & complexity")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "boxplot_throughput.png"))


def compute_confidence_intervals(df, metric="throughput", alpha=0.05):
    groups = df.groupby(["strategy","initial_tasks"]).apply(lambda g: g[metric])
    cis = groups.groupby(level=[0,1]).apply(lambda x: (
        x.mean() - sem(x) * t.ppf(1 - alpha/2, len(x)-1),
        x.mean() + sem(x) * t.ppf(1 - alpha/2, len(x)-1)
    ))
    ci_df = cis.apply(pd.Series).rename(columns={0:metric+"_ci_lower",1:metric+"_ci_upper"}).reset_index()
    return ci_df


def plot_error_bars(metric, metric_col, ylabel, filename, agg, ci_df, output_dir):
    plt.figure(figsize=(8, 5))
    merged = agg.merge(ci_df, on=["strategy","initial_tasks"])
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
    model = smf.ols('throughput ~ C(strategy)', data=df).fit()
    anova_res = anova_lm(model)
    eta2 = anova_res.loc['C(strategy)','sum_sq'] / anova_res['sum_sq'].sum()
    return model, anova_res, eta2


def run_posthoc_tukey(df, alpha=0.05):
    return pairwise_tukeyhsd(endog=df.throughput, groups=df.strategy, alpha=alpha)


def compute_cohens_d(df):
    strategies = df.strategy.unique()
    records = []
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            x1 = df[df.strategy==strategies[i]].throughput
            x2 = df[df.strategy==strategies[j]].throughput
            pooled_sd = np.sqrt(((len(x1)-1)*x1.std()**2 + (len(x2)-1)*x2.std()**2)/(len(x1)+len(x2)-2))
            d = (x1.mean() - x2.mean())/pooled_sd
            records.append({"pair":f"{strategies[i]} vs {strategies[j]}", "cohens_d":d})
    return pd.DataFrame.from_records(records)


def plot_effect_sizes(df_d, output_dir):
    plt.figure(figsize=(6,4))
    sns.barplot(data=df_d, x="pair", y="cohens_d")
    plt.title("Cohen's d by Strategy Pair")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cohens_d_bar.png"))


def fit_regression(df):
    return smf.ols('throughput ~ initial_tasks * C(strategy)', data=df).fit()


def save_regression_summary(model, output_dir):
    df_summary = summary_col([model], stars=True, float_format='%0.3f', model_names=['Regression'])
    with open(os.path.join(output_dir, "regression_summary.txt"), 'w') as f:
        f.write(df_summary.as_text())


def compute_slopes(agg):
    return {strat:np.polyfit(g.initial_tasks, g.mean_throughput,1)[0] for strat,g in agg.groupby("strategy")}


def compute_correlations(df):
    return {
        "throughput_vs_steps": pearsonr(df.throughput, df.avg_steps),
        "throughput_vs_collisions": pearsonr(df.throughput, df.collisions_per_delivery),
        "collisions_vs_area": pearsonr(df.collisions, df.warehouse_area)
    }


def plot_collisions_by_area(df, output_dir):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x="warehouse_area", y="collisions", hue="strategy")
    plt.xlabel("Warehouse area (units^2)")
    plt.ylabel("Total collisions")
    plt.title("Collisions vs Warehouse Size by Strategy")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "collisions_vs_area.png"))


def compute_composite(df):
    df = df.copy()
    for col in ["throughput","avg_steps","collisions_per_delivery"]:
        minv, maxv = df[col].min(), df[col].max()
        df[col+"_norm"] = (df[col] - minv) / (maxv - minv)
    df["composite_efficiency"] = df.throughput_norm - df.avg_steps_norm - df.collisions_per_delivery_norm
    return df


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("input_csv")
    p.add_argument("--output_dir", default="analysis_plots")
    args = p.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_and_clean(args.input_csv)
    df = augment(df)
    df = compute_composite(df)

    # Descriptive plots
    plot_distributions(df, args.output_dir)
    agg = aggregate(df)

    mapping = {
        "throughput": ("mean_throughput", "Mean throughput (deliveries/tick)", "throughput_CI.png"),
        "avg_steps": ("mean_steps", "Mean steps per delivery", "steps_CI.png"),
        "collisions_per_delivery": ("mean_collisions", "Mean collisions per delivery", "collisions_CI.png")
    }
    for metric, (col_name, ylabel, fname) in mapping.items():
        ci = compute_confidence_intervals(df, metric)
        plot_error_bars(metric, col_name, ylabel, fname, agg, ci, args.output_dir)

    plot_collisions_by_area(df, args.output_dir)

    # Inferential statistics
    model, anova_res, eta2 = run_anova(df)
    tukey = run_posthoc_tukey(df)

    # Effect sizes
    df_d = compute_cohens_d(df)
    df_d.to_csv(os.path.join(args.output_dir, "cohens_d.csv"), index=False)
    plot_effect_sizes(df_d, args.output_dir)

        # Regression summary
    save_regression_summary(model, args.output_dir)
    # Export regression table as markdown
    df_reg = pd.read_csv(os.path.join(args.output_dir, "regression_summary.txt"), sep="	", engine="python", comment="#") if False else None
    # Alternatively write model summary to markdown directly
    with open(os.path.join(args.output_dir, "regression_summary.md"), 'w') as f:
        f.write(model.summary().as_text())

    # Slopes and correlations
    slopes = compute_slopes(agg)
    pd.DataFrame.from_dict(slopes, orient='index', columns=['slope']).to_csv(
        os.path.join(args.output_dir, "throughput_slopes.csv"))
    corrs = compute_correlations(df)
    pd.DataFrame.from_dict(corrs, orient='index', columns=['correlation','p_value']).to_csv(
        os.path.join(args.output_dir, "correlations.csv"))

        # ANOVA and Tukey as text (and markdown)
    anova_df = anova_res.reset_index().rename(columns={'index':'source'})
    # Save as CSV and Markdown
    anova_df.to_csv(os.path.join(args.output_dir, "anova_results.csv"), index=False)
    with open(os.path.join(args.output_dir, "anova_results.md"), 'w') as f:
        f.write(anova_df.to_markdown(index=False))
    with open(os.path.join(args.output_dir, "anova_results.txt"), 'w') as f:
        f.write(anova_res.to_string())
        f.write(f"Eta-squared: {eta2:.3f}")
    # Tukey post-hoc
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

    # Save aggregated metrics
    agg.to_csv(os.path.join(args.output_dir, "aggregated_metrics.csv"), index=False)

if __name__ == '__main__':
    main()
