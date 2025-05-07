#!/usr/bin/env python3
"""
Enhanced analysis program for multi-agent warehouse simulation results.

This script loads batch_results.csv, computes derived metrics,
aggregates results by strategy and task complexity,
produces descriptive and inferential analyses:
 - distribution plots (boxplots)
 - means ± 95% confidence intervals
 - ANOVA + Tukey HSD post-hoc
 - effect sizes (eta-squared, Cohen's d)
 - regression with interaction
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
        df.groupby(["strategy","initial_tasks"]) 
          .agg(mean_throughput=("throughput","mean"),
               std_throughput=("throughput","std"),
               mean_steps=("avg_steps","mean"),
               mean_collisions=("collisions_per_delivery","mean"),
               mean_composite=("composite_efficiency","mean"))
          .reset_index()
    )


def plot_distributions(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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


def plot_error_bars(metric, ylabel, filename, agg, ci_df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    merged = agg.merge(ci_df, on=["strategy","initial_tasks"])
    plt.figure(figsize=(8, 5))
    for strat, g in merged.groupby("strategy"):
        plt.errorbar(g.initial_tasks, g[f"mean_{metric}"],
                     yerr=[g[f"mean_{metric}"]-g[f"{metric}_ci_lower"],
                           g[f"{metric}_ci_upper"]-g[f"mean_{metric}"]],
                     marker='o', label=strat)
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
    return anova_res, eta2


def run_posthoc_tukey(df, alpha=0.05):
    return pairwise_tukeyhsd(endog=df.throughput, groups=df.strategy, alpha=alpha)


def compute_cohens_d(df, group1, group2):
    x1 = df[df.strategy==group1].throughput
    x2 = df[df.strategy==group2].throughput
    n1, n2 = len(x1), len(x2)
    s1, s2 = x1.std(), x2.std()
    pooled_sd = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    return (x1.mean() - x2.mean()) / pooled_sd


def fit_regression(df):
    return smf.ols('throughput ~ initial_tasks * C(strategy)', data=df).fit()


def compute_slopes(agg):
    slopes = {}
    for strat, g in agg.groupby("strategy"):
        m, _ = np.polyfit(g.initial_tasks, g.mean_throughput, 1)
        slopes[strat] = m
    return slopes


def compute_correlations(df):
    return {
        "throughput_vs_steps": pearsonr(df.throughput, df.avg_steps),
        "throughput_vs_collisions": pearsonr(df.throughput, df.collisions_per_delivery),
        "collisions_vs_area": pearsonr(df.collisions, df.warehouse_area)
    }


def plot_collisions_by_area(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
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

    df = load_and_clean(args.input_csv)
    df = augment(df)
    df = compute_composite(df)

    # descriptive
    plot_distributions(df, args.output_dir)
    # Throughput CI and plot
    ci_tp = compute_confidence_intervals(df, "throughput")
    agg = aggregate(df)
    plot_error_bars("throughput", "Mean throughput (deliveries/tick)",
                    "throughput_CI.png", agg, ci_tp, args.output_dir)
    # Path efficiency CI and plot
    ci_steps = compute_confidence_intervals(df, "avg_steps")
    agg_steps = agg.rename(columns={"mean_steps": "mean_avg_steps"})
    plot_error_bars("avg_steps", "Mean steps per delivery",
                    "steps_CI.png", agg_steps, ci_steps, args.output_dir)
    # Collision rate CI and plot
    ci_coll = compute_confidence_intervals(df, "collisions_per_delivery")
    agg_coll = agg.rename(columns={"mean_collisions": "mean_collisions_per_delivery"})
    plot_error_bars("collisions_per_delivery", "Mean collisions per delivery",
                    "collisions_CI.png", agg_coll, ci_coll, args.output_dir)
    # Warehouse size scatter
    plot_collisions_by_area(df, args.output_dir)

    # inferential
    anova_res, eta2 = run_anova(df)
    print("ANOVA results:\n", anova_res)
    print(f"Eta-squared: {eta2:.3f}")
    tukey = run_posthoc_tukey(df)
    print(tukey)

    # effect sizes
    strategies = df.strategy.unique()
    for i in range(len(strategies)):
        for j in range(i+1, len(strategies)):
            d = compute_cohens_d(df, strategies[i], strategies[j])
            print(f"Cohen's d ({strategies[i]} vs {strategies[j]}): {d:.2f}")

    # regression
    reg = fit_regression(df)
    print(reg.summary())

    # slopes
    slopes = compute_slopes(agg)
    print("Slopes of throughput vs complexity:", slopes)

    # correlations
    corrs = compute_correlations(df)
    print("Correlations:", corrs)

    # save aggregated results
    agg.to_csv(os.path.join(args.output_dir, "aggregated_metrics.csv"), index=False)

if __name__ == '__main__':
    main()
