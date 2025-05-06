#!/usr/bin/env python3
import argparse
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import MultiComparison

def load_and_clean(path):
    """Load CSV and perform basic cleaning & type casting."""
    df = pd.read_csv(path)
    # standardize column names
    df.columns = (
        df.columns
          .str.strip()
          .str.lower()
          .str.replace(r'\s+', '_', regex=True)
    )
    # expected columns
    required = {'strategy', 'num_agents', 'ticks', 'total_deliveries',
                'total_task_steps', 'collisions', 'pendingtasks'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    # drop runs with zero ticks (avoid divide by zero)
    df = df[df['ticks'] > 0].copy()
    # cast categories
    df['strategy'] = df['strategy'].astype('category')
    df['num_agents'] = df['num_agents'].astype('category')
    return df

def compute_metrics(df):
    """Add derived metrics: efficiency, collision_rate, avg_steps."""
    df = df.copy()
    df['efficiency']     = df['total_deliveries'] / (df['num_agents'].astype(int) * df['ticks'])
    df['collision_rate'] = df['collisions'] / df['ticks']
    df['avg_steps']      = df['total_task_steps'] / df['total_deliveries']
    return df

def descriptive_stats(df, outdir):
    """Compute and save mean±std tables by strategy and by num_agents."""
    by_strat = df.groupby('strategy').agg({
        'efficiency': ['mean','std'],
        'collision_rate': ['mean','std'],
        'avg_steps': ['mean','std']
    })
    by_agents = df.groupby(['strategy','num_agents']).agg({
        'efficiency': ['mean','std'],
        'collision_rate': ['mean','std'],
        'avg_steps': ['mean','std']
    })
    by_strat.to_csv(os.path.join(outdir, 'desc_by_strategy.csv'))
    by_agents.to_csv(os.path.join(outdir, 'desc_by_strategy_num_agents.csv'))
    print("Descriptive tables saved.")

def plot_distributions(df, outdir):
    """Boxplots of efficiency, collision_rate, avg_steps by strategy."""
    metrics = ['efficiency','collision_rate','avg_steps']
    for m in metrics:
        plt.figure()
        sns.boxplot(x='strategy', y=m, data=df)
        plt.title(f'Distribution of {m}')
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f'box_{m}.png'))
        plt.close()
    print("Distribution plots saved.")

def plot_interactions(df, outdir):
    """Line plots of efficiency vs num_agents for each strategy."""
    plt.figure()
    sns.pointplot(x='num_agents', y='efficiency', hue='strategy', data=df, dodge=True, ci='sd')
    plt.title('Efficiency vs Number of Agents')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'efficiency_vs_num_agents.png'))
    plt.close()
    print("Interaction plot saved.")

def run_anova(df, outdir):
    """Two-way ANOVA on efficiency."""
    formula = 'efficiency ~ C(strategy) * C(num_agents)'
    model = smf.ols(formula, data=df).fit()
    anova = sm.stats.anova_lm(model, typ=2)
    with open(os.path.join(outdir, 'anova_efficiency.txt'), 'w') as f:
        f.write(anova.to_string())
    print("ANOVA results saved.")

def run_posthoc(df, outdir):
    """Tukey HSD for pairwise strategy comparisons on efficiency."""
    mc = MultiComparison(df['efficiency'], df['strategy'])
    result = mc.tukeyhsd()
    with open(os.path.join(outdir, 'tukey_efficiency.txt'), 'w') as f:
        f.write(result.summary().as_text())
    print("Tukey HSD results saved.")

def main(csv_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = load_and_clean(csv_path)
    df = compute_metrics(df)
    descriptive_stats(df, output_dir)
    plot_distributions(df, output_dir)
    plot_interactions(df, output_dir)
    run_anova(df, output_dir)
    run_posthoc(df, output_dir)
    print("All analysis complete. Outputs in:", output_dir)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Analyze warehouse strategy results")
    p.add_argument('csv', help="path to batch_results.csv")
    p.add_argument('-o','--out', default='analysis_output', help="output directory")
    args = p.parse_args()
    main(args.csv, args.out)
