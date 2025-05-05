import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# 1. Load results
df = pd.read_csv("batch_results.csv")

# 2. Compute efficiency metric: deliveries per agent per tick
# assume column 'ticks' shows number of ticks until termination per run
if 'ticks' not in df.columns:
    df['ticks'] = df['Step']  # adapt if different

# Avoid zero ticks
df['Efficiency'] = df['TotalDeliveries'] / (df['num_agents'] * df['ticks'].replace(0, np.nan))

# 3. Descriptive statistics
print("\n=== Descriptive Statistics by Strategy ===")
desc = df.groupby('strategy')['Efficiency'].agg(['mean','std','count'])
print(desc)

# 4. Full-factorial ANOVA
formula = ('Efficiency ~ C(strategy) + C(num_agents) + C(shelf_rows) '
    '+ C(shelf_edge_gap) + C(aisle_interval) + C(search_radius) '
    '+ C(strategy):C(num_agents)'
    )
model = ols(formula, data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)
print("\n=== ANOVA Results ===")
print(anova)

# 5. Compute partial eta-squared
total_ss = sum(anova['sum_sq']) + model.ssr
anova['eta_sq'] = anova['sum_sq'] / (anova['sum_sq'] + model.ssr)
print("\n=== Partial Eta-Squared ===")
print(anova['eta_sq'])

# 6. Post-hoc: Tukey HSD for strategy
tukey_strat = pairwise_tukeyhsd(df['Efficiency'], df['strategy'])
print("\n=== Tukey HSD: Strategy ===")
print(tukey_strat)

# 7. Interaction plot: strategy × num_agents
plt.figure()
sns.pointplot(data=df, x='num_agents', y='Efficiency', hue='strategy', dodge=True)
plt.title('Efficiency by Strategy and Number of Agents')
plt.savefig('interaction_strategy_agents.png')

# 8. Distribution plots
plt.figure()
sns.violinplot(data=df, x='strategy', y='Efficiency')
plt.title('Efficiency Distribution by Strategy')
plt.savefig('violin_strategy_efficiency.png')

# 9. Residual diagnostics
sm.qqplot(model.resid, line='45')
plt.title('QQ-Plot of ANOVA Residuals')
plt.savefig('qqplot_resid.png')

# 10. Homogeneity test
groups = [g['Efficiency'].dropna() for _,g in df.groupby('strategy')]
stat, p = stats.levene(*groups)
print(f"Levene's test: stat={stat:.3f}, p={p:.3f}")

# 11. Save summary
anova.to_csv('anova_summary.csv')
with open('tukey_strat.txt','w') as f:
    f.write(str(tukey_strat))

def main():
    pass

if __name__ == '__main__':
    main()
