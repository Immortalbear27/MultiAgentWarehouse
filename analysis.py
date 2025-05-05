import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import matplotlib.pyplot as plt
import scipy.stats as stats

# 1. Load the simulation results
df = pd.read_csv("_batch_results.csv")

# 2. Define a normalized efficiency metric
#    Assumes "ticks" column records the number of steps actually simulated per run
df["Efficiency"] = df["TotalDeliveries"] / (df["num_agents"] * df["ticks"])

# 3. Full-factorial ANOVA across all swept parameters
formula = (
    "Efficiency ~ "
    "C(strategy) * C(num_agents) * C(shelf_rows) * "
    "C(shelf_edge_gap) * C(aisle_interval) * C(search_radius)"
)
model = ols(formula, data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)
print("=== ANOVA Results ===")
print(anova)

# 4. Compute partial eta-squared for each effect
ss_error = model.ssr
anova["eta_sq"] = anova["sum_sq"] / (anova["sum_sq"] + ss_error)
print("\n=== Partial Eta-Squared ===")
print(anova["eta_sq"])

# 5. Post-hoc Tukey HSD on strategy main effect
tukey = pairwise_tukeyhsd(
    endog=df["Efficiency"],
    groups=df["strategy"],
    alpha=0.05
)
print("\n=== Tukey HSD for Strategy ===")
print(tukey.summary())

# 6. Interaction plot: strategy vs. shelf_rows
means = df.groupby(["shelf_rows", "strategy"])["Efficiency"].mean().unstack("strategy")
means.plot(marker="o")
plt.xlabel("Number of Shelf Rows")
plt.ylabel("Efficiency (Deliveries per Agent per Tick)")
plt.title("Strategy Performance vs Shelf Density")
plt.grid(True)
plt.show()

# 7. Diagnostics: residual QQ-plot
sm.qqplot(model.resid, line="45")
plt.title("QQ-Plot of ANOVA Residuals")
plt.show()

# 8. Diagnostics: Levene's test for equality of variances by strategy
groups = [group["Efficiency"].values for name, group in df.groupby("strategy")]
stat, pval = stats.levene(*groups)
print(f"\nLevene's test for homogeneity of variances by strategy: stat={stat:.3f}, p={pval:.3f}")

# 9. (Optional) Repeat Tukey for other factors, e.g., num_agents
tukey_agents = pairwise_tukeyhsd(
    endog=df["Efficiency"],
    groups=df["num_agents"].astype(str),
    alpha=0.05
)
print("\n=== Tukey HSD for Number of Agents ===")
print(tukey_agents.summary())

# 10. Save summary tables
anova.to_csv("anova_summary.csv")
tukey.summary().as_csv("tukey_strategy.csv")
tukey_agents.summary().as_csv("tukey_num_agents.csv")

