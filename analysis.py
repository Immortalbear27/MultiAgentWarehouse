import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols

df = pd.read_csv("results.csv")
model = ols("TotalDeliveries ~ C(strategy)*C(num_agents)", data=df).fit()
anova = sm.stats.anova_lm(model, typ=2)
print(anova)
