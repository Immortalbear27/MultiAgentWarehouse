                            OLS Regression Results                            
==============================================================================
Dep. Variable:             throughput   R-squared:                       0.161
Model:                            OLS   Adj. R-squared:                  0.160
Method:                 Least Squares   F-statistic:                     564.5
Date:                Wed, 07 May 2025   Prob (F-statistic):          4.67e-225
Time:                        16:25:24   Log-Likelihood:                 11694.
No. Observations:                5901   AIC:                        -2.338e+04
Df Residuals:                    5898   BIC:                        -2.336e+04
Df Model:                           2                                         
Covariance Type:            nonrobust                                         
================================================================================================
                                   coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------------------------
Intercept                        0.0793      0.001    105.788      0.000       0.078       0.081
C(strategy)[T.decentralised]    -0.0122      0.001    -11.312      0.000      -0.014      -0.010
C(strategy)[T.swarm]            -0.0348      0.001    -33.076      0.000      -0.037      -0.033
==============================================================================
Omnibus:                     1854.253   Durbin-Watson:                   1.256
Prob(Omnibus):                  0.000   Jarque-Bera (JB):            10101.458
Skew:                           1.400   Prob(JB):                         0.00
Kurtosis:                       8.765   Cond. No.                         3.72
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.