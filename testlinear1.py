import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import coremltools

"""
x = [[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]]
y = [2, 4, 6, 8, 10]
logit_mod = LinearRegression()
logit_mod.fit(x, y)
converted_model = coremltools.converters.sklearn.convert(logit_mod, ['x1', 'x2'], 'y')
converted_model.save("testlinear1.mlmodel")
print converted_model.predict({'x1' : 5, 'x2' : 5})
"""


x = [[1, 5], [2, 5], [3, 5], [4, 5], [5, 5]]
y = [2, 4, 6, 8, 10]
linear_mod = sm.OLS(y, x)
result = linear_mod.fit()
converted_model = coremltools.converters.statsmodels.convert(result, ['x1', 'x2'], 'y')
converted_model.save("testlinear1.mlmodel")
print converted_model.predict({'x1' : 5, 'x2' : 5})
