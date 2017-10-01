import statsmodels.api as sm
import coremltools

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog)
logit_mod = sm.OLS(y, x)
logit_res = logit_mod.fit()
#print logit_mod
#print logit_res
#print logit_res.model
converted_model = coremltools.converters.statsmodels.convert(logit_res, ['x'], 'y')
converted_model.save("testlinear1.mlmodel")
#prediction = converted_model.predict({'x' : 5})
#print prediction
print converted_model