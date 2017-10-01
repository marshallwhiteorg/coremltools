import statsmodels.api as sm
import coremltools

spector_data = sm.datasets.spector.load()
spector_data.exog = sm.add_constant(spector_data.exog)
logit_mod = sm.Logit(spector_data.endog, spector_data.exog)
logit_res = logit_mod.fit()
converted_model = coremltools.converters.statsmodels.convert(logit_res, ['x1', 'x2', 'x3'], ['y'])
print converted_model
