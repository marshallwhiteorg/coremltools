from sklearn.linear_model import LinearRegression
import pandas as pd
import coremltools
import statsmodels.api as sm

# Load data
data = pd.read_csv('houses.csv')

# Begin sklearn fitting
#-----
# Train a model
"""
model = LinearRegression()
model.fit(data[["Bedrooms", "Bathrooms", "Size"]], data["Price"])
coreml_model = coremltools.converters.sklearn.convert(model, ["Bedrooms", "Bathrooms", "Size"], "Price")
print coreml_model.predict({'Bedrooms': 1.0, 'Bathrooms': 1.0, 'Size': 1240})
"""

# Begin statsmodels fitting
#-----
smodel = sm.OLS(data["Price"], data[["Bedrooms", "Bathrooms", "Size"]])
result = smodel.fit()
coreml_model = coremltools.converters.statsmodels.convert(result, ["Bedrooms", "Bathrooms", "Size"], "Price")
print coreml_model.predict({'Bedrooms': 1.0, 'Bathrooms': 1.0, 'Size': 1240})
coreml_model.save("linearhousingmodel.mlmodel")
# print result.predict(['Bedrooms', 'Bathrooms', 'Size'], exog=[1.0, 1.0, 1240])
print result.predict([4.0, 2.5, 2500])
# print result.predict(exog={'Bedrooms': 1.0, 'Bathrooms': 1.0, 'Size': 1240})
# print result.summary()