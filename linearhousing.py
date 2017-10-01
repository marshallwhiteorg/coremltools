import pandas as pd
import coremltools
import statsmodels.api as sm

# Load data from file
data = pd.read_csv('houses.csv')

# Begin statsmodels fitting
statsmodels_linear_model = sm.OLS(data["Price"], data[["Bedrooms", "Bathrooms", "Size"]])
fitted_statsmodel_model = statsmodels_linear_model.fit()

# Convert fitted_statsmodel_model to .mlmodel
coreml_model = coremltools.converters.statsmodels.convert(fitted_statsmodel_model, ["Bedrooms", "Bathrooms", "Size"], "Price")

# Print statsmodels prediction
print "STATSMODELS PREDICTION (2 Bathrooms, 4 Bedrooms, 2000 SQ. Feet) \n" + str(fitted_statsmodel_model.predict([4, 2, 2000])[0]) + "\n"

# Print coreml_model prediction
print "COREML MODEL PREDICTION (2 Bathrooms, 4 Bedrooms, 2000 SQ. Feet) \n" + str(coreml_model.predict({'Bedrooms': 4, 'Bathrooms': 2, 'Size': 2000})['Price'])