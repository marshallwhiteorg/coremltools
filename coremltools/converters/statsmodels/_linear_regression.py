from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_regressor_interface_params
from ...proto import Model_pb2 as _Model_pb2
from ...proto import FeatureTypes_pb2 as _FeatureTypes_pb2

import numpy as _np

from ..._deps import HAS_STATSMODELS as _HAS_STATSMODELS
from ...models import MLModel as _MLModel

if _HAS_STATSMODELS:
	from . import _statsmodels_util
	import statsmodels
	from statsmodels.regression.linear_model import OLS
	model_type = 'regressor'
	statsmodels_class = statsmodels.regression.linear_model.RegressionResults

def convert(model, features, target):
    """Convert a linear regression model to the protobuf spec.
    Parameters
    ----------
    model: LinearRegression
        A trained linear regression encoder model.

    feature_names: [str]
        Name of the input columns.

    target: str
        Name of the output column.

    Returns
    -------
    model_spec: An object of type Model_pb.
        Protobuf representation of the model
    """
    if not(_HAS_STATSMODELS):
    	raise RuntimeError('statsmodels not found. statsmodels conversion API is disabled.')

    # check the statsmodels model
    _statsmodels_util.check_expected_type(model, RegressionResults)
    _statsmodels_util.check_fitted()

    return _MLModel(_convert(model, features, target))

def _convert(model, features, target):
	# Set the model class (regressor)
	spec = _Model_pb2.Model()
	spec.specificationVersion = SPECIFICATION_VERSION
	spec = set_regressor_interface_params(spec, features, target)

	# Add parameters for the linear regression
	lr = spec.glmRegressor

	if(isinstance(model.intercept_, _np.ndarray)):
		assert(len(model.intercept__) == 1)
		lr.offset.append(model.intercept_[0])
	else:
		lr.offset.append(model.intercept_)

	weights = lr.weights.add()
	for i in model.params:
		weights.value.append(i)
	return spec

def get_input_dimension(model):
	if not(_HAS_STATSMODELS):
		raise RuntimeError('statsmodels not found. statsmodels conversion API is disabled.')
	_statsmodels_util.check_fitted()
	return model.params.size