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
	statsmodels_class = statsmodels.regression.linear_model.OLS

