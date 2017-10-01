"""
The primary file for converting statsmodels models.
"""

from . import _linear_regession
from . import _logistic_regession

from ..._deps import HAS_STATSMODELS

from collections import namedtuple as _namedtuple
import numpy as _np
from six import string_types as _string_types

from ...models import _feature_management as _fm
from ...models import datatypes
from ...models.feature_vectorizer import create_feature_vectorizer
from ...models.pipeline import Pipeline, PipelineRegressor, PipelineClassifier

# simple list of supported statsmodels model types
_converter_module_list = [
	_linear_regession,
	_logistic_regession
]


_converter_lookup = dict( (md.sklearn_class, i) for i, md in enumerate(_converter_module_list))
_converter_functions = [md.convert for md in _converter_module_list]


def is_statsmodels_model(sm_obj):
	if not (HAS_STATSMODELS):
		raise RuntimeError('statsmodels not found. statsmodels conversion API is disabled.')
	return (sm_obj.__class__ in converter_lookup)

def _get_converter_module(sm_obj): 
	"""
    Returns the module holding the conversion functions for a 
    particular model).
    """
	try:
		cv_idx = _converter_lookup[sm_obj.__class__]
	except KeyError:
		raise ValueError(
			"Transformer '%s' not supported; supported transformers are %s."
			% (repr(sm_obj), 
				",".join(k.__name__ for k in _converter_module_list)))
	return _converter_module_list[cv_idx]


def _convert_statsmodels_model(input_sm_obj, input_features = None, 
	output_feature_name = None):
	"""
    Converts a generic statsmodels regressor into an coreML specification.
    Delegates to an factory function which provides the proper convert method to use.
    """
    if not(HAS_STATSMODELS):
        raise RuntimeError('scikit-learn not found. scikit-learn conversion API is disabled.')
    
    if input_sm_obj is None:
    	raise RuntimeError('Expecting one statsmodels model, none was provided.')

    if isinstance(input_sm_obj, list):
  		raise RuntimeError('Expecting only one statsmodels model, not a list.')  	

    if input_features is None:
        input_features = "input"

    if (output_feature_name is None):
    	output_feature_name = "output_val"

	# Calls the converter_module with the appropriate convert function and passes
	# it the statsmodel object, input features, and output feature. 

    return _get_converter_module(input_sm_obj).convert(
		input_sm_obj, input_features, output_feature_name)