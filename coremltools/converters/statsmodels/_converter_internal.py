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
	if not(HAS_STATSMODELS)

