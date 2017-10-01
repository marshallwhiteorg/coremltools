# Authored by Marshall White
# September 30, 2017
#
# Use of this source code is governed by a BSD-3-clause license that can be
# found in the LICENSE.txt file or at https://opensource.org/licenses/BSD-3-Clause

from collections import Iterable

from ..._deps import HAS_STATSMODELS as _HAS_STATSMODELS
from ...models import MLModel as _MLModel

if _HAS_STATSMODELS:
    from statsmodels.discrete.discrete_model import LogitResults
    from . import _sklearn_util

    statsmodels_class = LogitResults

from ... import SPECIFICATION_VERSION
from ...models._interface_management import set_classifier_interface_params
from ...proto import Model_pb2 as _Model_pb2

model_type = 'classifier'


def convert(model, feature_names, target):
    """Convert a Logistic Regression model to the protobuf spec.
    Parameters
    ----------
    model: LogitResults
        A LogitResults object

    feature_names: [str], optional (default=None)
        Name of the input columns.

    target: str, optional (default=None)
        Name of the output column.

    Returns
    -------
    model_spec: An object of type Model_pb.
        Protobuf representation of the model
    """
    if not (_HAS_STATSMODELS):
        raise RuntimeError('statsmodels not found. statsmodels conversion API is disabled.')

    return _MLModel(_convert(model, feature_names, target))


def _convert(model, feature_names, target):
    spec = _Model_pb2.Model()
    spec.specificationVersion = SPECIFICATION_VERSION

    set_classifier_interface_params(spec, feature_names, target, 'glmClassifier', output_features=target)

    glmClassifier = spec.glmClassifier

    glmClassifier.postEvaluationTransform = glmClassifier.Logit

    return spec