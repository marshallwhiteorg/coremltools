"""
Defines the primary function for converting statsmodels models.
"""


def convert(sm_obj, input_features=None,
        output_feature_names=None):
    from ...models import MLModel
    from ._converter_internal import _convert_statsmodels_model

    spec = _convert_statsmodels_model(
            sm_obj, input_features, output_feature_names, class_labels = None)
    # print spec.__class__
    # return MLModel(spec)
    return spec