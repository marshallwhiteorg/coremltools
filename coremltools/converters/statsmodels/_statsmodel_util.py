def check_fitted():
    """
    Simple function to check if statsmodel model is fitted (We expect
    that the model will be fitted before it is given to convert)
    """
    pass
    

def check_expected_type(model, expected_type):
    """Check if a model is of the right type. Raise error if not.

    Parameters
    ----------
    model: model
        Any scikit-learn model

    expected_type: Type
        Expected type of the scikit-learn.
    """
    if (model.__class__.__name__ != expected_type.__name__):
        raise TypeError("Expected model of type '%s' (got %s)" % \
                (expected_type.__name__, model.__class__.__name__))