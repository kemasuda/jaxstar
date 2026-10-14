"""Private compatibility helpers for ArviZ 0.x and 1.x result containers."""


def uses_datatree():
    import arviz as az

    return int(az.__version__.split(".", 1)[0]) >= 1


def is_inference_data(value):
    if uses_datatree():
        from xarray import DataTree
        return isinstance(value, DataTree)
    import arviz as az
    return isinstance(value, az.InferenceData)


def from_dict(groups, *, coords=None, dims=None):
    import arviz as az

    if uses_datatree():
        return az.from_dict(groups, coords=coords, dims=dims,
                            sample_dims=["chain", "draw"])
    return az.from_dict(**groups, coords=coords, dims=dims)
