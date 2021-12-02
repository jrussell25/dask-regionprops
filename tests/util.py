from itertools import product

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from skimage.draw import random_shapes
from skimage.measure import regionprops_table


def get_data(mode="numpy", shape=(3, 4, 256, 256)):

    data = np.zeros(shape, dtype="uint8")
    test_properties = ("label", "bbox", "area", "centroid")
    properties = pd.DataFrame()
    for n, (i, j) in enumerate(product(range(shape[0]), range(shape[1]))):
        im, regions = random_shapes(
            shape[-2:],
            max_shapes=10,
            min_shapes=5,
            multichannel=False,
            allow_overlap=False,
            random_seed=n,
        )
        im[im == 255] = 0
        data[i, j] = im
        im_props = pd.DataFrame(regionprops_table(im, properties=test_properties))
        im_props[["dim-0", "dim-1"]] = [i, j]
        properties = properties.append(im_props)

    if mode == "numpy":
        return data, properties

    elif mode == "dask":
        return da.from_array(data), properties

    elif mode == "xarray_numpy":
        x_data = xr.DataArray(data, dims=list("STYX"))
        return x_data, properties

    elif mode == "xarray_dask":
        d_data = da.from_array(data)
        x_data = xr.DataArray(d_data, dims=list("STYX"))
        return x_data, properties
    else:
        raise ValueError(
            f"Invalid mode argument. Found mode={mode}. Valid modes are "
            '"numpy", "dask", "xarray_numpy", "xarray_dask"'
        )
