from itertools import product

import dask.array as da
import numpy as np
import pandas as pd
import xarray as xr
from skimage.draw import random_shapes
from skimage.measure import regionprops_table
from skimage.segmentation import relabel_sequential


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
        x_properties = properties.rename({"dim-0": "S", "dim-1": "T"}, axis=1)
        return x_data, x_properties

    elif mode == "xarray_dask":
        d_data = da.from_array(data)
        x_data = xr.DataArray(d_data, dims=list("STYX"))
        x_properties = properties.rename({"dim-0": "S", "dim-1": "T"}, axis=1)
        return x_data, x_properties
    else:
        raise ValueError(
            f"Invalid mode argument. Found mode={mode}. Valid modes are "
            '"numpy", "dask", "xarray_numpy", "xarray_dask"'
        )


def random_labels(shape, max_shapes=20, min_shapes=10, min_size=20):
    labels = random_shapes(
        shape,
        max_shapes=max_shapes,
        min_shapes=min_shapes,
        min_size=min_size,
        multichannel=False,
    )[0]
    labels[labels == 255] = 0
    labels = relabel_sequential(labels)[0]
    return labels
