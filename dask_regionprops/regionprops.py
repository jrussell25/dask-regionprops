__all__ = [
    "regionprops",
    "regionprops_df",
]

import os
from itertools import product

import dask.dataframe as dd
import pandas as pd
import xarray as xr
from dask import delayed
from skimage.measure import regionprops_table

DEFAULT_PROPERTIES = (
    "label",
    "bbox",
    "centroid",
    "area",
    "convex_area",
    "eccentricity",
    "equivalent_diameter",
    "euler_number",
    "extent",
    "feret_diameter_max",
    "perimeter_crofton",
    "solidity",
    "moments_hu",
)


def regionprops_df(
    labels_im, intensity_im=None, props=DEFAULT_PROPERTIES, other_cols={}
):
    df = pd.DataFrame(regionprops_table(labels_im, intensity_im, properties=props))
    for k, v in other_cols.items():
        df[k] = v
    return df


def regionprops(labels, intensity=None, properties=DEFAULT_PROPERTIES, core_dims=None):
    """
    Loop over the frames of ds and compute the regionprops for
    each labelled image in each frame.

    Parameters
    ----------
    labels : array-like of int
        Array containing labelled regions. Background is assumed to have
        value 0 and will be ignored.
    intensity : array-like or None, Default None
        Optional intensity field to compute weighted region properties from.
    properties : str, tuple[str] default "non-image"
        Properties to compute for each region. Can pass an explicit tuple
        directly to regionprops or use one of the followings shortcuts:
        "minimal", "non-image", "all".
    """

    d_regionprops = delayed(regionprops_df)

    loop_sizes = _get_loop_sizes(labels, core_dims)

    labels_arr, intensity_arr = _get_arrays(labels, intensity)

    all_props = []

    for dims in product(*(range(v) for v in loop_sizes.values())):
        other_cols = dict(zip(loop_sizes.keys(), dims))

        frame_props = d_regionprops(
            labels_arr[dims], intensity_arr, properties, other_cols
        )
        all_props.append(frame_props)

    cell_props = dd.from_delayed(all_props, meta=all_props[0].compute())
    cell_props = cell_props.repartition(os.cpu_count() // 2)
    return cell_props


def _get_loop_sizes(labels, core_dims):

    if isinstance(labels, xr.DataArray):
        if core_dims is None:
            loop_sizes = {
                labels.dims[i]: labels.sizes[labels.dims[i]] for i in (-2, -1)
            }
        elif isinstance(core_dims[0], str):
            loop_dims = set(labels.dims) - set(core_dims)
            loop_sizes = {d: labels.sizes[d] for d in loop_dims}
        elif isinstance(core_dims[0], int):
            pos_core_dims = _get_pos_core_dims(core_dims, labels.ndim)
            loop_dims = set(range(labels.ndim)) - set(pos_core_dims)
            loop_sizes = {labels.dims[i]: labels.shape[i] for i in pos_core_dims}

    else:
        if core_dims is None:
            loop_shape = labels.shape[:-2]
            loop_sizes = {f"dim-{i}": v for i, v in enumerate(loop_shape)}

        else:
            pos_core_dims = _get_pos_core_dims(core_dims, labels.ndim)
            loop_dims = set(range(labels.ndim)) - set(pos_core_dims)
            loop_shape = (labels.shape[d] for d in loop_dims)

            loop_sizes = {f"dim-{i}": v for i, v in enumerate(loop_shape)}

    return loop_sizes


def _get_pos_core_dims(core_dims, ndim):
    pos_core_dims = []
    for d in core_dims:
        if d < 0:
            pos = ndim + d
            pos_core_dims.append(pos)
        else:
            pos_core_dims.append(d)
    return tuple(pos_core_dims)


def _get_arrays(labels, intensity):

    if intensity is None:
        intensity_arr = None
    else:
        if isinstance(intensity, xr.DataArray):
            intensity_arr = intensity.data
        else:
            intensity_arr = intensity

    if isinstance(labels, xr.DataArray):
        labels_arr = labels.data
    else:
        labels_arr = labels

    return labels_arr, intensity_arr
