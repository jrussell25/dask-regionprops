__all__ = [
    "regionprops",
    "regionprops_df",
]

import re
from itertools import product

import dask.dataframe as dd
import numpy as np
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

DEFAULT_WEIGHTED_PROPERTIES = (
    *DEFAULT_PROPERTIES,
    "centroid_weighted",
    "intensity_max",
    "intensity_mean",
    "intensity_min",
    "moments_weighted_hu",
)

DEFAULT_META = pd.DataFrame(
    regionprops_table(
        label_image=np.ones((1, 1), dtype="uint8"),
        intensity_image=np.ones((1, 1)),
        properties=DEFAULT_WEIGHTED_PROPERTIES,
    )
)

DEFAULT_PROPS_TO_COLS = {}
for prop in DEFAULT_WEIGHTED_PROPERTIES:
    col_list = []
    for c in DEFAULT_META.columns:
        stripped = re.sub("[-0-9]", "", c)
        if stripped == prop:
            col_list.append(c)
    DEFAULT_PROPS_TO_COLS[prop] = col_list


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
        "minimal", "non-image", "all". If provided an intensity image, basic
        weighted properties will also be computed by defualt.
    core_dims : tuple[int] or tuple[str] default None
        Dimensions of input arrays that correspond to spatial (xy) dimensions of each
        image. If None, it is assumed that the final two dimensions are the
        spatial dimensions.

    Returns
    -------
    regionprops_df : dask.DataFrame
        Lazily constructed dataframe containing columns for each specifified
        property.
    """

    d_regionprops = delayed(regionprops_df)

    loop_sizes = _get_loop_sizes(labels, core_dims)

    if intensity is not None:
        properties = DEFAULT_WEIGHTED_PROPERTIES

    meta = _get_meta(loop_sizes, properties)

    labels_arr, intensity_arr = _get_arrays(labels, intensity)

    all_props = []

    for dims in product(*(range(v) for v in loop_sizes.values())):
        other_cols = dict(zip(loop_sizes.keys(), dims))

        if intensity_arr is not None:
            frame_props = d_regionprops(
                labels_arr[dims], intensity_arr[dims], properties, other_cols
            )
        else:
            frame_props = d_regionprops(
                labels_arr[dims], None, properties, other_cols
            )

        all_props.append(frame_props)

    cell_props = dd.from_delayed(all_props, meta=meta)

    return cell_props


def _get_meta(loop_sizes, properties):

    meta = pd.DataFrame()
    for prop in properties:
        meta = meta.join(DEFAULT_META[DEFAULT_PROPS_TO_COLS[prop]])

    other_cols = pd.DataFrame(columns=list(loop_sizes.keys()), dtype=int)

    return meta.join(other_cols)


def _get_loop_sizes(labels, core_dims):

    if isinstance(labels, xr.DataArray):
        if core_dims is None:
            loop_sizes = {
                labels.dims[i]: labels.sizes[labels.dims[i]]
                for i in range(labels.ndim - 2)
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
