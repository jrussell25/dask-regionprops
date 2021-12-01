__all__ = [
    "regionprops",
    "regionprops_df",
]

import os
from itertools import product

import dask.dataframe as dd
import pandas as pd
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


def regionprops(labels, intensity=None, properties=DEFAULT_PROPERTIES, dims="STCZYX"):
    """
    Loop over the frames of ds and compute the regionprops for
    each labelled image in each frame.
    """
    if isinstance(dims, str):
        S, T, C, Z, Y, X = list(dims)
    elif isinstance(dims, list):
        S, T, C, Z, Y, X = dims

    d_regionprops = delayed(regionprops_df)

    loop_dims = {k: v for k, v in labels.sizes.items() if k not in [Y, X]}

    all_props = []

    for dims in product(*(range(v) for v in loop_dims.values())):
        other_cols = dict(zip(loop_dims.keys(), dims))
        if intensity is None:
            use_intensity = None
        else:
            use_intensity = intensity.data[dims]

        frame_props = d_regionprops(
            labels.data[dims], use_intensity, properties, other_cols
        )
        all_props.append(frame_props)

    cell_props = dd.from_delayed(all_props, meta=all_props[0].compute())
    cell_props = cell_props.repartition(os.cpu_count() // 2)
    return cell_props
