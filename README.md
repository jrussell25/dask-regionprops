# dask-regionprops

[![License](https://img.shields.io/pypi/l/dask-regionprops.svg?color=green)](https://github.com/jrussell25/dask-regionprops/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/dask-regionprops.svg?color=green)](https://pypi.org/project/dask-regionprops)
[![Python Version](https://img.shields.io/pypi/pyversions/dask-regionprops.svg?color=green)](https://python.org)
[![CI](https://github.com/jrussell25/dask-regionprops/actions/workflows/ci.yml/badge.svg)](https://github.com/jrussell25/dask-regionprops/actions)
[![codecov](https://codecov.io/gh/jrussell25/dask-regionprops/branch/master/graph/badge.svg)](https://codecov.io/gh/jrussell25/dask-regionprops)

## About

This is a small library that uses dask to compute regionprops in parallel.

In addition to parallelization, it adds a few features/specializations on top of
the `scikit-image` regionprops implementation.

1. `dask_regionprops` will return a dask dataframe containing the region properties as columns.
1. Arrays can be numpy or dask arrays as well as xarray DataArrays backed by either array libary.
1. ND arrays get processed as a sequence of 2D arrays. Typically we assume that the last two
   dimenions contain the images and the leading dimensions will be looped over.
1. In the ND case, the result dataframe will have columns that map each label


## Intended Use Case

I wrote this library to help analyze microscopy datasets. After segmentation I typically have a 4D xarray DataArray
where the dimensions are (Position, Time, Y, X). Importantly, I reuse label values between positions but not times
so for all of the time-points in position `S`, the region labelled `r` should refer to the same cell. Hopefully this
motivated the decision to return the leading dimensions in the dataframe. For instance if you want to get the properties
of a cell 5 in position 2 you could do something like:

```
from dask_regionprops import regionprops

# Assume data is a numpy/dask array that has dims corresponding to (S,T,Y,X)
props = regionprops(data)
single_cell_props = props.loc[(props["dim-0"]==2)&(props["label"]==5)]
```

If you are a more advanced pandas user, and you want to do this sort of analysis for many cells,
you might consider using the leading dimensions and region labels as a `multiindex` to more efficiently
access the data in this way.

Finally, a useful downstream application is to use the region properties as features for a classifer
or maybe even a clustering algorithm. I have personally input labelled regions and the corresponding 
fluorescence images to identify progression through the cell cycle.

## Contributions

Please feel free to open an issue or pull-request if you have questions or improvements for this library.

