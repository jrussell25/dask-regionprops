from typing import Union

import dask.array as da
import numpy as np
import xarray as xr

ArrayLike = Union[np.ndarray, da.Array, xr.DataArray]
