import os
from time import perf_counter

import dask.array as da
import pandas as pd
from dask import delayed
from dask.distributed import Client

from dask_regionprops import regionprops, regionprops_df
from dask_regionprops.utils import random_labels


def main():
    client = Client()
    print(f"Started Distrbuted client. Dashboard at {client.dashboard_link}")

    os.makedirs("tmp", exist_ok=True)

    delayed_labels = delayed(random_labels)

    if "random_labels.zarr" in os.listdir("tmp"):
        print("Found data on disk")

    else:
        print("Generating data. Labels are (512,1024,1024) array of uint8.")
        imgs = []
        for i in range(int(512)):
            labels = delayed_labels((1024, 1024))
            imgs.append(da.from_delayed(labels, (1024, 1024), dtype="uint8"))

        d_data = da.stack(imgs).rechunk((128, -1, -1))
        d_data.to_zarr("tmp/random_labels.zarr")

    # Test a dask array backed by a zarr store
    t0 = perf_counter()

    data_0 = da.from_zarr("tmp/random_labels.zarr")
    regionprops(data_0).compute()

    t1 = perf_counter()

    dask_dask_time = t1 - t0

    print(f"Full dask {t1-t0:0.2f}")

    # Test on a numpy array loaded from a zarr store
    t0 = perf_counter()

    data_1 = da.from_zarr("tmp/random_labels.zarr").compute()
    regionprops(data_1).compute()

    t1 = perf_counter()

    dask_np_time = t1 - t0

    print(f"Dask parallel on numpy array: {t1-t0:0.2f}")

    # Test a naive for loop on a numpy array loaded from a zarr store
    t0 = perf_counter()

    data_2 = da.from_zarr("tmp/random_labels.zarr").compute()
    rprops_2 = pd.DataFrame()
    for im in data_2:
        props = regionprops_df(im)
        rprops_2 = rprops_2.append(props)

    t1 = perf_counter()

    loop_np_time = t1 - t0

    print(f"For loop on numpy array: {t1-t0:0.2f}")
    print()

    print("Results")
    print("-------")

    print(f"Dask array speedup: {dask_dask_time/loop_np_time:0.2f}")
    print(f"Parallel numpy speedup: {dask_np_time/loop_np_time:0.2f}")


if __name__ == "__main__":
    main()
