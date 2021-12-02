from util import get_data

from dask_regionprops import regionprops

TEST_PROPERTIES = ("label", "bbox", "area", "centroid")


def test_numpy():
    data, reference_properites = get_data(mode="numpy")
    props = regionprops(data, properties=TEST_PROPERTIES).compute()
    assert (reference_properites == props).all().all()


def test_xarray_numpy():
    data, reference_properites = get_data(mode="xarray_numpy")
    props = regionprops(data, properties=TEST_PROPERTIES).compute()
    assert (reference_properites == props).all().all()


def test_dask():
    data, reference_properites = get_data(mode="dask")
    props = regionprops(data, properties=TEST_PROPERTIES).compute()
    assert (reference_properites == props).all().all()


def test_xarray_dask():
    data, reference_properites = get_data(mode="xarray_dask")
    props = regionprops(data, properties=TEST_PROPERTIES).compute()
    assert (reference_properites == props).all().all()
