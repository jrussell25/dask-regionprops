from util import get_data

from dask_regionprops import regionprops

TEST_PROPERTIES = ("label", "bbox", "area", "centroid")


def test_numpy():
    data, reference_properites = get_data()
    props = regionprops(data, properties=TEST_PROPERTIES).compute()
    assert (reference_properites == props).all().all()
