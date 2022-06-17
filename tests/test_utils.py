import os
from pathlib import Path

import numpy as np

from dask_regionprops.utils import random_labels

dir_ = Path(os.path.abspath(__file__)).parent


def test_random_labels() -> None:

    local_path = "data_files/reference_shapes_random_seed_1.npy"
    full_path = dir_.joinpath(local_path)
    print(full_path)
    reference = np.load(full_path)
    labels = random_labels(random_seed=1)
    np.testing.assert_allclose(reference, labels)
