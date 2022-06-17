from __future__ import annotations

import numpy as np
from skimage.draw import random_shapes
from skimage.measure import label


def random_labels(
    shape: tuple[int, int] = (512, 512),
    max_shapes: int = 20,
    min_shapes: int = 10,
    min_size: int = 20,
    random_seed: int | None = None,
) -> np.ndarray:
    labels = random_shapes(
        shape,
        max_shapes=max_shapes,
        min_shapes=min_shapes,
        min_size=min_size,
        channel_axis=None,
        random_seed=random_seed,
    )[0]
    labels[labels == 255] = 0
    labels = label(labels)
    return labels
