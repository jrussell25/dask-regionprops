from skimage.draw import random_shapes
from skimage.measure import label


def random_labels(shape, max_shapes=20, min_shapes=10, min_size=20):
    labels = random_shapes(
        shape,
        max_shapes=max_shapes,
        min_shapes=min_shapes,
        min_size=min_size,
        multichannel=False,
    )[0]
    labels[labels == 255] = 0
    labels = label(labels)
    return labels
