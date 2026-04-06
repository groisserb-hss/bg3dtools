"""Image and video I/O wrappers with network filesystem retry."""

import logging

from ._retry import retry_netfs

__all__ = ["load_image", "save_image", "save_video"]


@retry_netfs
def load_image(path):
    """Load an image as RGB numpy array. Pillow → imageio → cv2."""
    import numpy as np
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        return np.array(Image.open(path).convert('RGB'))
    try:
        import imageio.v2 as imageio
    except ImportError:
        pass
    else:
        return imageio.imread(path)
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise OSError('Failed to load image: %s' % path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


@retry_netfs
def save_image(path, image):
    """Save an RGB numpy array as an image. Pillow → imageio → cv2."""
    try:
        from PIL import Image
    except ImportError:
        pass
    else:
        Image.fromarray(image).save(path)
        return
    try:
        import imageio.v2 as imageio
    except ImportError:
        pass
    else:
        imageio.imwrite(path, image)
        return
    import cv2
    if image.ndim == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image):
        raise OSError('Failed to save image: %s' % path)


@retry_netfs
def save_video(path, array, fps=30, codec='libx264', format='mp4'):
    import imageio.v2 as imageio

    log = logging.getLogger(__name__)
    try:
        with imageio.get_writer(path, fps=fps, codec=codec, format=format) as writer:
            for frame in array:
                writer.append_data(frame)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise
