"""Image and video I/O wrappers with network filesystem retry."""

import logging

from ._retry import retry_netfs

__all__ = ["load_image", "image_dims", "save_image", "save_video"]


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
def image_dims(path):
    """Return (width, height) of an image by reading only its header.

    ``PIL.Image.open`` is lazy — it parses the header without decoding pixel
    data, so this reads a few KB instead of the whole file. Use instead of
    ``load_image(...).shape`` when only the dimensions are needed (full-res
    images on network mounts are expensive to pull and decode).
    """
    from PIL import Image
    with Image.open(path) as im:
        return im.size


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


def save_video(path, array, fps=30, codec='libx264', format='mp4'):
    """Encode to a local temp file, then deliver it to *path* (delete-first, retried copy).

    ffmpeg writing directly to a stale network mount blocks in kernel I/O
    without raising, which no retry decorator can catch; only the final
    copy touches the network (and carries its own retry). Same path as
    ``bg3dtools.image_tools.video.save_video``: ``encode_video`` +
    ``background_io.deliver``.
    """
    import os
    from bg3dtools.image_tools.video import encode_video
    from bg3dtools.utils.background_io import deliver

    log = logging.getLogger(__name__)
    suffix = os.path.splitext(str(path))[1] or ('.%s' % format)
    try:
        deliver(encode_video(array, fps, suffix=suffix, codec=codec, format=format), path)
    except Exception:
        log.error('Failed to save %s' % path)
        raise
