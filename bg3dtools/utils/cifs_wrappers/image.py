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


def save_video(path, array, fps=30, codec='libx264', format='mp4'):
    """Encode to a local temp file, then copy to *path* with retry.

    ffmpeg writing directly to a stale network mount blocks in kernel I/O
    without raising, which no retry decorator can catch; only the final
    ``copy_file`` touches the network (and carries its own retry).
    """
    import os
    import tempfile
    import imageio.v2 as imageio

    from .filesystem import copy_file

    log = logging.getLogger(__name__)
    suffix = os.path.splitext(str(path))[1] or ('.%s' % format)
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    # mkstemp creates 0600; copy preserves mode, so widen to the umask
    # default an ordinary file write would have gotten.
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp_path, 0o666 & ~umask)
    try:
        with imageio.get_writer(tmp_path, fps=fps, codec=codec, format=format) as writer:
            for frame in array:
                writer.append_data(frame)
        copy_file(tmp_path, path)
    except Exception as e:
        log.error('Failed to save %s' % path)
        raise
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
