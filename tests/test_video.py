"""load_video reads every frame of a video, or only the first n (the frame size without decoding the
whole recording, as humanfit's iPhone pipelines ask for it)."""
import numpy as np
import pytest

from bg3dtools.image_tools.video import load_video, save_video


@pytest.fixture
def video(tmp_path):
    frames = [np.full((48, 64, 3), 40 * i, dtype=np.uint8) for i in range(5)]
    path = tmp_path / 'clip.mp4'
    save_video(str(path), frames, fps=10)
    return path, len(frames)


def test_all_frames(video):
    path, n = video
    frames = load_video(path)
    assert len(frames) == n and frames[0].shape == (48, 64, 3) and frames[0].dtype == np.uint8


@pytest.mark.parametrize('n_frames', [1, 3])
def test_first_frames(video, n_frames):
    path, _ = video
    frames = load_video(path, n_frames=n_frames)
    assert len(frames) == n_frames
    np.testing.assert_array_equal(frames[0], load_video(path)[0])


def test_more_than_the_video_holds(video):
    path, n = video
    assert len(load_video(path, n_frames=n + 10)) == n


def test_zero_frames_is_refused(video):
    path, _ = video
    with pytest.raises(ValueError, match='n_frames'):
        load_video(path, n_frames=0)


def test_inside_a_background_block_the_copy_waits_for_the_writer(tmp_path):
    """The encode happens in the call; the copy to the destination is the block's writer's job,
    queued behind what the block submitted before it, and done when the block exits."""
    from functools import partial
    import threading
    from bg3dtools.utils.background_io import background_io

    hold = threading.Event()

    def holding():
        assert hold.wait(5)

    path = tmp_path / 'clip.mp4'
    frames = [np.full((48, 64, 3), 40 * i, dtype=np.uint8) for i in range(3)]
    with background_io() as out:
        out.submit(holding)
        save_video(str(path), frames, fps=10)
        assert not path.exists()                 # encoded locally, delivery queued behind the held job
        hold.set()
    assert len(load_video(path)) == 3


def test_a_video_replaces_the_previous_one(tmp_path):
    path = tmp_path / 'clip.mp4'
    path.write_bytes(b'not a video')
    save_video(str(path), [np.zeros((48, 64, 3), dtype=np.uint8)] * 2, fps=10)
    assert len(load_video(path)) == 2
