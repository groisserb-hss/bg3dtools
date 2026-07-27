"""Tests for bg3dtools.utils.cifs_wrappers.

These wrappers exist to survive flaky network mounts, but the failure modes they
guard against are all reproducible locally with monkeypatching -- no CIFS share
needed. The two behaviours pinned here are the ones that actually broke in
production:

* ``copy_file`` must not touch destination metadata (the copy2 -> copyfile fix).
* ``read_bytes`` must hand back raw bytes so the caller picks the encoding.
"""

import os
import shutil

import pytest

pytest.importorskip("tenacity")  # cifs_wrappers._retry imports it at module level

from bg3dtools.utils.cifs_wrappers import copy_file, read_bytes, read_text  # noqa: E402
from bg3dtools.utils.cifs_wrappers import filesystem, misc  # noqa: E402


# ---------------------------------------------------------------------------
# copy_file
# ---------------------------------------------------------------------------

def test_copy_file_copies_content(tmp_path):
    src = tmp_path / 'src.bin'
    dst = tmp_path / 'dst.bin'
    src.write_bytes(b'\x00\x01payload\xff')

    copy_file(str(src), str(dst))

    assert dst.read_bytes() == b'\x00\x01payload\xff'


def test_copy_file_does_not_set_destination_metadata(tmp_path, monkeypatch):
    """Regression guard for the copy2 -> copyfile fix.

    copy2 == copyfile + copystat, and copystat's utime() is only permitted for
    the file's owner. On a CIFS mount with forceuid mapping files to another uid,
    that raised EPERM *after* the bytes had already landed -- and since
    retry_netfs only retries EHOSTDOWN, it propagated straight out. Anyone who
    "restores" copy2 here reintroduces that, so assert copystat is never called.
    """
    called = []

    def exploding_copystat(*args, **kwargs):
        called.append(args)
        raise PermissionError(1, 'Operation not permitted')

    monkeypatch.setattr(shutil, 'copystat', exploding_copystat)

    src = tmp_path / 'src.txt'
    dst = tmp_path / 'dst.txt'
    src.write_text('hello')

    copy_file(str(src), str(dst))   # must not raise

    assert called == [], 'copy_file called copystat -- did it revert to copy2?'
    assert dst.read_text() == 'hello'


def test_copy_file_ignores_source_mode(tmp_path):
    """copyfile lets open(dst,'wb') create the file, so an exotic source mode
    is deliberately NOT carried over. Two call sites (cifs_wrappers.image and
    image_tools.video) rely on this to widen a 0600 mkstemp temp file."""
    src = tmp_path / 'src.txt'
    dst = tmp_path / 'dst.txt'
    src.write_text('x')
    os.chmod(src, 0o600)

    copy_file(str(src), str(dst))

    umask = os.umask(0)
    os.umask(umask)
    assert (dst.stat().st_mode & 0o777) == (0o666 & ~umask)


def test_copy_file_logs_the_exception_detail(tmp_path, caplog):
    """The whole point of the logging change: the errno must reach the log, not
    just the path. A bare 'Failed to copy X to Y' is what hid the EPERM."""
    src = tmp_path / 'missing.txt'
    dst = tmp_path / 'dst.txt'

    with caplog.at_level('ERROR'):
        with pytest.raises(FileNotFoundError):
            copy_file(str(src), str(dst))

    assert 'FileNotFoundError' in caplog.text


# ---------------------------------------------------------------------------
# read_bytes
# ---------------------------------------------------------------------------

def test_read_bytes_returns_bytes(tmp_path):
    p = tmp_path / 'blob.bin'
    p.write_bytes(b'\x89PNG\r\n\x1a\n')

    data = read_bytes(str(p))

    assert isinstance(data, bytes)
    assert data == b'\x89PNG\r\n\x1a\n'


def test_read_bytes_handles_undecodable_file(tmp_path):
    """The motivating case: a latin-1 0xA9 (copyright sign) in an otherwise
    ASCII vendor file. read_text() blows up under a UTF-8 locale; read_bytes
    hands the caller the bytes so it can pick the encoding."""
    p = tmp_path / 'vendor.txt'
    p.write_bytes(b'# Copyright \xa9 2024\nmesh 1 2 3\n')

    raw = read_bytes(str(p))
    assert raw.decode('latin-1') == '# Copyright \xa9 2024\nmesh 1 2 3\n'

    with pytest.raises(UnicodeDecodeError):
        raw.decode('utf-8')


def test_read_bytes_logs_the_exception_detail(tmp_path, caplog):
    with caplog.at_level('ERROR'):
        with pytest.raises(FileNotFoundError):
            read_bytes(str(tmp_path / 'nope.bin'))

    assert 'FileNotFoundError' in caplog.text


def test_read_text_still_works(tmp_path):
    p = tmp_path / 'plain.txt'
    p.write_text('line1\nline2\n')

    assert read_text(str(p)) == 'line1\nline2\n'


# ---------------------------------------------------------------------------
# __all__ hygiene
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('module', [filesystem, misc])
def test_public_functions_are_declared_in_module_all(module):
    """CLAUDE.md: 'All modules have explicit __all__ exports.' read_bytes was
    added to the package __init__ but missed here, so pin the invariant rather
    than the one name."""
    public = {
        name for name, obj in vars(module).items()
        if callable(obj) and not name.startswith('_')
        and getattr(obj, '__module__', None) == module.__name__
    }
    missing = public - set(module.__all__)
    assert not missing, '%s defines %s but omits them from __all__' % (
        module.__name__, sorted(missing))


def test_package_all_matches_importable_names():
    """Every name in the package __all__ must actually be importable from it."""
    import bg3dtools.utils.cifs_wrappers as cw

    missing = [n for n in cw.__all__ if not hasattr(cw, n)]
    assert not missing, 'cifs_wrappers.__all__ lists unimportable names: %s' % missing
