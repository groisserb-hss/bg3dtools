"""read_triangle_mesh returns geometry only, so it never reads an OBJ's material or texture image
(on the 3dMD data drive a frame's texture is as large as its OBJ) -- and the geometry is the same."""
import numpy as np
import trimesh

from bg3dtools.mesh.mesh_io import read_triangle_mesh

OBJ = """mtllib frame.mtl
v 0 0 0
v 1 0 0
v 0 1 0
v 0 0 1
vt 0 0
vt 1 0
vt 0 1
vt 1 1
usemtl skin
f 1/1 2/2 3/3
f 1/4 3/3 4/2
f 1/1 4/2 2/3
"""


def _textured_obj(tmp_path):
    from PIL import Image
    (tmp_path / 'frame.obj').write_text(OBJ)
    (tmp_path / 'frame.mtl').write_text('newmtl skin\nmap_Kd frame.jpg\n')
    Image.fromarray(np.full((8, 8, 3), 200, dtype=np.uint8)).save(tmp_path / 'frame.jpg')
    return tmp_path / 'frame.obj'


def test_geometry_matches_a_load_with_materials(tmp_path):
    obj = _textured_obj(tmp_path)
    full = trimesh.load(str(obj), process=False)
    v, f = read_triangle_mesh(str(obj))
    np.testing.assert_array_equal(v, full.vertices)
    np.testing.assert_array_equal(f, full.faces)


def test_neither_material_nor_texture_is_read(tmp_path, monkeypatch):
    obj = _textured_obj(tmp_path)
    asked = []
    get = trimesh.resolvers.FilePathResolver.get

    def spy(self, name):
        asked.append(name)
        return get(self, name)

    monkeypatch.setattr(trimesh.resolvers.FilePathResolver, 'get', spy)
    trimesh.load(str(obj), process=False)
    assert asked == ['frame.mtl', 'frame.jpg']      # what a plain load reads beside the OBJ
    asked.clear()
    read_triangle_mesh(str(obj))
    assert asked == []
