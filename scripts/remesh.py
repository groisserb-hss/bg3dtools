"""
Blender-based mesh remeshing script.

This script uses Blender's remesh and decimate modifiers to clean and
simplify meshes. Run via: `blender -b -P remesh.py`
"""

import bpy
from os.path import expanduser, basename, join, exists
from pathlib import Path
from os import makedirs

# call with:
# $ blender -b -P remesh.py
input_dir = expanduser("~/umesh/syntheticE/scans")
remesh_dir = expanduser("~/umesh/syntheticE/remeshed")

if not exists(remesh_dir):
    makedirs(remesh_dir)

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False, confirm=False)


# directory = os.fsencode(specimen_dir)
# for filename in os.listdir(directory):
pathlist = Path(input_dir).glob('**/*.ply')
for path in pathlist:

    filename = str(path)
    remeshname = join(remesh_dir, basename(filename))
    if exists(remeshname):
        continue

    Path(remeshname).touch()

    bpy.ops.import_mesh.ply(filepath=filename)
    # bpy.ops.object.select_all()
    obj_name = bpy.context.selected_objects[0].name
    bpy.context.view_layer.objects.active = bpy.data.objects[obj_name]

    bpy.ops.object.modifier_add(type='REMESH')
    bpy.context.object.modifiers["Remesh"].octree_depth=10
    bpy.context.object.modifiers["Remesh"].mode = 'SMOOTH'
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier='Remesh')

    bpy.ops.object.modifier_add(type='DECIMATE')
    bpy.context.object.modifiers["Decimate"].use_collapse_triangulate = True
    bpy.context.object.modifiers["Decimate"].ratio = 30000 / len(bpy.context.object.data.polygons)
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Decimate")

    bpy.ops.export_mesh.ply(filepath=remeshname, use_mesh_modifiers=False, use_normals=False,
                            use_uv_coords=False, use_colors=False)

    bpy.ops.object.delete(use_global=False, confirm=False)
