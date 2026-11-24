"""Where does the pypi libigl 2.5.1 Linux wheel die between a nearest-point query and the
barycentric coordinates of its answer?

On SAR-PC (2026-11-24) the interpreter segfaulted inside ``igl.barycentric_coordinates_tri``
on the first one-point call of this sequence, as SAR_annotations then ran it:

    sqrD, I, C = igl_compat.point_mesh_squared_distance(point (1, 3), verts, faces)
    tri = verts[faces[I]]
    igl.barycentric_coordinates_tri(C, tri[:, 0], tri[:, 1], tri[:, 2])

``probe_igl_barycentric.py`` then showed that a one-row barycentric call on arrays made by
numpy survives on that build. What differs in the sequence is that ``C`` comes back from
igl. This probe replays the sequence and varies one thing at a time: which arrays are
copies owned by numpy, how many points, what ran before, and whether igl is called at all.
A segfault cannot be caught, so every case runs in its own interpreter and is reported by
how that interpreter ended; a case that dies prints the last Python frames it was in.

The mesh is the STAR template when SAR_annotations is importable (the mesh of the crash),
else an icosphere. ``--mesh file.npz`` (arrays ``verts`` (N, 3), ``faces`` (M, 3)) names another.

Run:  python scripts/probe_igl_nearest_then_barycentric.py [--repeat 20]
"""
import argparse
import subprocess
import sys

SETUP = r'''
import faulthandler, sys
faulthandler.enable()
import numpy as np
{preload}
import igl
from bg3dtools import igl_compat as ic

RAW_BARY = getattr(igl, "barycentric_coordinates_tri", None) or igl.barycentric_coordinates
MESH = {mesh!r}


def load_mesh():
    if MESH:
        with np.load(MESH) as z:
            return np.asarray(z["verts"], np.float64), np.asarray(z["faces"], np.int64), None
    try:
        from sar_fiducials.star_prior import load_star_template
    except ImportError:
        return (*icosphere(), None)
    star = load_star_template()
    verts = np.asarray(star["v_template"], np.float64) * 1000.0
    sites = np.asarray(star["lm_map"], np.float64).T @ verts
    return verts, np.asarray(star["faces"], np.int64), sites


def icosphere(levels=4, radius=300.0):
    """A sphere of 20 * 4**levels faces, mm."""
    t = (1 + 5 ** 0.5) / 2
    verts = [np.array(x, np.float64) for x in ([-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0], [0, -1, t], [0, 1, t],
                                               [0, -1, -t], [0, 1, -t], [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1])]
    faces = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6),
             (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10),
             (8, 6, 7), (9, 8, 1)]
    for _ in range(levels):
        middle, split = {{}}, []
        for tri in faces:
            mids = []
            for i, j in ((0, 1), (1, 2), (2, 0)):
                key = (min(tri[i], tri[j]), max(tri[i], tri[j]))
                if key not in middle:
                    middle[key] = len(verts)
                    verts.append((verts[key[0]] + verts[key[1]]) / 2)
                mids.append(middle[key])
            a, b, c = tri
            ab, bc, ca = mids
            split += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = split
    v = np.array(verts)
    return radius * v / np.linalg.norm(v, axis=1, keepdims=True), np.array(faces, np.int64)


VERTS, FACES, SITES = load_mesh()
if SITES is None:
    SITES = VERTS[:: max(len(VERTS) // 15, 1)][:15] * 1.01
REPEAT = {repeat}


def describe(name, a):
    a = np.asarray(a)
    return "%s shape %s %s own=%s base=%s C=%s F=%s" % (name, a.shape, a.dtype, a.flags.owndata,
        type(a.base).__name__, a.flags.c_contiguous, a.flags.f_contiguous)


def old_bary_wrapper(p, a, b, c):
    """igl_compat.barycentric_coordinates_tri as it was on 2026-11-24, before one point was padded to two rows."""
    p, a, b, c = (np.ascontiguousarray(x, dtype=np.float64) for x in (p, a, b, c))
    if p.ndim == 1:
        p, a, b, c = p[None, :], a[None, :], b[None, :], c[None, :]
    return np.ascontiguousarray(RAW_BARY(p, a, b, c), dtype=np.float64).reshape(-1, 3)


def numpy_bary(p, a, b, c):
    e1, e2, d = b - a, c - a, p - a
    d11, d12, d22 = (e1 * e1).sum(1), (e1 * e2).sum(1), (e2 * e2).sum(1)
    p1, p2 = (d * e1).sum(1), (d * e2).sum(1)
    det = d11 * d22 - d12 * d12
    v, w = (d22 * p1 - d12 * p2) / det, (d11 * p2 - d12 * p1) / det
    return np.column_stack([1 - v - w, v, w])


def sequence(points, bary=old_bary_wrapper, copy=(), nearest=ic.point_mesh_squared_distance, verts=None, weights=False):
    """The crash's sequence for ``points (P, 3)``; ``copy`` names what is copied into numpy-owned arrays first."""
    verts = VERTS if verts is None else np.asarray(verts, np.float64)
    points = np.asarray(points, np.float64).reshape(-1, 3)
    sq, fidx, closest = nearest(points, verts, FACES)
    fidx = np.asarray(fidx, np.int64).reshape(-1)
    closest = np.asarray(closest, np.float64).reshape(-1, 3)
    tri = verts[FACES[fidx]]
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    if "closest" in copy:
        closest = np.array(closest, dtype=np.float64, order="C", copy=True)
    if "corners" in copy:
        a, b, c = (np.array(x, dtype=np.float64, order="C", copy=True) for x in (a, b, c))
    w = bary(closest, a, b, c)
    back = np.einsum("pk,pkc->pc", w, tri)
    return (fidx, w) if weights else float(np.abs(back - closest).max())


def session_replay(root, session):
    """The run's own calls for one session, in its order: each labelled frame's marks onto the frame's
    registration (many points), then each landmark's median template position onto the template (one point)."""
    from sar_fiducials.identify import seeds as S
    from sar_fiducials.labeling.store import LabelStore
    lab = S.seed_labels(LabelStore().load(session))
    reg_of = S.registration_reader(root + "/" + session)
    by_frame, tpl = {{}}, {{}}
    for m in lab.marks:
        by_frame.setdefault((m.scan, m.label), []).append(m)
    for key, marks in by_frame.items():
        reg = reg_of(*key)
        if reg is None:
            continue
        fidx, w = sequence(np.array([m.pos for m in marks]), verts=reg, weights=True)
        for m, p in zip(marks, np.einsum("pk,pkc->pc", w, VERTS[FACES[fidx]])):
            tpl.setdefault(m.name, []).append(p)
    worst = max(sequence(np.median(np.array(pts), axis=0)[None]) for pts in tpl.values())
    return "%d frames, %d one-point calls, worst rebuild error %.2g" % (len(by_frame), len(tpl), worst)


def each_site(**kw):
    worst = 0.0
    for _ in range(REPEAT):
        for s in SITES:
            worst = max(worst, sequence(s[None], **kw))
    return "%d calls, worst rebuild error %.2g" % (REPEAT * len(SITES), worst)
'''

CASES = {
    # what igl hands back
    "what nearest-point returns, raw igl, 1 point":
        "out = igl.point_mesh_squared_distance(SITES[:1].copy(), VERTS, FACES)\n"
        "print('; '.join(describe(n, x) for n, x in zip(('sqrD', 'I', 'C'), out)))",
    "what nearest-point returns, wrapper, 1 point":
        "out = ic.point_mesh_squared_distance(SITES[:1].copy(), VERTS, FACES)\n"
        "print('; '.join(describe(n, x) for n, x in zip(('sqrD', 'I', 'C'), out)))",
    # the crash, and one thing changed at a time
    "THE CRASH'S SEQUENCE: 1 point, old wrapper": "print(each_site())",
    "  closest point copied first": "print(each_site(copy=('closest',)))",
    "  triangle corners copied first": "print(each_site(copy=('corners',)))",
    "  both copied first": "print(each_site(copy=('closest', 'corners')))",
    "  raw igl barycentric, no wrapper": "print(each_site(bary=lambda p, a, b, c: np.asarray(RAW_BARY(p, a, b, c)).reshape(-1, 3)))",
    "  raw igl nearest and barycentric": "print(each_site(nearest=igl.point_mesh_squared_distance, bary=lambda p, a, b, c: np.asarray(RAW_BARY(p, a, b, c)).reshape(-1, 3)))",
    "  today's wrapper (1 point as 2 rows)": "print(each_site(bary=ic.barycentric_coordinates_tri))",
    "  weights by arithmetic, no igl barycentric": "print(each_site(bary=numpy_bary))",
    # how many points, and what ran before
    "all sites in one call, old wrapper": "print('worst rebuild error %.2g' % max(sequence(SITES) for _ in range(REPEAT)))",
    "2 points per call, old wrapper": "print('worst rebuild error %.2g' % max(sequence(SITES[i:i + 2]) for _ in range(REPEAT) for i in range(len(SITES) - 1)))",
    "many points first, then 1 point (the run's order)": "sequence(SITES); print(each_site())",
    # is the nearest-point query itself sound for one point?
    "nearest-point alone, 1 point, then heavy allocation":
        "for _ in range(REPEAT):\n"
        "    for s in SITES:\n"
        "        d, i, c = ic.point_mesh_squared_distance(s[None], VERTS, FACES)\n"
        "        junk = [np.random.rand(n) for n in (3, 30, 300, 3000)]\n"
        "        assert np.isfinite(c).all() and 0 <= int(i[0]) < len(FACES)\n"
        "print('%d calls, answers finite and in range' % (REPEAT * len(SITES)))",
}
SESSION_CASE = "the run's own calls for {session} (its marks, its registrations)"
PRELOADS = {"": "", " [torch, scipy, sklearn loaded]": "import torch, scipy.spatial, scipy.optimize, sklearn.neighbors"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeat", type=int, default=20, help="times each case walks its query points (default 20)")
    ap.add_argument("--root", default="", help="3dMD root: also replay a session's own calls (needs SAR_annotations)")
    ap.add_argument("--session", default="200001a", help="the session to replay with --root (default 200001a, the crash's)")
    ap.add_argument("--mesh", default="", help="npz with verts (N,3) and faces (M,3); default: the STAR template, else an icosphere")
    args = ap.parse_args()
    from importlib import metadata
    try:
        build = f"pypi libigl {metadata.version('libigl')}"
    except metadata.PackageNotFoundError:
        build = "no libigl distribution (a conda igl package)"
    print(f"{sys.platform}, python {sys.version.split()[0]}, {build}")
    died = 0
    cases = dict(CASES)
    if args.root:
        cases[SESSION_CASE.format(session=args.session)] = f"print(session_replay({args.root!r}, {args.session!r}))"
    for tag, preload in PRELOADS.items():
        for name, body in cases.items():
            if tag and not name.lstrip().startswith(("THE CRASH", "many points first", "the run's own")):
                continue
            code = SETUP.format(preload=preload, mesh=args.mesh, repeat=args.repeat) + body
            run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
            ok = run.returncode == 0
            died += not ok
            print(f"  {'ok  ' if ok else 'DIED'}  {name + tag}")
            if ok:
                print(f"          {run.stdout.strip()}")
            else:
                frames = [ln for ln in run.stderr.splitlines() if ln.startswith(("Fatal", "  File")) or "Error" in ln][:6]
                print(f"          exit {run.returncode}" + "".join(f"\n          {ln.strip()}" for ln in frames))
    return 1 if died else 0


if __name__ == "__main__":
    raise SystemExit(main())
