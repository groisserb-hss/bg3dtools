"""Which call shapes of igl's barycentric coordinates does the installed build survive?

The pypi libigl 2.5.1 Linux wheel segfaulted on a one-point call (SAR-PC, 2026-11-24). A
segfault cannot be caught, so each case runs in its own interpreter and is reported by how
that interpreter ended. ``raw`` cases call igl directly; ``wrapper`` goes through
``bg3dtools.igl_compat``, which sends one point as two rows.

Run:  python scripts/probe_igl_barycentric.py
"""
import subprocess
import sys

SETUP = """
import faulthandler, numpy as np, igl
faulthandler.enable()
f = getattr(igl, "barycentric_coordinates_tri", None) or igl.barycentric_coordinates
tri = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
q = np.array([[0.25, 0.25, 0.]])
def rows(n, order="C"):
    return [np.array(np.repeat(x[None] if x.ndim == 1 else x, n, axis=0), dtype=np.float64, order=order) for x in (q[0], *tri)]
"""
CASES = {
    "raw, 5 rows": "print(np.asarray(f(*rows(5)))[0])",
    "raw, 2 rows": "print(np.asarray(f(*rows(2)))[0])",
    "raw, 1 row, C order": "print(np.asarray(f(*rows(1))).ravel())",
    "raw, 1 row, Fortran order": "print(np.asarray(f(*rows(1, 'F'))).ravel())",
    "raw, 1 row, after a 5-row call": "f(*rows(5)); print(np.asarray(f(*rows(1))).ravel())",
    "wrapper, 1 point": "from bg3dtools import igl_compat as ic; print(ic.barycentric_coordinates_tri(*rows(1))[0])",
}


def main() -> int:
    from importlib import metadata
    try:
        build = f"pypi libigl {metadata.version('libigl')}"
    except metadata.PackageNotFoundError:
        build = "no libigl distribution (a conda igl package)"
    print(f"{sys.platform}, python {sys.version.split()[0]}, {build}")
    died = 0
    for name, body in CASES.items():
        run = subprocess.run([sys.executable, "-c", SETUP + body], capture_output=True, text=True)
        ok = run.returncode == 0
        died += not ok
        print(f"  {'ok  ' if ok else 'DIED'}  {name:32s} {run.stdout.strip() if ok else f'exit {run.returncode}'}")
    return 1 if died else 0


if __name__ == "__main__":
    raise SystemExit(main())
