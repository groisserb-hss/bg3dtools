# Changelog

Versions are tagged `spinescrews-X.Y.Z` — the tag marks which bg3dtools commit
the downstream **spinescrews** project is built against. See the "Releasing"
section of `CLAUDE.md`.

The 1.0.0 and 1.0.1 entries below were reconstructed from git history after the
fact; they were never written at release time.

## 1.0.2 — 2026-07-27

The first release in which `pyproject.toml` and `bg3dtools.__version__` actually
carry the version number. Both had read `0.1.0` since the initial commit, through
the 1.0.0 and 1.0.1 tags.

### Added

- **`bg3dtools.igl_compat`** — every `igl.*` call in `bg3dtools` and
  `spectral_match` now goes through one module presenting a single stable
  contract (igl 2.5.1 conventions) regardless of which binding is installed.
  igl 2.6 is a nanobind rewrite with breaking changes to names, return arity,
  return order and dtypes; a machine that drifted to 2.6.1 is what prompted this.
  Wrappers normalise index arrays to C-contiguous int64 and coordinates to
  C-contiguous float64, and preserve a leading dimension of 1 where 2.5.1
  squeezes it. Availability is exposed as the `AVAILABLE` frozenset rather than
  `hasattr` checks. Covered by `tests/test_igl_compat.py`, which runs against
  both bindings.
- **Rotation averaging on SO(3)** in `transforms_unified`: `average_quaternions`
  (Markley's eigenvector method), `average_rotations`, and
  `quat_geodesic_distance`. Batch-dimension agnostic, numpy and torch.
- **Render fallback for headless hosts** (`render/scan.py`). Open3D's EGL init
  can SIGSEGV natively — uncatchably — on a host with no usable GPU stack, so the
  offscreen tier is now gated behind a one-time subprocess crash probe whose
  verdict is cached on disk per host/env/open3d-version. On Linux a failed probe
  is retried with `EGL_PLATFORM=surfaceless` before giving up. Set
  `BG3DTOOLS_RENDER_PROBE=force` to re-probe or `=never` to skip.
- **MediaPipe GPU delegate**, opt-in, with CPU fallback — `use_gpu=True` per call
  or `MEDIAPIPE_USE_GPU=1` per process.
- `cifs_wrappers.read_bytes` (raw bytes, caller controls decoding) and
  `cifs_wrappers.image_dims` (header-only width/height read).
- `mesh.internal_edges`; optional `weights` for `rigid_reg`; `rigid_reg_robust`
  (Cauchy IRLS).
- Packaging metadata that was absent: `readme`, `license`, `license-files`,
  `authors`, `urls`. The built `PKG-INFO` had no description or license field.
- `CHANGELOG.md`, and a "Releasing" section in `CLAUDE.md` — the release process
  was previously undocumented, which is how the version drift went unnoticed.

### Fixed

- **`point_simplex_squared_distance` read the vertex matrix column-major** on igl
  2.5.1, so it returned wrong distances and wrong closest points. Verified
  against an independent reference (Ericson §5.1.5) over 1387 point/face pairs:
  F-order correct 1387/1387 at 8.9e-16, C-order correct 24/1387. This is the one
  place the compat layer deliberately changes 2.5.1's numbers. Its consumer
  `MeshProjector` was genuinely broken before the fix (max |d²−ref| 1.9 → 1.5e-08).
- `quat_geodesic_distance` used `arccos`, which loses precision catastrophically
  near identity; it now uses `atan2` on the half-angle. Guarded by tests down to
  1e-9 rad at `rtol=1e-12`.
- `copy_file` uses `shutil.copyfile`, not `copy2`. `copy2`'s `copystat()` sets
  timestamps on the destination, which the kernel permits only for the file's
  owner — on a CIFS mount with `forceuid`, that raised `EPERM` *after* the data
  had copied, and `retry_netfs` only retries `EHOSTDOWN` so it propagated out.
- `save_video` encodes to a local temp file and then copies. ffmpeg writing
  directly to a stale network mount blocks in kernel I/O without raising, and
  imageio waits for it forever.
- Network-filesystem wrappers in `filesystem.py` and the read paths in `misc.py`
  now log the caught exception, not just the path. A bare "Failed to copy X to Y"
  is what hid the `EPERM` above.
- `read_bytes` was exported from the `cifs_wrappers` package but missing from
  `misc.__all__`.
- libigl index dtypes are canonicalized to int64 at every boundary. libigl
  requires all array integer arguments of a call to share one dtype, and NumPy's
  default int is int32 on Windows but int64 elsewhere, so this only failed on
  some hosts.
- `mediapipe` was imported at module scope in the delegate module despite being
  an optional dep.

### Changed

- **The MediaPipe GPU delegate is opt-in, not the default.** As first written it
  flipped all three detectors to try-GPU-first. Reverted because the
  `except Exception` fallback cannot catch a MediaPipe GL failure that arrives as
  an absl `CHECK`/`abort()`, and because GPU and CPU TFLite inference are not
  bit-identical — the default would have silently changed landmark values
  depending on host capability.
- `package-data` narrowed from `**/*.npy` to `pose_landmarking/*.npy`.
- `build-system` no longer requires `setuptools-scm`; it was listed without a
  `[tool.setuptools_scm]` table, so it never derived anything. Now requires
  `setuptools>=77` for PEP 639.
- `environment.yml` pins `igl=2.5.1` from conda-forge.

### Removed

- Unreferenced Open3D-vs-PyTorch depth/rgb debug dumps
  (`mesh/{depth_o3d,depth_pt,rgb_o3d,rgb_pt}.npy`), which `package-data` was
  shipping in every wheel — 9.2 MB in the checkout, ~1 MB deflated per wheel.
  Nothing loaded them, and `depth_o3d`/`depth_pt` were byte-identical to each
  other, as were `rgb_o3d`/`rgb_pt`. Also `pytorch/hrnet.bkp`, a 2024 backup
  beside the live `hrnet.py`. Both recoverable from git history.

### Documentation

- Corrected a long-repeated claim that `igl.cotmatrix`/`igl.massmatrix` "return
  all zeros" on 2.5.1. They do not — disproved across 11 input variations on both
  bindings. The custom implementations in `mesh/laplace.py` are kept for real but
  different reasons, now recorded: `cotangent_weights` has the opposite sign
  (`-cotangent_weights(v,f)` equals `igl.cotmatrix(v,f)` to 1.3e-15),
  `fem_mass_matrix`'s diagonal sums to half igl's, and `lumped_vertex_areas`
  differs ~0.1% from a different lumping. No implementation changed.
- Dropped the claim that this package provides "parametric body models
  (SMPL/STAR)". It does not, and never did — what exists is interop with SMPL's
  joint convention via a SMPL→BlazePose regressor. The `bg3dtools` docstring also
  documented two subpackages, `articulated_models` and `coregistration`, that do
  not exist anywhere in the repo.
- Documented the `match` extra, absent from both README and CLAUDE.md since it
  was added, and corrected `all`'s "everything above" (it includes `match`).
- `THIRD_PARTY_NOTICES.txt` now covers the torchvision-derived reference
  detection scripts in `pytorch/detection/`, and flags `pytorch/hrnet.py` as
  **license-unresolved** — it credits "Shuchen Du" with no stated terms.

## 1.0.1 — 2026-07-08

- `rigid_reg`: optional weighted Kabsch fit, plus tests.

## 1.0.0 — 2026-07-06

First tagged revision. Notable content at that point:

- Unified diagnostic renderer (`render/scan.py`) with a tiered backend:
  OffscreenRenderer → legacy Visualizer → matplotlib Agg. The legacy Visualizer
  was made to fail *catchably* on a headless host, and a dead Open3D backend is
  probed once per process rather than per call.
- libigl index-dtype harmonization for Windows wheels (`int32` vs `int64`).
- Cosine-similarity weighting for nricp.
- libigl pinned (2.5.0, later 2.5.1) for cross-platform stability.
- tensorflow made optional; open3d preloaded in tests to dodge the
  torch-before-open3d import-order segfault on macOS.
