# Segmentation

A set of tools for editing 3D scenes reconstructed as Gaussian Splats: taking a
scanned room (as a splat/point cloud `.ply`), isolating an object inside it via
clustering, and repositioning, relighting, or extracting that object as a
standalone splat.

The core entry point is `scene_composer.py`, which combines DBSCAN-based
object selection with PCA Gaussian fitting and an interactive Open3D viewer for
moving, rotating, and relighting the selected object before exporting the
result as a standard 3DGS `.ply`. The other `scene_composer_*` scripts are
earlier iterations of the same pipeline (different rendering backends), kept
for reference — `scene_composer.py` is the one to use.

See `3D-Gaussian-Splat-Scene-Editing-Pipeline.pdf` / `.pptx` for the design
writeup behind this pipeline.

# scene_composer.py

Glue script that ties together a room/environment point cloud, DBSCAN
clustering, and PCA-based Gaussian splat fitting into one interactive
workflow: pick an object out of a scanned room, reposition it in 3D, and
export the combined (or object-only) scene as a standard 3D Gaussian
Splatting (3DGS) `.ply`.

It depends on two sibling modules living in the same folder:

| Module | Used for |
|---|---|
| `point_cloud_gui.py` | loading, DBSCAN clustering, cluster display/export helpers (`pcg`) |
| `gaussian_splat_render.py` | PCA-based Gaussian fitting (`estimate_gaussians`) and the optional photoreal splat preview window (`gsr`) |

---

## Workflow

1. **Load** an environment (room) point cloud.
2. **Clustering Workbench** — an interactive window for ground-plane
   removal, DBSCAN tuning, multi-cluster selection, an isolated preview,
   and a rectangle/lasso clean-up tool for stray points. Press **Continue**
   once your object selection is right.
3. The moment Continue is pressed, two things happen **in parallel**:
   - **Background thread** — fits Gaussians (mean / color / opacity / cov3d)
     for just the selected object's points.
   - **Main thread** — opens an Open3D viewer showing the room with the
     object's points removed, and starts the render/interaction loop
     immediately, without waiting on the fit.
4. As soon as the background fit finishes, it's converted into a movable
   "splat proxy" (a colored point cloud) and hot-swapped into the
   already-open viewer. From then on it can be translated and rotated in
   place, pivoting about its own centroid — never the world origin or the
   room.
5. Press **E** or **X** to export a standard 3DGS `.ply` (whole scene or
   object-only), ready for any Gaussian-splat renderer (SuperSplat, gsplat,
   Postshot, etc.).

### Why the viewer shows a point cloud, not "true" splats

`gaussian_splat_render.py`'s photoreal splat rasterizer runs in its own
moderngl/moderngl_window context — a separate, GPU-shader-driven render
loop that can't be embedded as a sub-widget inside an Open3D GLFW window.
For the combined, *movable* scene, the fitted Gaussians are shown as a
colored Open3D point cloud (their means/colors) — the **"splat proxy."**
Pass `--true-splat-preview` to additionally open a second, read-only window
with a true photoreal splat preview of the object right after fitting
completes.

---

## Usage

```bash
python scene_composer.py <room.ply>
python scene_composer.py                     # opens a file dialog instead
```

### Command-line flags

| Flag | Effect |
|---|---|
| `--light-gui` | Opens a tkinter side panel with lighting controls alongside the 3D viewer (see below). |
| `--no-light` | Disables lighting/shading entirely. The object is always shown and exported at its true, unshaded colors; no shadow computation runs and `--light-gui` is ignored if also passed. |
| `--true-splat-preview` | After the object finishes fitting, also opens a second, read-only window with a true photoreal Gaussian-splat render of just that object. |

---

## Controls in the combined viewer

| Key(s) | Action |
|---|---|
| Mouse | Standard Open3D orbit / pan / zoom (left/right drag, scroll) |
| `J` / `L` | Move object −X / +X |
| `I` / `K` | Move object +Y / −Y |
| `U` / `O` | Move object −Z / +Z (closer / farther) |
| `N` / `M` | Yaw object − / + (around its own centroid, Y axis) |
| `T` / `G` | Pitch object − / + (around its own centroid, X axis) |
| `R` | Reset object to its original fitted position/orientation |
| `H` | Preview shading at the object's current pose, using whichever light is active (no-op if lighting is disabled) |
| `A` / `D` | Move POINT light −X / +X (point-light mode only) |
| `S` / `W` | Move POINT light −Y / +Y (point-light mode only) |
| `F` / `V` | Move POINT light down / up (point-light mode only) |
| `B` | Toggle light type: Point (inside the room) ↔ Directional (sun-like, outside the room) |
| `E` | Export the **whole scene** (room + moved/rotated object) as one 3DGS `.ply` |
| `X` | Export just the **object's** fitted Gaussians (no room) |
| `Esc` / `Q` | Quit |

Shading (when enabled) is computed **once**, right before writing, for `E`
and `X` — not on every keypress — so moving/rotating the object stays
responsive.

---

## Lighting

Two light types are supported, switchable at any time (`--light-gui` radio
buttons, or the `B` key):

- **Point** *(default)* — a real light source positioned inside the room,
  like a lamp or bulb. Drag it anywhere within the room's bounding box with
  the panel's X/Y/Z sliders, or nudge it with `A`/`D` (left/right),
  `S`/`W` (back/forward), `F`/`V` (down/up). Gets dimmer with distance from
  the object by default (toggle in the panel). Modeled as a per-point
  ray-march against the room's own geometry to test for occluders between
  the object and the bulb.
- **Directional** — the original sun-like light, infinitely far away,
  controlled by azimuth/elevation. Modeled as a light-space KD-tree lookup:
  every object point is projected into "light space" and tested against
  the room's own points for anything closer to the light along the same
  ray.

A yellow gizmo (glowing bulb for point mode, arrow for directional mode)
tracks whichever light is active. **"Estimate light from scene"** analyzes
the room's own shading to suggest a direction, and drops the point light
inward from that direction so it starts somewhere sensible.

### Disabling lighting

Pass `--no-light` on the command line to turn the whole lighting/shading
system off:

```bash
python scene_composer.py <room.ply> --no-light
```

With lighting disabled:
- `H` (and the automatic re-shade on export) leaves the object at its true,
  unshaded colors instead of computing occlusion/shadowing.
- No shadow KD-tree / ray-march work runs at all.
- `--light-gui` is ignored (the panel is not opened) if passed alongside
  `--no-light`.
- Exported `.ply` files (via `E`/`X`) contain the object's original,
  unmodified colors.

This is controlled by `CombinedScene.lighting_enabled` (default `True`),
checked at the top of `_apply_occlusion_shading()`.

---

## Light control panel (`--light-gui`)

```bash
python scene_composer.py <room.ply> --light-gui
```

Adds a small tkinter side panel (same toolkit as the cluster picker)
alongside the *same* 3D viewer — no second scene, no duplicated geometry.
All the usual keyboard controls (move/rotate/reset/export) keep working
exactly as before; the panel just adds control over the light, and shading
auto-updates whenever you move a slider, nudge the light, or click
"Estimate light from scene."

---

## Data preservation on export

Point clouds are loaded through `SourceSplat`, which reads every original
3DGS vertex property (`scale_i`, `rot_i`, `opacity`, `f_dc_i`, `f_rest_i`)
straight from the `.ply` once, up front, and keeps them index-aligned for
the whole run. A disposable xyz+rgb view of the same data is handed to
Open3D for clustering/ground-removal/viewing — the real splat attributes
never pass through an `o3d.PointCloud` and can't be silently dropped.

On export:
- **Untouched room and object points** are written back with their
  **original** `scale`/`rot`/`opacity`/spherical-harmonic coefficients,
  byte-for-byte, whenever the source file had real splat data.
- **Only the object's `mean` (position) and `rot` (rotation)** change,
  driven by the viewer's accumulated translation/rotation. `scale`,
  `opacity`, and SH color are rigid-motion invariant and pass through
  unchanged (aside from the DC color term, which is dimmed if occlusion
  shading was applied and lighting is enabled).
- If the source `.ply` has **no** splat attributes (a plain colored point
  cloud), the tool falls back to synthetic Gaussians: PCA-fit
  mean/color/opacity/covariance for the object, and small isotropic
  Gaussians (sized from local point spacing) for the room.

---

## Key classes and functions

| Name | Role |
|---|---|
| `SourceSplat` | One-time reader of the raw `.ply`; keeps every original splat attribute index-aligned in memory for the whole run. |
| `ClusteringWorkbench` | Interactive ground-removal + DBSCAN + multi-select + clean-up UI; hands the final object selection to `on_continue`. |
| `CleanupTool` | Rectangle/lasso tool for removing stray points from the current cluster selection (2D projection: top/front/side). |
| `show_labeled_cluster_overview` | Shift-click picker for identifying cluster IDs directly on the scene, useful with hundreds of clusters. |
| `SplatJob` | Runs `estimate_gaussians` (or reuses original attributes) for the selected object on a background thread. |
| `CombinedScene` | Owns the live Open3D viewer: object transform state, both light models, occlusion shading, key bindings, and export. |
| `reposed_object_gaussians` / `export_object_as_gaussian_splat` / `export_scene_as_gaussian_splat` | Re-pose the object's Gaussians by the accumulated transform and write standard 3DGS `.ply` output, preserving original attributes wherever possible. |
| `write_gaussian_ply_encoded` | Low-level writer — takes already activation-encoded splat parameters and writes them verbatim, with no re-derivation. |

---

## Requirements

- `numpy`, `open3d`, `tkinter` (stdlib), `matplotlib`
- `scipy` and `plyfile` are auto-installed on first run if missing
  (`_ensure()`)
- `point_cloud_gui.py` and `gaussian_splat_render.py` must be present in
  the same directory
