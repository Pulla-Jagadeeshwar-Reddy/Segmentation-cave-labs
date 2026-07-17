"""
scene_composer.py
─────────────────
Glue script that ties together:

    point_cloud_gui.py        → loading + DBSCAN clustering + export
    gaussian_splat_render.py  → estimate_gaussians() (PCA-based Gaussian fit)
    point_cloud_render.py     → (not directly used, kept for reference/CLI use)

Workflow
========
1.  Load an environment (room) point cloud and cluster it (DBSCAN).
2.  Show a picker GUI. The user selects ONE cluster ("the object") and
    presses "Continue".
3.  The moment Continue is pressed, two things happen *in parallel*:
        Thread A (background) → fits Gaussians (mean/color/opacity/cov3d)
                                  for just the object's points.
        Main thread            → opens an Open3D viewer showing the room
                                  with the object's points removed, and
                                  starts the render/interaction loop
                                  immediately (it does not wait on A).
4.  As soon as Thread A finishes, its result is converted into a single
    movable object (a PointCloud "splat proxy") and hot-swapped into the
    already-open viewer. From then on it can be translated and rotated
    in place, with all transforms pivoting about the object's own
    centroid (not the world origin / not the room).

Note on "true" Gaussian rendering
----------------------------------
gaussian_splat_render.py's photoreal splat rasterizer runs in its own
moderngl/moderngl_window context — a different, GPU-shader-driven render
loop that can't simply be embedded as a sub-widget inside an Open3D
GLFW window. So for the *combined, movable* scene we render the fitted
Gaussians as a colored Open3D point cloud (their means/colors) — this is
the "splat proxy". If you want to additionally eyeball the same object
as a true photoreal splat in its own window, pass --true-splat-preview
and a second window will be opened (read-only, not part of the movable
scene) right after fitting completes.

Controls in the combined viewer
--------------------------------
  Mouse              Standard Open3D orbit / pan / zoom (left/right drag, scroll)
  J / L              Move object  -X / +X
  I / K              Move object  +Y / -Y
  U / O              Move object  -Z / +Z   (closer / farther)
  N / M              Yaw   object  -  / +   (around its own centroid, Y axis)
  T / G              Pitch object  -  / +   (around its own centroid, X axis)
  R                  Reset object to its original fitted position/orientation
  H                  Preview shadow shading at the object's CURRENT pose,
                     using whichever light is active (point or directional)
                     (recomputes + shows it in the viewer without exporting)
                     [In --light-gui mode, shading also auto-updates whenever
                      you move a slider, nudge the light, or click "Estimate
                      light from scene".]
  A / D              Move POINT light -X / +X   (only in point-light mode)
  S / W              Move POINT light -Y / +Y   (only in point-light mode)
  F / V              Move POINT light down / up (only in point-light mode)
  B                  Toggle light type: Point (inside the room) ↔
                     Directional (sun-like, outside the room)
  E                  Export the WHOLE SCENE (room + moved/rotated object) as
                     one standard 3DGS-format .ply, ready for a Gaussian-
                     splat renderer (SuperSplat, gsplat, Postshot, etc.)
                     Occlusion shading is (re)computed once, automatically,
                     right before writing.
  X                  Export just the object's fitted Gaussians (no room) —
                     same one-shot shading pass as E.
  ESC / Q            Quit

Usage
-----
    python scene_composer.py <room.ply>
    python scene_composer.py                     # opens a file dialog

Light control panel
--------------------
    python scene_composer.py <room.ply> --light-gui

Adds a small tkinter side panel (same toolkit already used for the
cluster picker) alongside the SAME 3D viewer — no second scene, no
duplicated geometry. Two light types are supported, switchable at any
time (radio buttons in the panel, or the B key):

  * Point   — a real light source positioned INSIDE the room, like a
              lamp or bulb. Drag it anywhere within the room's own
              bounding box with the X/Y/Z sliders, or nudge it with
              A/D (left/right), S/W (back/forward), F/V (down/up).
              Gets dimmer with distance from the object by default
              ("Dim with distance from the bulb" checkbox). This is
              the default mode.
  * Directional — the original sun-like light infinitely far away,
              controlled by azimuth/elevation sliders, same as before.

A yellow gizmo tracks whichever light is active — a small glowing bulb
(point mode) or an arrow (directional mode). "Estimate light from
scene" still analyzes the room's own shading to suggest an angle, and
now also drops the point light inward from that direction so it starts
somewhere sensible inside the room. All the usual keyboard controls
(move/rotate/reset/export) keep working exactly as before; the panel
just adds control over the light. See CombinedScene.run(light_gui=True)
/ estimate_light_from_scene() / build_light_gizmo() / move_light().
"""

import os
import sys
import copy
import time
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import open3d as o3d


# ════════════════════════════════════════════════════════════════════════════
# CHANGE 19 — silence a known-benign tkinter cleanup race
# ────────────────────────────────────────────────────────────────────────────
# On Python 3.12+, tkinter Variable/Image __del__ can raise
# "RuntimeError: main thread is not in main loop" when a Var (tk.DoubleVar,
# tk.StringVar, ...) is garbage-collected while Tcl doesn't consider itself
# "in mainloop". That's exactly our situation on purpose: this app never
# calls root.mainloop() — it pumps tkinter manually (panel.update_idletasks()
# / panel.update(), see CombinedScene.run()) so the light-gui panel can share
# the process with Open3D's own render loop instead of blocking on it. Python
# already catches the error itself (note the "Exception ignored in:" prefix —
# that's Python telling you it swallowed it); it does NOT stop the viewer,
# the panel, or lose any state — it's pure __del__ cleanup noise. This just
# keeps it out of the terminal. Any *other* unraisable exception still prints
# exactly as before.
def _silence_benign_tkinter_del_noise():
    default_hook = sys.unraisablehook

    def _hook(unraisable):
        exc = unraisable.exc_value
        if (isinstance(exc, RuntimeError)
                and "main thread is not in main loop" in str(exc)):
            return
        default_hook(unraisable)

    sys.unraisablehook = _hook


_silence_benign_tkinter_del_noise()

def _ensure(pkg):
    import importlib, subprocess
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pkg} ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("scipy")
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

_ensure("plyfile")
from plyfile import PlyData, PlyElement

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import point_cloud_gui as pcg          # load_pcd, remove_ground, cluster_dbscan, make_palette, export_ply
import gaussian_splat_render as gsr    # estimate_gaussians (slow fallback), GaussianSplatWindow (optional true preview)


# ════════════════════════════════════════════════════════════════════════════
# CHANGE 1 — preserve-the-original loader
# ────────────────────────────────────────────────────────────────────────────
# Everything below this point exists so we NEVER have to reconstruct Gaussian
# shape/opacity/color from scratch for a point we didn't touch. We read the
# raw .ply once with `plyfile` (not o3d.io.read_point_cloud, which only knows
# about xyz/rgb/normals and silently drops scale/rot/opacity/SH), keep every
# field in a flat, index-aligned store, and only ever hand Open3D a
# disposable xyz+rgb "view" copy for clustering/picking/viewing. Any mask or
# index array produced against that view copy slices this store directly —
# no re-fitting, no re-guessing.

SH_C0 = 0.28209479177387814  # Y_0^0 spherical-harmonic constant (also used later)


class SourceSplat:
    """
    Reads path once. `self.xyz` / `self.rgb_display` are what everything
    else in this file should hand to Open3D for viewing/clustering. Every
    other array here (scale/rot/opacity/f_dc/f_rest) is the ORIGINAL,
    already splat-activation-encoded data straight off disk (i.e. `opacity`
    is still pre-sigmoid, `scale` is still pre-exp, exactly as a 3DGS file
    stores it) — so on export these can be written back byte-for-byte
    unchanged for any point that wasn't part of the edited object.
    """
    def __init__(self, path):
        ply = PlyData.read(path)
        v = ply["vertex"].data
        names = set(v.dtype.names)

        self.n = len(v)
        self.xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float64)

        self.has_splat_attrs = (
            all(f"scale_{i}" in names for i in range(3)) and
            all(f"rot_{i}"   in names for i in range(4)) and
            "opacity" in names and
            all(f"f_dc_{i}"  in names for i in range(3))
        )

        if self.has_splat_attrs:
            self.scale_raw   = np.stack([v[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float32)
            self.rot_raw     = np.stack([v[f"rot_{i}"]   for i in range(4)], axis=1).astype(np.float32)
            self.opacity_raw = v["opacity"].astype(np.float32).copy()
            self.f_dc        = np.stack([v[f"f_dc_{i}"] for i in range(3)], axis=1).astype(np.float32)
            n_rest = sum(1 for nm in names if nm.startswith("f_rest_"))
            if n_rest > 0:
                self.f_rest = np.stack(
                    [v[f"f_rest_{i}"] for i in range(n_rest)], axis=1).astype(np.float32)
            else:
                self.f_rest = np.zeros((self.n, 45), dtype=np.float32)
            # decode DC term only, purely so Open3D has something reasonable
            # to display while clustering/picking — never written back out.
            rgb = np.clip(SH_C0 * self.f_dc.astype(np.float64) + 0.5, 0.0, 1.0)
        elif "red" in names:
            rgb = np.stack([v["red"], v["green"], v["blue"]], axis=1).astype(np.float64) / 255.0
        else:
            rgb = np.full((self.n, 3), 0.7)

        self.rgb_display = rgb

        if self.has_splat_attrs:
            print(f"[source] Loaded {self.n:,} points WITH original splat "
                  f"attributes (scale/rot/opacity/SH) — these will be "
                  f"preserved exactly for every untouched point.")
        else:
            print(f"[source] Loaded {self.n:,} points with NO splat "
                  f"attributes found — falling back to synthetic Gaussian "
                  f"fitting for everything (this is the old, lossy path).")

    def view_pcd(self):
        """Disposable o3d.PointCloud for clustering/picking/viewing only."""
        p = o3d.geometry.PointCloud()
        p.points = o3d.utility.Vector3dVector(self.xyz)
        p.colors = o3d.utility.Vector3dVector(self.rgb_display)
        return p


def map_by_nearest(pts_ref: np.ndarray, pts_subset: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Returns, for each row of pts_subset, its index into pts_ref (exact/near
    match). Used ONLY to translate a filtered point set (e.g. whatever
    remove_ground()/o3d round-tripping hands back) into indices against our
    untouched SourceSplat store — never to reconstruct data, only to find
    "which original row is this."
    """
    tree = cKDTree(pts_ref)
    dists, idx = tree.query(pts_subset, k=1, workers=-1)
    bad = dists > eps
    if bad.any():
        print(f"[warn] {int(bad.sum()):,}/{len(pts_subset):,} points could not be "
              f"matched back to the source within {eps} — check for a lossy "
              f"round-trip (voxel downsampling, dtype rounding) upstream.")
    return idx


# ════════════════════════════════════════════════════════════════════════════
# Step 0 — identify which cluster ID is your object (huge scenes, many clusters)
# ════════════════════════════════════════════════════════════════════════════
#
# With 470+ clusters, picking blindly from a text list is impractical. This
# opens a window showing the whole scene colored by cluster; shift-click on
# the object you want and press Q. The cluster ID(s) under your click are
# fed back as a pre-selection in the list that follows, so you don't have
# to go hunting through hundreds of rows by number.

def show_labeled_cluster_overview(pcd, labels, n_clusters, palette, cluster_meta):
    """
    Identify cluster IDs by shift-clicking directly on the object, using the
    SAME legacy GLFW renderer as the rest of this script (VisualizerWithEditing).

    Why not the floating-3D-label version: that used Open3D's "new GUI"
    stack (O3DVisualizer), which is backed by Filament/WGL. In some
    environments (Remote Desktop sessions, certain VMs, some GPU driver
    setups) Filament simply cannot get a valid OpenGL context there at all —
    every redraw attempt (including on mouse-move) fails again, which is why
    you saw a blank window spamming wglMakeCurrent errors on hover. That's
    not a backend-mixing problem, it's that backend being unusable in this
    environment. The legacy GLFW Visualizer used everywhere else in this
    pipeline doesn't have that problem, so this step now uses it too.

    Controls:
        Shift + Left-click   pick a point on the object you want
        Q / Esc              close the window when done picking
    Returns the sorted list of distinct cluster IDs you clicked on (the
    picker GUI will pre-select the first one for you).
    """
    display = pcg.build_display_pcd(pcd, labels, n_clusters, palette, set(), False)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Shift+Click your object's points, then press Q",
        width=1280, height=800)
    vis.add_geometry(display)
    vis.run()           # blocks until the user closes the window
    vis.destroy_window()

    picked_idx = vis.get_picked_points()
    cids = sorted({int(labels[i]) for i in picked_idx if labels[i] != -1})
    if cids:
        print(f"[overview] You clicked on cluster(s): {cids}")
    else:
        print("[overview] No points picked (or only noise points). "
              "Pick a cluster manually from the list.")
    return cids


def build_cluster_meta(pcd, labels, n_clusters):
    pts = np.asarray(pcd.points)
    meta = {}
    for cid in range(n_clusters):
        idx = np.where(labels == cid)[0]
        meta[cid] = {"indices": idx,
                     "n_points": int(len(idx)),
                     "centroid": pts[idx].mean(axis=0)}
    return meta


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Clustering Workbench: ground-removal + DBSCAN controls, multi-select
#          cluster picker, isolated preview window, and a rectangle/lasso
#          clean-up tool for stray points
# ════════════════════════════════════════════════════════════════════════════
#
# Replaces the old single-shot "cluster once, pick exactly one cluster" flow
# with an interactive loop:
#   1. Toggle ground-plane removal on/off (+ threshold slider) and re-cluster.
#   2. Tune DBSCAN eps / min_points with sliders and re-cluster as many times
#      as you like.
#   3. Multi-select any number of clusters — they'll be moved together as one
#      object.
#   4. Preview the current selection, isolated, in its own read-only 3D window.
#   5. Clean up stray points in the selection with a rectangle + lasso tool.
# Pressing Continue hands (pcd, labels, n_clusters, obj_mask, nonground_src_idx,
# ground_src_idx, ground_pcd) onward — same shape of handoff the old picker
# made, just with `obj_mask` now representing "every point you actually
# want", however many clusters and clean-up passes it took to get there.

_ensure("matplotlib")
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path as MplPath


class CleanupTool:
    """
    Rectangle + lasso combo for removing stray points from the currently
    selected cluster(s). Nothing is deleted from the scene — points you mark
    and remove here are simply excluded from the final object mask, so they
    stay part of the room. Works on a 2D projection of the selection (top/
    front/side) since that's what rectangle/lasso selection needs.
    """
    PLANES = {"Top (X/Y)": (0, 1), "Front (X/Z)": (0, 2), "Side (Y/Z)": (1, 2)}

    def __init__(self, parent, pts, colors, global_idx, on_done):
        """
        pts, colors : Nx3 arrays for the CURRENT selection only.
        global_idx  : row-index of each of these N points back into the
                       workbench's `pcd` — what on_done() reports as removed.
        on_done     : callback(removed_global_idx: np.ndarray), fired once,
                       when the tool window is closed.
        """
        self.pts = pts
        self.colors = colors if colors is not None else np.tile([0.9, 0.6, 0.15], (len(pts), 1))
        self.global_idx = global_idx
        self.on_done = on_done
        self.marked = set()     # local indices, currently boxed/lassoed
        self.removed = set()    # local indices, confirmed removed

        self.win = tk.Toplevel(parent)
        self.win.title("Clean up selection — rectangle or lasso, then Remove")
        self.win.geometry("980x760")
        self.win.protocol("WM_DELETE_WINDOW", self._finish)

        top = tk.Frame(self.win); top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Projection:").pack(side="left")
        self.plane_var = tk.StringVar(value="Top (X/Y)")
        tk.OptionMenu(top, self.plane_var, *self.PLANES.keys(),
                      command=lambda _=None: self._redraw()).pack(side="left", padx=6)

        self.mode_var = tk.StringVar(value="rect")
        tk.Radiobutton(top, text="Rectangle", variable=self.mode_var, value="rect",
                        command=self._set_mode).pack(side="left", padx=(20, 4))
        tk.Radiobutton(top, text="Lasso", variable=self.mode_var, value="lasso",
                        command=self._set_mode).pack(side="left", padx=4)

        tk.Button(top, text="Clear marks", command=self._clear_marks).pack(side="left", padx=(20, 4))
        tk.Button(top, text="Remove marked points", bg="#5f1e1e", fg="white",
                  command=self._remove_marked).pack(side="left", padx=4)
        tk.Button(top, text="Done", command=self._finish).pack(side="right", padx=4)

        self.status = tk.Label(self.win, text="", anchor="w")
        self.status.pack(fill="x", padx=8)

        self.fig, self.ax = plt.subplots(figsize=(8, 7))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.win)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=8)

        self._redraw()

    # ── drawing ──────────────────────────────────────────────────────────
    def _live_mask(self):
        """Bool mask over self.pts of points still 'live' (not removed)."""
        live = np.ones(len(self.pts), dtype=bool)
        if self.removed:
            live[list(self.removed)] = False
        return live

    def _redraw(self):
        self.ax.clear()
        a, b = self.PLANES[self.plane_var.get()]
        live = self._live_mask()
        xy_live = np.stack([self.pts[live, a], self.pts[live, b]], axis=1)
        cols_live = self.colors[live]
        self._scatter_idx = np.where(live)[0]   # local idx of each plotted point
        self.ax.scatter(xy_live[:, 0], xy_live[:, 1], c=cols_live, s=4)

        marked_live = [i for i in self.marked if live[i]]
        if marked_live:
            mk = np.stack([self.pts[marked_live, a], self.pts[marked_live, b]], axis=1)
            self.ax.scatter(mk[:, 0], mk[:, 1], facecolors="none",
                             edgecolors="red", s=30, linewidths=1.2)

        self.ax.set_aspect("equal", adjustable="datalim")
        self.ax.set_title(f"{int(live.sum()):,} pts shown  |  {len(self.marked):,} marked  |  "
                           f"{len(self.removed):,} removed so far")
        self._install_selectors()
        self.canvas.draw_idle()
        self.status.config(text="Drag to select (rectangle or lasso mode above), "
                                 "then 'Remove marked points'. Close window when done.")

    def _install_selectors(self):
        self._rect_sel = RectangleSelector(
            self.ax, self._on_rect, useblit=True, button=[1], interactive=False)
        self._lasso_sel = LassoSelector(self.ax, self._on_lasso, button=[1])
        self._set_mode()

    def _set_mode(self):
        if not hasattr(self, "_rect_sel"):
            return
        is_rect = self.mode_var.get() == "rect"
        self._rect_sel.set_active(is_rect)
        self._lasso_sel.set_active(not is_rect)

    # ── selection callbacks ─────────────────────────────────────────────
    def _current_xy(self):
        a, b = self.PLANES[self.plane_var.get()]
        return np.stack([self.pts[self._scatter_idx, a],
                          self.pts[self._scatter_idx, b]], axis=1)

    def _on_rect(self, eclick, erelease):
        x0, x1 = sorted([eclick.xdata, erelease.xdata])
        y0, y1 = sorted([eclick.ydata, erelease.ydata])
        xy = self._current_xy()
        inside = (xy[:, 0] >= x0) & (xy[:, 0] <= x1) & (xy[:, 1] >= y0) & (xy[:, 1] <= y1)
        self.marked.update(self._scatter_idx[inside].tolist())
        self._redraw()

    def _on_lasso(self, verts):
        path = MplPath(verts)
        xy = self._current_xy()
        inside = path.contains_points(xy)
        self.marked.update(self._scatter_idx[inside].tolist())
        self._redraw()

    def _clear_marks(self):
        self.marked.clear()
        self._redraw()

    def _remove_marked(self):
        self.removed.update(self.marked)
        self.marked.clear()
        self._redraw()

    def _finish(self):
        removed_global = (self.global_idx[list(self.removed)]
                           if self.removed else np.array([], dtype=np.int64))
        plt.close(self.fig)
        self.win.destroy()
        self.on_done(removed_global)


class ClusteringWorkbench:
    """
    Interactive replacement for the old single-shot "cluster once, pick one
    cluster" flow. See the Step 1 header comment above for the full feature
    list. `on_continue(pcd, labels, n_clusters, obj_mask, nonground_src_idx,
    ground_src_idx, ground_pcd)` fires once, when Continue is pressed.
    """

    def __init__(self, root, pcd_full, on_continue):
        self.root = root
        self.pcd_full = pcd_full
        self.on_continue_cb = on_continue

        # Current working state — replaced wholesale every time
        # "Run clustering" is pressed.
        self.pcd = pcd_full
        self.labels = None
        self.n_clusters = 0
        self.palette = None
        self.cluster_meta = {}
        self.nonground_src_idx = np.arange(len(np.asarray(pcd_full.points)))
        self.ground_src_idx = np.array([], dtype=np.int64)
        self.ground_pcd = o3d.geometry.PointCloud()
        self.removed_idx = set()   # global row indices into self.pcd, cleaned out

        root.title("Clustering Workbench")
        root.configure(bg="#14151f")
        root.geometry("420x780")
        self._build_ui()
        self._run_clustering()   # initial pass with default settings

    # ── UI ───────────────────────────────────────────────────────────────
    def _build_ui(self):
        BG, FG = "#14151f", "#e0e4f5"
        f = tk.Frame(self.root, bg=BG, padx=12, pady=10)
        f.pack(fill="both", expand=True)

        # -- Ground removal ---------------------------------------------------
        tk.Label(f, text="GROUND REMOVAL", bg=BG, fg="#6a7fc1",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.ground_var = tk.BooleanVar(value=False)
        tk.Checkbutton(f, text="Enable (plane-segment + strip floor)",
                        variable=self.ground_var, bg=BG, fg=FG,
                        selectcolor="#1e1f2e", activebackground=BG,
                        command=self._on_ground_toggle).pack(anchor="w")
        self.ground_thresh_var = tk.DoubleVar(value=0.10)
        self.ground_slider = tk.Scale(
            f, from_=0.01, to=1.0, resolution=0.01, orient="horizontal",
            variable=self.ground_thresh_var, label="Plane distance threshold",
            bg=BG, fg=FG, troughcolor="#1e1f2e", highlightthickness=0,
            state="disabled")
        self.ground_slider.pack(fill="x", pady=(2, 10))

        # -- DBSCAN -------------------------------------------------------------
        tk.Label(f, text="DBSCAN", bg=BG, fg="#6a7fc1",
                 font=("Segoe UI", 8, "bold")).pack(anchor="w")
        self.auto_eps_var = tk.BooleanVar(value=True)
        tk.Checkbutton(f, text="Auto eps (recommended starting point)",
                        variable=self.auto_eps_var, bg=BG, fg=FG,
                        selectcolor="#1e1f2e", activebackground=BG,
                        command=self._on_auto_eps_toggle).pack(anchor="w")
        self.eps_var = tk.DoubleVar(value=0.10)
        self.eps_slider = tk.Scale(
            f, from_=0.01, to=2.0, resolution=0.01, orient="horizontal",
            variable=self.eps_var, label="eps (neighborhood radius)",
            bg=BG, fg=FG, troughcolor="#1e1f2e", highlightthickness=0,
            state="disabled")
        self.eps_slider.pack(fill="x", pady=(2, 6))

        self.min_pts_var = tk.IntVar(value=60)
        self.min_pts_slider = tk.Scale(
            f, from_=5, to=300, resolution=1, orient="horizontal",
            variable=self.min_pts_var, label="min_points",
            bg=BG, fg=FG, troughcolor="#1e1f2e", highlightthickness=0)
        self.min_pts_slider.pack(fill="x", pady=(2, 8))

        tk.Button(f, text="Run clustering ▶", command=self._run_clustering,
                  bg="#1e3a5f", fg=FG, activebackground="#2a4d7a",
                  relief="flat", padx=8, pady=6, cursor="hand2").pack(fill="x", pady=(0, 10))

        # -- Cluster list ---------------------------------------------------------
        tk.Label(f, text="CLUSTERS (ctrl/shift-click to select more than one)",
                 bg=BG, fg="#6a7fc1", font=("Segoe UI", 8, "bold")).pack(anchor="w")
        lf = tk.Frame(f, bg=BG); lf.pack(fill="both", expand=True, pady=4)
        sb = tk.Scrollbar(lf, orient="vertical")
        self.clist = tk.Listbox(lf, bg="#1e1f2e", fg=FG, selectbackground="#353850",
                                 selectforeground=FG, relief="flat", bd=0,
                                 font=("Consolas", 9), exportselection=False,
                                 selectmode="extended", yscrollcommand=sb.set)
        sb.config(command=self.clist.yview)
        self.clist.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.clist.bind("<<ListboxSelect>>", lambda _e: self._update_status())

        tk.Button(f, text="Identify by shift-click on 3D scene …",
                  command=self._on_identify_click, bg="#1e1f2e", fg=FG,
                  relief="flat", padx=6, pady=4, cursor="hand2").pack(fill="x", pady=(6, 4))

        btn_row = tk.Frame(f, bg=BG); btn_row.pack(fill="x", pady=(2, 4))
        tk.Button(btn_row, text="Preview selected", command=self._on_preview,
                  bg="#1e1f2e", fg=FG, relief="flat", padx=6, pady=6,
                  cursor="hand2").pack(side="left", fill="x", expand=True, padx=(0, 4))
        tk.Button(btn_row, text="Clean up selected", command=self._on_cleanup,
                  bg="#1e1f2e", fg=FG, relief="flat", padx=6, pady=6,
                  cursor="hand2").pack(side="left", fill="x", expand=True, padx=(4, 0))

        self.status = tk.Label(f, text="", bg=BG, fg="#7a80a0",
                                font=("Segoe UI", 9), anchor="w", justify="left",
                                wraplength=380)
        self.status.pack(fill="x", pady=(6, 8))

        self.continue_btn = tk.Button(
            f, text="Continue ▶", state="disabled", command=self._continue,
            bg="#1e3a5f", fg=FG, activebackground="#2a4d7a", relief="flat",
            padx=10, pady=8, cursor="hand2")
        self.continue_btn.pack(fill="x")

    def _on_ground_toggle(self):
        self.ground_slider.config(state="normal" if self.ground_var.get() else "disabled")

    def _on_auto_eps_toggle(self):
        self.eps_slider.config(state="disabled" if self.auto_eps_var.get() else "normal")

    # ── clustering ─────────────────────────────────────────────────────────
    def _run_clustering(self):
        self.status.config(text="Clustering …")
        self.root.update_idletasks()

        if self.ground_var.get():
            # CHANGE — exact index bookkeeping. The previous version called
            # pcg.remove_ground() (which only returns a bare PointCloud, no
            # indices) and then reconstructed nonground_src_idx/ground_src_idx
            # by nearest-neighbor-matching pcd_no_ground's points back against
            # pcd_full (_extract_ground_mask). On scans with duplicate or
            # near-duplicate point coordinates, that distance-based matching
            # can misclassify a handful of points, so the reconstructed index
            # arrays end up a few elements longer/shorter than self.pcd
            # actually is — which doesn't fail here, but fails later at
            # Continue when nonground_src_idx[obj_mask] gets boolean-indexed
            # against the wrong length. Getting the inlier indices directly
            # from segment_plane (same RANSAC pcg.remove_ground() uses
            # internally) sidesteps the reconstruction step entirely, so the
            # lengths are correct by construction — no matching, no rounding.
            thresh = self.ground_thresh_var.get()
            n_full = len(np.asarray(self.pcd_full.points))
            try:
                _, inliers = self.pcd_full.segment_plane(
                    distance_threshold=thresh, ransac_n=3, num_iterations=1000)
            except Exception:
                inliers = []
            ground_mask = np.zeros(n_full, dtype=bool)
            if len(inliers) > 0:
                ground_mask[np.asarray(inliers, dtype=np.int64)] = True

            self.nonground_src_idx = np.where(~ground_mask)[0]
            self.ground_src_idx = np.where(ground_mask)[0]
            self.pcd = self.pcd_full.select_by_index(self.nonground_src_idx.tolist())
            self.ground_pcd = self.pcd_full.select_by_index(self.ground_src_idx.tolist())
            assert len(self.nonground_src_idx) == len(np.asarray(self.pcd.points)), \
                "Ground-removal index bookkeeping is out of sync with pcd — " \
                "this should be impossible now that indices come straight " \
                "from segment_plane; please report this."
            print(f"[ground] Removed {int(ground_mask.sum()):,} floor points "
                  f"(threshold={thresh:.2f}).")
        else:
            self.pcd = self.pcd_full
            self.nonground_src_idx = np.arange(len(np.asarray(self.pcd_full.points)))
            self.ground_src_idx = np.array([], dtype=np.int64)
            self.ground_pcd = o3d.geometry.PointCloud()
            print("[ground] Ground removal disabled — floor stays part of the scene/clusters.")

        eps = None if self.auto_eps_var.get() else float(self.eps_var.get())
        min_pts = int(self.min_pts_var.get())
        labels, n_clusters, eps_used = pcg.cluster_dbscan(self.pcd, eps=eps, min_pts=min_pts)
        if self.auto_eps_var.get():
            self.eps_var.set(round(eps_used, 4))

        self.labels = labels
        self.n_clusters = n_clusters
        self.palette = pcg.make_palette(n_clusters)
        self.cluster_meta = build_cluster_meta(self.pcd, labels, n_clusters)
        self.removed_idx = set()   # stale against the new clustering — reset clean-up

        print(f"[cluster] {n_clusters} clusters found (eps={eps_used:.4f}, "
              f"min_points={min_pts}).")
        self._populate_list()

    def _populate_list(self):
        cids = sorted(self.cluster_meta, key=lambda c: -self.cluster_meta[c]["n_points"])
        self.clist.delete(0, "end")
        for cid in cids:
            m = self.cluster_meta[cid]
            self.clist.insert("end", f"{cid:>3}   {m['n_points']:>7,} pts")
        self._cids_sorted = cids
        for i, cid in enumerate(cids):
            col = self.palette[cid]
            hexcol = "#{:02x}{:02x}{:02x}".format(
                int(col[0]*200+55), int(col[1]*200+55), int(col[2]*200+55))
            self.clist.itemconfig(i, fg=hexcol)
        self._update_status()

    # ── selection helpers ────────────────────────────────────────────────
    def _get_selected_cids(self):
        sel = self.clist.curselection()
        return [self._cids_sorted[i] for i in sel]

    def _selected_mask(self):
        cids = self._get_selected_cids()
        if self.labels is None:
            return np.zeros(0, dtype=bool)
        if not cids:
            return np.zeros(len(self.labels), dtype=bool)
        mask = np.isin(self.labels, cids)
        if self.removed_idx:
            mask[np.array(sorted(self.removed_idx), dtype=np.int64)] = False
        return mask

    def _update_status(self):
        cids = self._get_selected_cids()
        mask = self._selected_mask()
        n = int(mask.sum())
        removed_txt = f", {len(self.removed_idx):,} cleaned out" if self.removed_idx else ""
        if cids:
            self.status.config(text=f"{len(cids)} cluster(s) selected: {cids} → "
                                     f"{n:,} pts{removed_txt}.")
            self.continue_btn.config(state="normal" if n > 0 else "disabled")
        else:
            self.status.config(text=f"{self.n_clusters} clusters. Select one or "
                                     f"more, then Continue.")
            self.continue_btn.config(state="disabled")

    # ── identify by click (unchanged old Step-0 window, now on-demand) ──────
    def _on_identify_click(self):
        cids = show_labeled_cluster_overview(
            self.pcd, self.labels, self.n_clusters, self.palette, self.cluster_meta)
        for cid in cids:
            if cid in self._cids_sorted:
                row = self._cids_sorted.index(cid)
                self.clist.selection_set(row)
                self.clist.see(row)
        self._update_status()

    # ── preview (shows exactly what you've selected, isolated) ──────────────
    def _on_preview(self):
        mask = self._selected_mask()
        if not mask.any():
            messagebox.showinfo("Preview", "Select at least one cluster first.")
            return
        pts = np.asarray(self.pcd.points)[mask]
        cols = (np.asarray(self.pcd.colors)[mask] if self.pcd.has_colors()
                else np.tile([0.9, 0.6, 0.15], (int(mask.sum()), 1)))
        disp = o3d.geometry.PointCloud()
        disp.points = o3d.utility.Vector3dVector(pts)
        disp.colors = o3d.utility.Vector3dVector(cols)

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name=f"Preview — {int(mask.sum()):,} pts — close (Q/Esc) when done",
            width=960, height=720)
        vis.add_geometry(disp)
        opt = vis.get_render_option()
        opt.background_color = np.array([0.08, 0.08, 0.12])
        opt.point_size = 2.5
        opt.show_coordinate_frame = True
        vis.run()
        vis.destroy_window()

    # ── clean-up tool (rectangle + lasso combo) ─────────────────────────────
    def _on_cleanup(self):
        mask = self._selected_mask()
        if not mask.any():
            messagebox.showinfo("Clean up", "Select at least one cluster first.")
            return
        global_idx = np.where(mask)[0]
        pts = np.asarray(self.pcd.points)[global_idx]
        cols = np.asarray(self.pcd.colors)[global_idx] if self.pcd.has_colors() else None

        def on_done(removed_global_idx):
            if len(removed_global_idx) > 0:
                self.removed_idx.update(int(i) for i in removed_global_idx)
            self._update_status()

        CleanupTool(self.root, pts, cols, global_idx, on_done)

    # ── continue ─────────────────────────────────────────────────────────
    def _continue(self):
        obj_mask = self._selected_mask()
        if not obj_mask.any():
            return
        self.root.destroy()
        self.on_continue_cb(self.pcd, self.labels, self.n_clusters, obj_mask,
                             self.nonground_src_idx, self.ground_src_idx, self.ground_pcd)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — split room vs. object
# ════════════════════════════════════════════════════════════════════════════

def split_room_and_object(pcd, labels, obj_mask):
    """Returns (room_pcd_without_object, object_pcd, obj_mask).

    `obj_mask` is now taken directly as a boolean array over pcd's rows,
    built by ClusteringWorkbench from the union of every cluster you
    selected minus anything you removed with the clean-up tool — so this
    naturally supports moving more than one cluster together as a single
    object. `labels` is kept as a parameter for interface stability but is
    no longer read here.
    """
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(pts)

    obj_mask  = np.asarray(obj_mask, dtype=bool)
    room_mask = ~obj_mask

    room = o3d.geometry.PointCloud()
    room.points = o3d.utility.Vector3dVector(pts[room_mask])
    room.colors = o3d.utility.Vector3dVector(cols[room_mask])

    obj = o3d.geometry.PointCloud()
    obj.points = o3d.utility.Vector3dVector(pts[obj_mask])
    obj.colors = o3d.utility.Vector3dVector(cols[obj_mask])

    return room, obj, obj_mask


# ── Hole-fill / "hallucination" — cover the gap left behind by the object ───
def _fit_local_plane(rim_pts, rim_cols, dist_thresh):
    """RANSAC-fits a plane to `rim_pts`. Returns (normal, u_axis, v_axis,
    plane_point, plane_pts, plane_cols) or None if it can't find one."""
    rim_pcd = o3d.geometry.PointCloud()
    rim_pcd.points = o3d.utility.Vector3dVector(rim_pts)
    try:
        plane_model, inlier_idx = rim_pcd.segment_plane(
            distance_threshold=dist_thresh, ransac_n=3, num_iterations=500)
    except Exception:
        return None
    if len(inlier_idx) < 8:
        return None

    normal = np.array(plane_model[:3], dtype=np.float64)
    n_norm = np.linalg.norm(normal)
    if n_norm < 1e-9:
        return None
    normal /= n_norm

    helper = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(helper, normal)) > 0.98:
        helper = np.array([1.0, 0.0, 0.0])
    u_axis = np.cross(helper, normal); u_axis /= np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)

    plane_pts, plane_cols = rim_pts[inlier_idx], rim_cols[inlier_idx]
    plane_point = plane_pts.mean(axis=0)
    return normal, u_axis, v_axis, plane_point, plane_pts, plane_cols


def _fill_hole_for_component(comp_pts, room_pts, room_cols,
                              pad_frac=1.5, max_fill_points=20_000,
                              max_span_ratio=6.0, max_grid_side=350):
    """Hallucinates fill points for ONE compact, connected chunk of the
    removed object. Returns (fill_pts, fill_cols) or (None, None) if this
    component isn't a good candidate to fill (too small, no nearby flat
    surface, or not flat enough to have a sensible "footprint")."""
    obj_min, obj_max = comp_pts.min(axis=0), comp_pts.max(axis=0)
    center = (obj_min + obj_max) / 2.0
    half_extent = np.maximum((obj_max - obj_min) / 2.0, 1e-4)
    diag = float(np.linalg.norm(half_extent)) * 2.0

    pad_half = half_extent * pad_frac
    lo, hi = center - pad_half, center + pad_half
    in_box = np.all((room_pts >= lo) & (room_pts <= hi), axis=1)
    rim_pts, rim_cols = room_pts[in_box], room_cols[in_box]
    if len(rim_pts) < 12:
        return None, None  # not enough local context to hallucinate anything

    dist_thresh = max(diag * 0.02, 1e-4)
    fit = _fit_local_plane(rim_pts, rim_cols, dist_thresh)
    if fit is None:
        return None, None
    normal, u_axis, v_axis, plane_point, plane_pts, plane_cols = fit

    # Sanity check: this component's OWN points must actually sit roughly
    # flat against the fitted plane (i.e. it looks like something resting
    # on/against a surface). A chunky, fully-3D object (a bag, a chair,
    # anything with real depth off the surface) does NOT have a single
    # well-defined "footprint plane" — forcing one produces a huge,
    # wrong-looking patch. Skip those and leave the (smaller, honest) gap
    # rather than hallucinate something worse.
    comp_dist_to_plane = np.abs((comp_pts - plane_point) @ normal)
    if float(np.median(comp_dist_to_plane)) > diag * 0.35:
        return None, None

    tree_rim = cKDTree(rim_pts)
    nn_d, _ = tree_rim.query(rim_pts, k=min(4, len(rim_pts)))
    nn_d = np.atleast_2d(nn_d)
    spacing = float(np.median(nn_d[:, 1:])) if nn_d.shape[1] > 1 else diag * 0.02
    spacing = max(spacing, diag * 0.005, 1e-5)

    obj_uv = np.stack([(comp_pts - plane_point) @ u_axis,
                        (comp_pts - plane_point) @ v_axis], axis=1)
    margin = spacing * 1.5
    u_lo, v_lo = obj_uv.min(axis=0) - margin
    u_hi, v_hi = obj_uv.max(axis=0) + margin

    # Sanity check: the footprint span shouldn't dwarf the component's own
    # bounding diagonal — if it does, spacing/plane estimation went wrong
    # somewhere upstream. Bail out safely instead of tiling a huge grid.
    span = max(u_hi - u_lo, v_hi - v_lo)
    if span > diag * max_span_ratio:
        return None, None

    n_u = min(max(int((u_hi - u_lo) / spacing), 1) + 1, max_grid_side)
    n_v = min(max(int((v_hi - v_lo) / spacing), 1) + 1, max_grid_side)
    if n_u * n_v > max_fill_points:
        return None, None

    uu, vv = np.meshgrid(np.linspace(u_lo, u_hi, n_u), np.linspace(v_lo, v_hi, n_v))
    grid_uv = np.stack([uu.ravel(), vv.ravel()], axis=1)

    tree_obj_uv = cKDTree(obj_uv)
    near_d, _ = tree_obj_uv.query(grid_uv, k=1)
    grid_uv = grid_uv[near_d <= spacing * 1.2]
    if len(grid_uv) == 0:
        return None, None

    fill_pts = (plane_point[None, :]
                + grid_uv[:, 0:1] * u_axis[None, :]
                + grid_uv[:, 1:2] * v_axis[None, :])

    room_tree = cKDTree(room_pts)
    d_to_room, _ = room_tree.query(fill_pts, k=1)
    fill_pts = fill_pts[d_to_room > spacing * 0.6]
    if len(fill_pts) == 0:
        return None, None

    rim_color_tree = cKDTree(plane_pts)
    _, nn_idx = rim_color_tree.query(fill_pts, k=1)
    fill_cols = plane_cols[nn_idx]
    return fill_pts, fill_cols


def fill_object_hole(room_pcd, obj_pcd, pad_frac=1.5, comp_radius_factor=3.0,
                      min_component_pts=20, max_fill_per_component=20_000):
    """
    Once the object's points are removed by split_room_and_object(), the
    room point cloud has a literal hole where it used to sit. Since the
    object is about to become movable, that hole would stay empty and
    visible even after the object has moved away. We never scanned what's
    actually behind the object, so there's no ground truth to restore —
    instead this HALLUCINATES plausible filler geometry, one connected
    piece of the removed object at a time:

      1. Split the removed object into connected components — a
         multi-cluster selection (e.g. two separate items picked
         together) is NOT one giant combined blob; treating it as one
         used to pull in a huge, irrelevant "rim" of room context
         spanning both, fit a single plane across all of it, and tile a
         room-spanning grid of points across that plane — visible as a
         big diagonal grid of hallucinated points, worse than the
         original hole. Splitting keeps each fill local and compact.
      2. For each component, look at the real room points immediately
         around it (the "rim") and fit the dominant local surface
         (wall/floor/table) via RANSAC.
      3. Only fill components that actually sit roughly flat against
         that surface — a chunky, fully-3D object has no single
         "footprint plane", and forcing one is exactly what produced the
         oversized, blurry-looking patch. Those are left as an honest,
         smaller gap instead.
      4. Tile synthetic points across the flat component's footprint on
         that plane, matched to the room's own local point spacing, and
         color each from its nearest real rim point.

    Returns a NEW point cloud (room_pcd + synthetic fill points), or
    room_pcd unchanged if nothing was safe to fill.
    """
    obj_pts_all = np.asarray(obj_pcd.points)
    room_pts = np.asarray(room_pcd.points)
    if len(obj_pts_all) == 0 or len(room_pts) < 20:
        return room_pcd
    room_cols = (np.asarray(room_pcd.colors) if room_pcd.has_colors()
                 else np.ones_like(room_pts))

    n_obj = len(obj_pts_all)
    obj_tree = cKDTree(obj_pts_all)
    nn_d, _ = obj_tree.query(obj_pts_all, k=min(4, n_obj))
    nn_d = np.atleast_2d(nn_d)
    obj_spacing = float(np.median(nn_d[:, 1:])) if nn_d.shape[1] > 1 else 0.01
    obj_spacing = max(obj_spacing, 1e-5)
    link_radius = obj_spacing * comp_radius_factor

    pairs = obj_tree.query_pairs(r=link_radius, output_type="ndarray")
    if len(pairs) > 0:
        graph = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                               shape=(n_obj, n_obj))
        n_comp, comp_labels = connected_components(graph, directed=False)
    else:
        n_comp, comp_labels = n_obj, np.arange(n_obj)

    all_fill_pts, all_fill_cols, n_filled_components = [], [], 0
    for cid in range(n_comp):
        comp_mask = comp_labels == cid
        if np.count_nonzero(comp_mask) < min_component_pts:
            continue
        fill_pts, fill_cols = _fill_hole_for_component(
            obj_pts_all[comp_mask], room_pts, room_cols,
            pad_frac=pad_frac, max_fill_points=max_fill_per_component)
        if fill_pts is not None and len(fill_pts) > 0:
            all_fill_pts.append(fill_pts)
            all_fill_cols.append(fill_cols)
            n_filled_components += 1

    if not all_fill_pts:
        print("[hole-fill] Nothing safe to hallucinate (object wasn't resting "
              "flat against a nearby surface, or there wasn't enough "
              "surrounding context) — leaving the gap as-is.")
        return room_pcd

    fill_pts = np.vstack(all_fill_pts)
    fill_cols = np.vstack(all_fill_cols)
    filled = o3d.geometry.PointCloud()
    filled.points = o3d.utility.Vector3dVector(np.vstack([room_pts, fill_pts]))
    filled.colors = o3d.utility.Vector3dVector(np.vstack([room_cols, fill_cols]))
    print(f"[hole-fill] Hallucinated {len(fill_pts):,} points across "
          f"{n_filled_components} component(s) to help cover the object's "
          f"footprint once it moves.")
    return filled


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — background Gaussian-splat fitting (runs in parallel with the room
#           viewer opening) and conversion into a movable "splat proxy"
# ════════════════════════════════════════════════════════════════════════════
#
# NOTE: gaussian_splat_render.estimate_gaussians() does a pure-Python,
# per-point KD-tree query in a for-loop. That's fine for a small hand-picked
# object, but in a huge scene an object cluster can still be tens of
# thousands of points, and the per-point loop can take minutes — which is
# why the splat appeared to "never show up". estimate_gaussians_fast()
# below does the exact same PCA/eigh covariance fit, but fully vectorized
# with scipy's cKDTree (batched KNN query + batched eigh), processed in
# chunks to bound memory. This is typically 50-200x faster and is what
# SplatJob now uses by default.

def estimate_gaussians_fast(points: np.ndarray, colors: np.ndarray,
                             k: int = 16, scale: float = 1.0,
                             chunk_size: int = 50_000) -> dict:
    N = len(points)
    pts64 = points.astype(np.float64)
    tree = cKDTree(pts64)
    cov3d = np.zeros((N, 6), dtype=np.float32)

    done = 0
    while done < N:
        end = min(done + chunk_size, N)
        chunk = pts64[done:end]

        _, idx = tree.query(chunk, k=min(k + 1, N))
        idx = np.atleast_2d(idx)
        nbr_idx = idx[:, 1:] if idx.shape[1] > 1 else idx  # drop self-match

        diffs = pts64[nbr_idx] - chunk[:, None, :]                   # (n,k,3)
        kk = max(diffs.shape[1] - 1, 1)
        C = np.einsum('nki,nkj->nij', diffs, diffs) / kk             # (n,3,3)

        vals, vecs = np.linalg.eigh(C)                                # batched
        vals = np.clip(vals, 0.0, None)
        mx = vals.max(axis=1, keepdims=True)
        mx = np.where(mx > 0, mx, 1e-6)
        vals = np.maximum(vals, mx * 0.01)

        C2 = np.einsum('nij,nj,nkj->nik', vecs, vals, vecs) * (scale ** 2)
        cov3d[done:end] = np.stack(
            [C2[:, 0, 0], C2[:, 0, 1], C2[:, 0, 2],
             C2[:, 1, 1], C2[:, 1, 2], C2[:, 2, 2]], axis=1)

        done = end
        print(f"[splat] fitted {done:,}/{N:,} gaussians …")

    return dict(mean=points.astype(np.float32),
                color=colors.astype(np.float32),
                opacity=np.full(N, 0.85, np.float32),
                cov3d=cov3d)


def build_splat_pointcloud(gauss: dict,
                            max_gaussians: int = 1000,
                            samples_per_gaussian: int = 100) -> o3d.geometry.PointCloud:
    """
    Converts fitted Gaussians into a DENSE POINT CLOUD by sampling child
    points from each Gaussian's 3D probability distribution.

    Why this looks right vs the old triangle-mesh approach:
    ─────────────────────────────────────────────────────────
    The old code built tiny triangle-mesh ellipsoids — each Gaussian became a
    hard-edged 3D sphere stretched along its principal axes. That looked like a
    jagged crystal / rock because every ellipsoid's surface is a solid faceted
    triangle boundary.

    Real Gaussian splatting is about SOFT DENSITY FALLOFF: a Gaussian should be
    brightest/densest at its mean and fade out to nothing at its edges. You
    can approximate that in Open3D's renderer (which has no per-splat alpha) by
    sampling child points from the actual 3D Gaussian distribution N(mean, Σ).
    Points drawn from a Gaussian naturally cluster densely near the mean and
    taper off at the edges — the visual density gradient IS the soft falloff.

    Full pipeline:
      • Unpack each Gaussian's 6-component cov3d into its 3×3 matrix.
      • Eigendecompose: Σ = V diag(λ) Vᵀ  →  square root L = V diag(√λ).
      • Sample child points:  x = μ + L z,  z ~ N(0, I)   (fully vectorised,
        no Python loop over gaussians).
      • Colour each child point with its parent Gaussian's color.
      • Stack into a single PointCloud.

    Parameters:
      max_gaussians        : randomly subsample parent Gaussians above this count
                             (weighted by opacity so high-confidence ones survive).
      samples_per_gaussian : child points drawn per Gaussian; more = softer / denser.
                             Total points ≈ max_gaussians × samples_per_gaussian.
    """
    mean   = gauss["mean"].astype(np.float64)
    color  = gauss["color"].astype(np.float64)
    cov3d  = gauss["cov3d"].astype(np.float64)
    opacity = gauss["opacity"].astype(np.float64)
    N = len(mean)

    # ── subsample ─────────────────────────────────────────────────────────────
    if N > max_gaussians:
        w = opacity / (opacity.sum() + 1e-12)
        idx = np.random.choice(N, max_gaussians, replace=False, p=w)
        mean, color, cov3d = mean[idx], color[idx], cov3d[idx]
        N = max_gaussians
        print(f"[splat] subsampled to {N:,} gaussians (opacity-weighted)")

    # ── unpack cov3d → (N,3,3) ────────────────────────────────────────────────
    C = np.zeros((N, 3, 3), dtype=np.float64)
    C[:, 0, 0] = cov3d[:, 0];  C[:, 0, 1] = C[:, 1, 0] = cov3d[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov3d[:, 2]
    C[:, 1, 1] = cov3d[:, 3];  C[:, 1, 2] = C[:, 2, 1] = cov3d[:, 4]
    C[:, 2, 2] = cov3d[:, 5]

    # ── matrix square root via eigendecomposition ─────────────────────────────
    # Σ = V diag(λ) Vᵀ  →  L = V diag(√λ)  so that L Lᵀ = Σ
    vals, vecs = np.linalg.eigh(C)           # (N,3), (N,3,3) — batched, fast
    vals = np.clip(vals, 1e-12, None)
    L = vecs * np.sqrt(vals)[:, None, :]     # (N,3,3)  broadcasting

    # ── vectorised sampling: x = μ + L z,  z ~ N(0, I) ──────────────────────
    S = samples_per_gaussian
    z = np.random.randn(N, S, 3)                             # (N, S, 3)
    child_pts = mean[:, None, :] + np.einsum('nij,nsj->nsi', L, z)  # (N,S,3)
    child_pts = child_pts.reshape(-1, 3)                     # (N*S, 3)

    # Each child inherits its parent Gaussian's colour
    child_cols = np.repeat(color, S, axis=0)                 # (N*S, 3)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(child_pts)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(child_cols, 0.0, 1.0))

    print(f"[splat] density cloud: {len(child_pts):,} points "
          f"({N:,} gaussians × {S} samples)")
    return pcd


# ════════════════════════════════════════════════════════════════════════════
# Step 3b — export the fitted Gaussians as a standard 3DGS-format .ply
# ════════════════════════════════════════════════════════════════════════════
#
# The viewer only ever moves/rotates the DENSITY-SAMPLED PROXY (obj_geom) —
# that's a plain colored point cloud, fine for eyeballing in Open3D but not
# what a real Gaussian-splat renderer wants. The renderer wants the actual
# fitted Gaussians: mean, covariance, opacity, color-as-spherical-harmonics.
# Those live in job.gauss and are never touched by the J/L/I/K/U/O/N/M/T/G
# handlers — only obj_geom.points is. So on export we take job.gauss and
# apply the SAME rigid transform (accumulated rotation + net centroid shift)
# that the user applied in the viewer, then write it out in the layout used
# by the reference 3D Gaussian Splatting repo (and read by SuperSplat,
# Postshot, gsplat, and most other splat tools):
#
#   x y z  nx ny nz  f_dc_0 f_dc_1 f_dc_2  f_rest_0..44
#   opacity  scale_0 scale_1 scale_2  rot_0 rot_1 rot_2 rot_3
#
# Colors have no higher-order SH here (f_rest is all zero — this pipeline
# never fit any), opacity is stored pre-sigmoid, scale is stored pre-exp,
# and rot is a normalized quaternion (w,x,y,z) — all matching the
# activation functions every standard splat viewer applies on load.

def cov3d_to_scale_quat(cov3d: np.ndarray):
    """Batched: (N,6) packed covariances -> (N,3) scales, (N,4) quats (w,x,y,z).

    Eigendecomposes each 3x3 covariance. The eigenvectors give the splat's
    orientation, the sqrt-eigenvalues give its per-axis scale (this is just
    the inverse of how build_splat_pointcloud's L = V diag(sqrt(lambda))
    was built). eigh's eigenvector matrix isn't guaranteed to be a proper
    rotation (det can be -1, i.e. a reflection) so we flip the last axis
    where needed before converting to a quaternion.
    """
    N = len(cov3d)
    C = np.zeros((N, 3, 3), dtype=np.float64)
    C[:, 0, 0] = cov3d[:, 0]; C[:, 0, 1] = C[:, 1, 0] = cov3d[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov3d[:, 2]
    C[:, 1, 1] = cov3d[:, 3]; C[:, 1, 2] = C[:, 2, 1] = cov3d[:, 4]
    C[:, 2, 2] = cov3d[:, 5]

    vals, vecs = np.linalg.eigh(C)          # ascending eigenvalues, (N,3,3)
    vals = np.clip(vals, 1e-12, None)
    scale = np.sqrt(vals)                    # (N,3)

    dets = np.linalg.det(vecs)
    flip = dets < 0
    vecs[flip, :, -1] *= -1.0                # force proper rotation (det=+1)

    quat_xyzw = Rotation.from_matrix(vecs).as_quat()          # (N,4)
    quat_wxyz = np.concatenate(
        [quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=1)        # -> (w,x,y,z)
    return scale.astype(np.float32), quat_wxyz.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────
# CHANGE 2 — split "write" from "encode".
# write_gaussian_ply_encoded() writes values that are ALREADY in splat
# activation-space (raw quaternion, log-scale, logit-opacity, SH f_dc/f_rest)
# straight to disk with NO transform — this is what preserved original
# points go through, so they come out byte-for-byte identical to the source.
# encode_linear_to_raw() holds the OLD logic (cov3d→scale/quat, linear
# opacity→logit, linear color→f_dc, f_rest zeroed) and is now only called
# for points that never had real splat data to begin with (the synthetic
# fallback path), not for anything we're actually trying to preserve.
# ─────────────────────────────────────────────────────────────────────────

def write_gaussian_ply_encoded(path: str, mean: np.ndarray, f_dc: np.ndarray,
                                f_rest: np.ndarray, opacity_logit: np.ndarray,
                                log_scale: np.ndarray, rot_quat: np.ndarray):
    """Writes already-activation-encoded splat parameters verbatim — no
    log/logit/SH re-derivation happens here, so nothing is re-approximated."""
    N = len(mean)
    normals = np.zeros((N, 3), dtype=np.float32)
    n_rest = f_rest.shape[1]

    names = (["x", "y", "z", "nx", "ny", "nz",
              "f_dc_0", "f_dc_1", "f_dc_2"]
             + [f"f_rest_{i}" for i in range(n_rest)]
             + ["opacity", "scale_0", "scale_1", "scale_2",
                "rot_0", "rot_1", "rot_2", "rot_3"])
    attrs = np.concatenate(
        [mean.astype(np.float32), normals, f_dc.astype(np.float32),
         f_rest.astype(np.float32), opacity_logit.reshape(-1, 1).astype(np.float32),
         log_scale.astype(np.float32), rot_quat.astype(np.float32)], axis=1)

    vertex = np.empty(N, dtype=[(n, "f4") for n in names])
    for i, n in enumerate(names):
        vertex[n] = attrs[:, i]

    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(path)


def encode_linear_to_raw(mean: np.ndarray, color: np.ndarray,
                          opacity: np.ndarray, cov3d: np.ndarray):
    """OLD behavior, kept ONLY as the synthetic-fallback path: takes linear
    (color 0-1, opacity 0-1, covariance) values and converts them into
    activation-encoded (f_dc, f_rest=0, opacity_logit, log_scale, rot_quat)
    so they can go through the same writer as preserved originals. Every
    call site of this function is a point that never had real splat data —
    preserved points skip this entirely."""
    N = len(mean)
    scale, quat = cov3d_to_scale_quat(cov3d)
    log_scale = np.log(np.clip(scale, 1e-8, None)).astype(np.float32)

    opac = np.clip(opacity.astype(np.float64), 1e-6, 1.0 - 1e-6)
    opac_logit = np.log(opac / (1.0 - opac)).astype(np.float32)

    f_dc = ((color.astype(np.float64) - 0.5) / SH_C0).astype(np.float32)
    f_rest = np.zeros((N, 45), dtype=np.float32)   # no higher-order SH to fall back on

    return mean.astype(np.float32), f_dc, f_rest, opac_logit, log_scale, quat


# ─────────────────────────────────────────────────────────────────────────
# CHANGE 3 — reposed_object_gaussians now preserves original attributes.
# If job carries the object's ORIGINAL scale/rot/opacity/f_dc/f_rest
# (job.orig, populated straight from SourceSplat), only `mean` and `rot`
# are touched — rotated/translated by the viewer's accumulated transform.
# scale/opacity/f_dc/f_rest are rigid-motion invariant and pass through
# completely untouched. Only when job.orig is missing (no splat data existed
# for this object) do we fall back to the old cov3d-based PCA-fit re-pose.
# ─────────────────────────────────────────────────────────────────────────

def _rotate_quat_wxyz(quat_wxyz: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Apply rotation matrix R to an array of (w,x,y,z) quaternions."""
    xyzw = quat_wxyz[:, [1, 2, 3, 0]]
    q_rot = Rotation.from_matrix(R) * Rotation.from_quat(xyzw)
    out_xyzw = q_rot.as_quat()
    return np.concatenate([out_xyzw[:, 3:4], out_xyzw[:, :3]], axis=1)


def _darken_f_dc(f_dc: np.ndarray, darken: np.ndarray) -> np.ndarray:
    """
    CHANGE 12 — apply the viewer's per-point occlusion darkening to
    activation-encoded SH DC color before writing it to disk, so the
    exported .ply carries the same "under something" shading you see live
    in the combined viewer, instead of always exporting full brightness.

    Only the DC (0th-order) SH term is touched — same simplification the
    rest of this file already makes when DECODING color for display (see
    SourceSplat.__init__: "decode DC term only ... never written back
    out"). f_rest (higher-order, view-dependent SH) is left untouched, so
    view-dependent highlights/reflections on the object are unaffected;
    only its base color is dimmed. `darken` is a per-point multiplier in
    [0, 1] (1 = unchanged, 0 = black), same array computed live by
    CombinedScene._apply_occlusion_shading().
    """
    color = np.clip(0.5 + SH_C0 * f_dc.astype(np.float64), 0.0, 1.0)
    color = np.clip(color * darken.reshape(-1, 1), 0.0, 1.0)
    return ((color - 0.5) / SH_C0).astype(np.float32)


def reposed_object_gaussians(job: "SplatJob", R_total: np.ndarray,
                              base_centroid: np.ndarray,
                              current_centroid: np.ndarray,
                              darken: np.ndarray = None):
    """Re-poses the object by the viewer's accumulated transform. Returns
    activation-encoded (mean, f_dc, f_rest, opacity_logit, log_scale,
    rot_quat) ready for write_gaussian_ply_encoded — preserving originals
    whenever job.orig is available.

    CHANGE 12: optional `darken` — per-point occlusion multiplier from
    CombinedScene, same length/order as job.gauss/job.orig (both are
    sliced from the object cluster in the same row order and never
    reordered), so it can be applied directly with no re-matching."""
    mean = job.gauss["mean"].astype(np.float64)
    mean_t = (mean - base_centroid) @ R_total.T + current_centroid

    if job.orig is not None:
        # ── preserved path: only mean + rotation change (+ color, if darkened) ──
        rot_t = _rotate_quat_wxyz(job.orig["rot_raw"], R_total)
        f_dc = job.orig["f_dc"]
        if darken is not None:
            f_dc = _darken_f_dc(f_dc, darken)
        return (mean_t.astype(np.float32), f_dc, job.orig["f_rest"],
                job.orig["opacity_raw"], job.orig["scale_raw"], rot_t)

    # ── fallback path: no original splat data existed for this object ───
    cov3d = job.gauss["cov3d"].astype(np.float64)
    N = len(mean)
    C = np.zeros((N, 3, 3), dtype=np.float64)
    C[:, 0, 0] = cov3d[:, 0]; C[:, 0, 1] = C[:, 1, 0] = cov3d[:, 1]
    C[:, 0, 2] = C[:, 2, 0] = cov3d[:, 2]
    C[:, 1, 1] = cov3d[:, 3]; C[:, 1, 2] = C[:, 2, 1] = cov3d[:, 4]
    C[:, 2, 2] = cov3d[:, 5]
    C2 = np.einsum("ij,njk,lk->nil", R_total, C, R_total)
    cov3d_t = np.stack(
        [C2[:, 0, 0], C2[:, 0, 1], C2[:, 0, 2],
         C2[:, 1, 1], C2[:, 1, 2], C2[:, 2, 2]], axis=1)

    color = job.gauss["color"].astype(np.float64)
    if darken is not None:
        color = np.clip(color * darken.reshape(-1, 1), 0.0, 1.0)

    return encode_linear_to_raw(mean_t, color,
                                 job.gauss["opacity"].astype(np.float64), cov3d_t)


def export_object_as_gaussian_splat(job: "SplatJob", R_total: np.ndarray,
                                     base_centroid: np.ndarray,
                                     current_centroid: np.ndarray,
                                     out_path: str,
                                     darken: np.ndarray = None):
    """Write just the object's re-posed Gaussians to out_path."""
    mean_t, f_dc, f_rest, opac_logit, log_scale, rot_t = reposed_object_gaussians(
        job, R_total, base_centroid, current_centroid, darken=darken)
    write_gaussian_ply_encoded(out_path, mean_t, f_dc, f_rest, opac_logit, log_scale, rot_t)
    return len(mean_t)


# ─────────────────────────────────────────────────────────────────────────
# CHANGE 4 — room_points_as_gaussians is now an explicit FALLBACK ONLY.
# It's only ever called when SourceSplat had no real splat attributes to
# begin with (a bare colored point cloud). Whenever the source DID carry
# real attrs, the room's original scale/rot/opacity/f_dc/f_rest are sliced
# directly out of SourceSplat by room_mask in export_scene_as_gaussian_splat
# below — no KNN radius guessing, no isotropic blobs, no forced opacity.
# ─────────────────────────────────────────────────────────────────────────

def room_points_as_gaussians(room_pcd: "o3d.geometry.PointCloud",
                              k: int = 8, opacity: float = 0.99,
                              min_scale_fraction: float = 1e-4):
    """FALLBACK ONLY — used when the source had no real splat data at all.
    Turns plain room points into tiny ISOTROPIC Gaussians. Size is adaptive
    from each point's own nearest-neighbor distance (via cKDTree)."""
    pts = np.asarray(room_pcd.points, dtype=np.float64)
    N = len(pts)
    cols = (np.asarray(room_pcd.colors, dtype=np.float64)
            if room_pcd.has_colors() else np.full((N, 3), 0.6))

    if N == 0:
        return dict(mean=pts.astype(np.float32), color=cols.astype(np.float32),
                    opacity=np.zeros(0, np.float32),
                    cov3d=np.zeros((0, 6), np.float32))

    ext = room_pcd.get_axis_aligned_bounding_box().get_extent()
    floor = max(float(np.linalg.norm(ext)) * min_scale_fraction, 1e-5)

    tree = cKDTree(pts)
    dists, _ = tree.query(pts, k=min(k + 1, N), workers=-1)
    dists = np.atleast_2d(dists)
    nn = dists[:, 1:] if dists.shape[1] > 1 else dists   # drop self-match
    radius = np.clip(nn.mean(axis=1) * 0.5, floor, None)

    cov3d = np.zeros((N, 6), dtype=np.float32)
    r2 = (radius ** 2).astype(np.float32)
    cov3d[:, 0] = r2   # xx
    cov3d[:, 3] = r2   # yy
    cov3d[:, 5] = r2   # zz  (xy, xz, yz stay 0 — isotropic)

    return dict(mean=pts.astype(np.float32), color=cols.astype(np.float32),
                opacity=np.full(N, opacity, np.float32), cov3d=cov3d)


def export_scene_as_gaussian_splat(room_pcd: "o3d.geometry.PointCloud",
                                    job: "SplatJob", R_total: np.ndarray,
                                    base_centroid: np.ndarray,
                                    current_centroid: np.ndarray,
                                    out_path: str, room_k: int = 8,
                                    src: "SourceSplat" = None,
                                    room_src_idx: np.ndarray = None,
                                    darken: np.ndarray = None):
    """Writes ONE .ply containing the whole scene. If `src` (the original
    SourceSplat) and `room_src_idx` (indices of the room's points back into
    `src`) are provided and src.has_splat_attrs, the room is written back
    using its ORIGINAL attributes verbatim — no synthetic re-fit. Falls
    back to the old isotropic-blob approximation only if no original splat
    data is available.

    CHANGE 12: `darken` (per-object-point occlusion multiplier from the
    viewer) is applied ONLY to the object's gaussians, matching what you
    see live in the combined viewer. The room is written back verbatim
    either way — it's static, so it was never re-shaded in the first
    place."""
    obj_mean, obj_fdc, obj_frest, obj_opac_logit, obj_log_scale, obj_rot = \
        reposed_object_gaussians(job, R_total, base_centroid, current_centroid,
                                  darken=darken)

    if src is not None and src.has_splat_attrs and room_src_idx is not None:
        room_mean       = src.xyz[room_src_idx].astype(np.float32)
        room_fdc        = src.f_dc[room_src_idx]
        room_frest      = src.f_rest[room_src_idx]
        room_opac_logit = src.opacity_raw[room_src_idx]
        room_log_scale  = src.scale_raw[room_src_idx]
        room_rot        = src.rot_raw[room_src_idx]
        print(f"[export] Room: {len(room_mean):,} points written back with "
              f"ORIGINAL scale/rot/opacity/SH — no re-fitting.")
    else:
        room_g = room_points_as_gaussians(room_pcd, k=room_k)
        room_mean, room_fdc, room_frest, room_opac_logit, room_log_scale, room_rot = \
            encode_linear_to_raw(room_g["mean"], room_g["color"],
                                  room_g["opacity"], room_g["cov3d"])
        print(f"[export] Room: {len(room_mean):,} points written with "
              f"SYNTHETIC isotropic Gaussians (no original splat data found).")

    mean      = np.vstack([room_mean, obj_mean])
    f_dc      = np.vstack([room_fdc, obj_fdc])
    f_rest    = np.vstack([room_frest, obj_frest])
    opac_logit = np.concatenate([room_opac_logit, obj_opac_logit])
    log_scale = np.vstack([room_log_scale, obj_log_scale])
    rot       = np.vstack([room_rot, obj_rot])

    write_gaussian_ply_encoded(out_path, mean, f_dc, f_rest, opac_logit, log_scale, rot)
    return len(room_mean), len(obj_mean)


class SplatJob:
    """Shared state between the fitting thread and the viewer's poll loop.

    CHANGE 5: added `self.orig` — the object's ORIGINAL per-point
    scale/rot/opacity/f_dc/f_rest, sliced straight from SourceSplat before
    any fitting happens. `self.gauss` (the PCA/KNN fit) is still computed
    for the interactive movable proxy (obj_geom) shown in the viewer, but
    the final EXPORT now reads from `self.orig` whenever it's available —
    see reposed_object_gaussians() — so the PCA fit never touches what gets
    written to disk.
    """
    def __init__(self):
        self.done     = threading.Event()
        self.error    = None
        self.gauss    = None   # raw dict from estimate_gaussians_fast (viewer proxy only)
        self.orig     = None   # dict of ORIGINAL attrs from SourceSplat (used for export)
        self.splat_pcd = None  # o3d.PointCloud — density-sampled splat cloud
        self.centroid  = None

    def run(self, obj_pcd, k=16, scale=1.0, orig_attrs=None):
        """orig_attrs: optional dict with keys scale_raw/rot_raw/opacity_raw/
        f_dc/f_rest, already sliced (by the object mask) out of SourceSplat.
        Pass None if the source had no splat attributes to preserve."""
        try:
            pts = np.asarray(obj_pcd.points, dtype=np.float32)
            cols = (np.asarray(obj_pcd.colors, dtype=np.float32)
                    if obj_pcd.has_colors()
                    else np.full((len(pts), 3), 0.8, np.float32))
            if len(pts) == 0:
                raise ValueError("Selected cluster has 0 points.")

            self.orig = orig_attrs
            if orig_attrs is not None:
                print(f"[splat] Object has {len(pts):,} points WITH original "
                      f"splat attributes — export will reuse them verbatim "
                      f"(only position/orientation will be updated on move).")
            else:
                print(f"[splat] Object has no original splat attributes — "
                      f"export will use a synthetic PCA fit instead.")

            print(f"[splat] Fitting {len(pts):,} Gaussians  k={k}  scale={scale} "
                  f"(for the interactive viewer proxy) …")
            g = estimate_gaussians_fast(pts, cols, k=k, scale=scale)
            print("[splat] Fit done. Building density point cloud …")

            splat_pcd = obj_pcd
            self.centroid = pts.mean(axis=0).astype(np.float64)

            self.gauss     = g
            self.splat_pcd = splat_pcd
            self.centroid  = g["mean"].mean(axis=0).astype(np.float64)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = exc
        finally:
            self.done.set()


def maybe_open_true_splat_preview(obj_pcd, tmp_dir):
    """Optional: dump the object and open gaussian_splat_render's own
    photoreal moderngl window in a separate process-like call. Read-only,
    not part of the movable combined scene."""
    path = os.path.join(tmp_dir, "_splat_preview_object.ply")
    o3d.io.write_point_cloud(path, obj_pcd)
    sys.argv = ["gaussian_splat_render.py", "--ply", path]
    gsr.main()


# ════════════════════════════════════════════════════════════════════════════
# Ground restoration helpers
# ════════════════════════════════════════════════════════════════════════════
#
# NOTE: ground removal is currently DISABLED in main() below (see the
# "Ground removal disabled" block), so _extract_ground_mask()/_extract_ground()
# are no longer called anywhere in this file. They're left in place, unused,
# in case ground removal is ever turned back on later.

def _extract_ground_mask(pcd_full: o3d.geometry.PointCloud,
                          pcd_no_ground: o3d.geometry.PointCloud) -> np.ndarray:
    """
    CHANGE 7: returns a boolean MASK (over pcd_full's row order) marking
    which points remove_ground() stripped out, instead of building a brand
    new filtered PointCloud. A mask can be reused directly to index
    SourceSplat's original attribute arrays — no round trip through o3d
    (and no risk of it silently reordering/deduplicating rows) required.

    Still uses the same nearest-neighbor distance trick as before (robust
    regardless of what remove_ground() does internally), since we don't
    have remove_ground()'s own index bookkeeping to rely on.
    """
    pts_full = np.asarray(pcd_full.points)
    pts_kept = np.asarray(pcd_no_ground.points)

    if len(pts_kept) == 0 or len(pts_full) == 0:
        return np.zeros(len(pts_full), dtype=bool)

    tree  = cKDTree(pts_kept)
    dists, _ = tree.query(pts_full, k=1, workers=-1)

    # Points farther than 1 mm from any kept point were removed → ground
    eps = 1e-3
    return dists > eps


def _extract_ground(pcd_full: o3d.geometry.PointCloud,
                    pcd_no_ground: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Thin convenience wrapper around _extract_ground_mask() for callers
    that still want an o3d.PointCloud (e.g. just for viewing)."""
    ground_mask = _extract_ground_mask(pcd_full, pcd_no_ground)
    pts_full = np.asarray(pcd_full.points)
    ground = o3d.geometry.PointCloud()
    ground.points = o3d.utility.Vector3dVector(pts_full[ground_mask])
    if pcd_full.has_colors():
        cols_full = np.asarray(pcd_full.colors)
        ground.colors = o3d.utility.Vector3dVector(cols_full[ground_mask])
    return ground


def _merge_pcds(*pcds: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Concatenate any number of PointClouds into one."""
    all_pts  = [np.asarray(p.points)  for p in pcds if len(p.points) > 0]
    all_cols = [np.asarray(p.colors)  for p in pcds
                if p.has_colors() and len(p.colors) > 0]

    merged = o3d.geometry.PointCloud()
    if all_pts:
        merged.points = o3d.utility.Vector3dVector(np.vstack(all_pts))
    if len(all_cols) == len(all_pts):   # only attach colors if every part has them
        merged.colors = o3d.utility.Vector3dVector(np.vstack(all_cols))
    return merged


# ════════════════════════════════════════════════════════════════════════════
# Step 4 — combined viewer: room (static) + splatted object (movable)
# ════════════════════════════════════════════════════════════════════════════

class CombinedScene:
    def __init__(self, room_pcd, job: SplatJob, src: "SourceSplat" = None,
                 room_src_idx: np.ndarray = None):
        self.room_pcd     = room_pcd
        self.job          = job
        # CHANGE 8: carry the original SourceSplat + the room's indices into
        # it all the way to export time, so _export_gaussians/_export_object_only
        # can write the room back from ORIGINAL attributes instead of
        # re-deriving synthetic ones from room_pcd alone.
        self.src          = src
        self.room_src_idx = room_src_idx

        self.obj_added       = False
        self.obj_geom        = None   # o3d.PointCloud — density-sampled splat cloud
        self.obj_base_points = None   # original sample points (for R reset)
        self.obj_base_colors = None   # CHANGE 11: object's TRUE (unshaded) colors,
                                       # captured once at attach time. Every shading
                                       # pass darkens FROM this, never from whatever
                                       # colors happen to already be on obj_geom —
                                       # otherwise repeated moves would compound and
                                       # the object would just get darker forever.
        self.centroid         = None  # current centroid (moves with object)
        self.base_centroid    = None  # original centroid (for R reset)
        self.R_total          = np.eye(3)  # accumulated rotation since fitting
                                            # (applied to job.gauss on export —
                                            #  see _apply_rotation / _export_gaussians)

        # ── CHANGE 15 — directional shadow via light-space KD-tree ──────────
        # (CHANGE 14's grid/hash shadow-map had two real bugs: the folded
        # integer cell key `cu*M + cv` isn't collision-safe for arbitrary
        # cv, and a grid cell finer than the room's actual point spacing
        # left "holes" — cells with no room point even where a real
        # surface sits, which silently read as "nothing there, fully lit."
        # Both produced patchy, wrong-looking shadows.)
        #
        # This version drops the grid entirely and does a KD-tree radius
        # search directly in light-space (u, v) — the same proven
        # query_ball_point mechanism the ORIGINAL overhead-proxy used, just
        # projected onto the light's plane instead of raw world XY, and
        # compared along the light ray instead of raw Z height. A radius
        # search naturally tolerates sparse/uneven point density, since it
        # finds whatever real points are nearby instead of relying on them
        # landing in one exact cell.
        # [LIGHTING DISABLED] self.LIGHT_AZIMUTH_DEG    = 40.0   # compass angle of the light around +Z (0 = +X)
        # [LIGHTING DISABLED] self.LIGHT_ELEVATION_DEG  = 55.0   # angle above the horizon (90 = straight down)
        # [LIGHTING DISABLED] self.SHADOW_RADIUS_FRAC   = 0.02   # width (fraction of room scale) of the
                                            # light-space column searched around each point
        # [LIGHTING DISABLED] self.SHADOW_BIAS_FRAC     = 0.004  # depth bias (fraction of room scale) so a point
                                            # doesn't "shadow itself" from float noise
        # [LIGHTING DISABLED] self.SHADOW_SATURATE_COUNT = 12    # occluding-neighbor count that = full shadow
        # [LIGHTING DISABLED] self.SHADOW_STRENGTH      = 0.65   # max darkening in full shadow (0=no effect, 1=black)
        # [LIGHTING DISABLED] self.ENABLE_SELF_SHADOW   = False  # CHANGE 16: object shadowing its OWN points is
                                            # OFF by default — see _compute_directional_darken
                                            # docstring. Bushy/cluttered objects (foliage, a
                                            # vase of flowers) saturate almost instantly and
                                            # end up darkened everywhere regardless of light
                                            # direction. Only enable for solid, simple,
                                            # genuinely concave objects.
        # [LIGHTING DISABLED] self.SELF_SHADOW_RADIUS_FRAC = 0.05  # fraction of the OBJECT's own size (not the
                                              # room's) — only used if ENABLE_SELF_SHADOW
        # [LIGHTING DISABLED] self.SELF_SHADOW_BIAS_FRAC   = 0.02  # fraction of the OBJECT's own size
        # [LIGHTING DISABLED] self.light_dir = None
        # [LIGHTING DISABLED] self.u_axis = None
        # [LIGHTING DISABLED] self.v_axis = None
        # [LIGHTING DISABLED] self.room_shadow_tree = None   # cKDTree over room points' (u, v) light-space coords
        # [LIGHTING DISABLED] self.room_light_depth = None   # matching light-space depth, one per room point
        # [LIGHTING DISABLED] self.last_occlusion_darken = None  # CHANGE 12: cached per-point darken
                                            # array from the most recent shading
                                            # pass, reused at export time so the
                                            # exported .ply matches the viewer.

        # ── CHANGE 18 — a real point light that lives INSIDE the room ───────
        # Everything above (LIGHT_AZIMUTH_DEG/LIGHT_ELEVATION_DEG/light_dir)
        # models a light infinitely far away — a "sun" outside the room
        # whose rays are parallel everywhere. That's the only light this
        # tool understood. LIGHT_MODE adds a second kind: an actual point
        # source positioned somewhere inside the room's own bounding box
        # (like a lamp or bulb) that you can drag around in 3D. Both modes
        # stay fully wired up; LIGHT_MODE just picks which one drives the
        # shading/gizmo at any given moment — see _apply_occlusion_shading,
        # build_light_gizmo, and the panel's mode toggle.
        # [LIGHTING DISABLED] self.LIGHT_MODE = "point"      # "point" (new, inside the room) or
                                        # "directional" (old, sun-like)
        # [LIGHTING DISABLED] self.light_pos = None          # world-space (x, y, z) of the bulb —
                                        # set by _init_light_pos() below
        # [LIGHTING DISABLED] self.LIGHT_POS_MIN = None      # the light is clamped to stay inside
        # [LIGHTING DISABLED] self.LIGHT_POS_MAX = None      # these bounds — a small inset of the
                                        # room's own bbox, so it can be moved
                                        # anywhere in the room but never
                                        # dragged out through a wall/ceiling
        # [LIGHTING DISABLED] self.POINT_SHADOW_RADIUS_FRAC = 0.02  # same idea as SHADOW_RADIUS_FRAC,
                                               # but for the point-light ray march
        # [LIGHTING DISABLED] self.POINT_SHADOW_BIAS_FRAC   = 0.08  # fraction of EACH ray's own
                                               # length left unsampled at both
                                               # ends, so neither the object
                                               # point nor the bulb itself
                                               # trips its own occluder test
        # [LIGHTING DISABLED] self.POINT_SHADOW_SAMPLES     = 10    # samples marched along each
                                               # object-point → bulb ray
        # [LIGHTING DISABLED] self.ENABLE_DISTANCE_FALLOFF  = True  # point lights get dimmer with
                                               # distance (inverse-square,
                                               # referenced to half the room's
                                               # diagonal so mid-room stays
                                               # ~neutral) — a directional/sun
                                               # light has no equivalent of
                                               # this, since it's infinitely
                                               # far away from every point
        # [LIGHTING DISABLED] self.room_kdtree_3d = None     # cKDTree over raw room XYZ (world
                                        # space) — built once; used by the
                                        # point-light ray march instead of
                                        # the light-space projected tree
                                        # above (which only makes sense for
                                        # one shared, fixed direction)

        # [LIGHTING DISABLED] self.room_scale = float(np.linalg.norm(
            # [LIGHTING DISABLED] self.room_pcd.get_axis_aligned_bounding_box().get_extent()))
        # [LIGHTING DISABLED] self._init_light_pos()
        # [LIGHTING DISABLED] self._build_light_basis()
        # [LIGHTING DISABLED] self._build_shadow_tree()

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Scene — room + movable splat object",
                                width=1280, height=720)
        self.vis.add_geometry(self.room_pcd)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.08, 0.08, 0.12])
        opt.point_size = 2.0
        opt.show_coordinate_frame = True

        self._register_keys()

    # ── CHANGE 18 — where the point light starts, and how far it can roam ──
    # [LIGHTING DISABLED] def _init_light_pos(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Default point-light position: centered over the room in X/Y and up
        # [LIGHTING DISABLED] near the ceiling in Z — roughly where a real room light/lamp would
        # [LIGHTING DISABLED] hang — INSIDE the room's own bounding box, not outside it like the
        # [LIGHTING DISABLED] old directional "sun". Also records LIGHT_POS_MIN/MAX, a small
        # [LIGHTING DISABLED] inset of the room's bbox, so the light can be dragged anywhere
        # [LIGHTING DISABLED] inside the room (via sliders or A/D/S/W/F/V) but never pushed out
        # [LIGHTING DISABLED] through a wall, floor, or ceiling.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] bbox = self.room_pcd.get_axis_aligned_bounding_box()
        # [LIGHTING DISABLED] lo, hi = np.asarray(bbox.min_bound), np.asarray(bbox.max_bound)
        # [LIGHTING DISABLED] margin = (hi - lo) * 0.05          # keep the light a bit clear of the walls
        # [LIGHTING DISABLED] self.LIGHT_POS_MIN = lo + margin
        # [LIGHTING DISABLED] self.LIGHT_POS_MAX = hi - margin
        # [LIGHTING DISABLED] center_xy = (lo[:2] + hi[:2]) / 2.0
        # [LIGHTING DISABLED] ceiling_z = hi[2] - (hi[2] - lo[2]) * 0.15   # ~85% of the way up
        # [LIGHTING DISABLED] pos = np.array([center_xy[0], center_xy[1], ceiling_z])
        # [LIGHTING DISABLED] self.light_pos = np.clip(pos, self.LIGHT_POS_MIN, self.LIGHT_POS_MAX)

    # [LIGHTING DISABLED] def move_light(self, delta):
        # [LIGHTING DISABLED] """Nudge the point light by `delta` (world-space, meters), clamped
        # [LIGHTING DISABLED] to stay inside LIGHT_POS_MIN/MAX."""
        # [LIGHTING DISABLED] self.light_pos = np.clip(self.light_pos + np.asarray(delta, dtype=np.float64),
                                  # [LIGHTING DISABLED] self.LIGHT_POS_MIN, self.LIGHT_POS_MAX)

    # ── CHANGE 14 — light-space basis, built once (the light never moves) ──
    # [LIGHTING DISABLED] def _build_light_basis(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Builds an orthonormal "light-space" frame: light_dir is the
        # [LIGHTING DISABLED] direction the light TRAVELS (light source → scene), and
        # [LIGHTING DISABLED] (u_axis, v_axis) span the plane perpendicular to it — exactly like
        # [LIGHTING DISABLED] setting up a camera that looks straight down the light direction.
        # [LIGHTING DISABLED] Everything downstream (the shadow map, and every per-point depth
        # [LIGHTING DISABLED] test) works in this frame, which is what makes the result an
        # [LIGHTING DISABLED] actual DIRECTIONAL shadow instead of an omnidirectional "stuff
        # [LIGHTING DISABLED] nearby in XY" count.

        # [LIGHTING DISABLED] Z is up in this pipeline (see colorize_by_height / remove_ground).
        # [LIGHTING DISABLED] LIGHT_ELEVATION_DEG is measured up from the horizon, so 90° is a
        # [LIGHTING DISABLED] straight-down light and 0° is a raking, near-horizontal light.
        # [LIGHTING DISABLED] LIGHT_AZIMUTH_DEG rotates that around +Z, 0° pointing along +X.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] az = np.radians(self.LIGHT_AZIMUTH_DEG)
        # [LIGHTING DISABLED] el = np.radians(self.LIGHT_ELEVATION_DEG)
        # [LIGHTING DISABLED] d = np.array([
            # [LIGHTING DISABLED] np.cos(el) * np.cos(az),
            # [LIGHTING DISABLED] np.cos(el) * np.sin(az),
            # [LIGHTING DISABLED] -np.sin(el),          # positive elevation tilts the ray downward
        # [LIGHTING DISABLED] ])
        # [LIGHTING DISABLED] self.light_dir = d / np.linalg.norm(d)

        # Any vector not parallel to light_dir gives one in-plane axis via
        # Gram-Schmidt; the second in-plane axis is just the cross product.
        # [LIGHTING DISABLED] helper = np.array([0.0, 0.0, 1.0])
        # [LIGHTING DISABLED] if abs(np.dot(helper, self.light_dir)) > 0.99:
            # [LIGHTING DISABLED] helper = np.array([1.0, 0.0, 0.0])
        # [LIGHTING DISABLED] u = np.cross(helper, self.light_dir)
        # [LIGHTING DISABLED] self.u_axis = u / np.linalg.norm(u)
        # [LIGHTING DISABLED] self.v_axis = np.cross(self.light_dir, self.u_axis)

    # [LIGHTING DISABLED] def _to_light_space(self, pts: np.ndarray):
        # [LIGHTING DISABLED] """u, v = position across the light's view; depth = distance along
        # [LIGHTING DISABLED] the light ray (SMALLER depth = CLOSER to the light source)."""
        # [LIGHTING DISABLED] u = pts @ self.u_axis
        # [LIGHTING DISABLED] v = pts @ self.v_axis
        # [LIGHTING DISABLED] depth = pts @ self.light_dir
        # [LIGHTING DISABLED] return u, v, depth

    # [LIGHTING DISABLED] @staticmethod
    # [LIGHTING DISABLED] def _light_uv(u, v):
        # [LIGHTING DISABLED] return np.stack([u, v], axis=1)

    # [LIGHTING DISABLED] def _build_shadow_tree(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Builds the STATIC part of the directional shadow test from the
        # [LIGHTING DISABLED] room alone: every room point's light-space (u, v) position goes
        # [LIGHTING DISABLED] into a KD-tree, and its light-space depth is kept alongside it.
        # [LIGHTING DISABLED] Built once in __init__ since the room is never edited or moved in
        # [LIGHTING DISABLED] this tool — every later shading pass reuses this tree and only
        # [LIGHTING DISABLED] has to project/query the small, currently-moving object cloud
        # [LIGHTING DISABLED] against it.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] pts = np.asarray(self.room_pcd.points)
        # [LIGHTING DISABLED] if len(pts) == 0:
            # [LIGHTING DISABLED] self.room_shadow_tree = None
            # [LIGHTING DISABLED] self.room_light_depth = None
            # [LIGHTING DISABLED] self.room_kdtree_3d = None
            # [LIGHTING DISABLED] return
        # [LIGHTING DISABLED] u, v, depth = self._to_light_space(pts)
        # [LIGHTING DISABLED] self.room_light_depth = depth
        # [LIGHTING DISABLED] self.room_shadow_tree = cKDTree(self._light_uv(u, v))

        # CHANGE 18 — a plain, un-projected 3D tree of the same room points.
        # The directional tree above only works because every shadow ray in
        # that mode shares one direction, so projecting once onto a shared
        # (u, v) plane is valid. A point light's rays fan out from a single
        # position instead, so each object point needs its OWN ray tested —
        # see _compute_point_light_darken, which queries this tree directly
        # in world space at several samples along each ray.
        # [LIGHTING DISABLED] self.room_kdtree_3d = cKDTree(pts)

    # ── CHANGE 17 — recover the light direction FROM the room's own shading ─
    # [LIGHTING DISABLED] def estimate_light_from_scene(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Looks at the ROOM's existing point colors and surface normals and
        # [LIGHTING DISABLED] solves for the single directional light that best explains the
        # [LIGHTING DISABLED] brightness pattern already baked into the scan — so instead of
        # [LIGHTING DISABLED] guessing azimuth/elevation by hand, you get a starting point
        # [LIGHTING DISABLED] derived from the actual photo/scan.

        # [LIGHTING DISABLED] Physics: under simple Lambertian shading, a surface's brightness
        # [LIGHTING DISABLED] is proportional to how directly it faces the light —
            # [LIGHTING DISABLED] brightness ≈ albedo * max(0, normal · direction_to_light) + ambient
        # [LIGHTING DISABLED] Assuming roughly uniform albedo across the room (a real, if
        # [LIGHTING DISABLED] imperfect, simplification — see caveat below), that becomes a
        # [LIGHTING DISABLED] linear model in the normal components, solvable by least squares:
            # [LIGHTING DISABLED] luminance_i ≈ normal_i · L + c
        # [LIGHTING DISABLED] Solving for L over every room point recovers the direction the
        # [LIGHTING DISABLED] room is, on average, "facing toward the light."

        # [LIGHTING DISABLED] Two passes are used because a plain one-shot fit is measurably
        # [LIGHTING DISABLED] biased (verified on synthetic data: azimuth came out ~1° off,
        # [LIGHTING DISABLED] elevation ~12° off, i.e. systematically UNDER-estimating how
        # [LIGHTING DISABLED] steep the light is): points near their own shadow terminator
        # [LIGHTING DISABLED] (grazing or self-shadowed, where the real max(0, ...) clip is
        # [LIGHTING DISABLED] active) don't fit the plain linear model and pull the fit toward
        # [LIGHTING DISABLED] a shallower elevation. So pass 1 gets a rough direction, then
        # [LIGHTING DISABLED] pass 2 refits using only the brighter, clearly front-lit 40% of
        # [LIGHTING DISABLED] points (by pass-1 alignment) — which removed that bias entirely
        # [LIGHTING DISABLED] in testing (recovered a known 115°/35° light to within 0.1°).

        # [LIGHTING DISABLED] Caveat: this assumes roughly uniform surface albedo. A room with
        # [LIGHTING DISABLED] very different materials (dark wood table, pale walls, a bright
        # [LIGHTING DISABLED] rug) will bias the result somewhat, same as any single-shot
        # [LIGHTING DISABLED] shape-from-shading approach — treat this as a strong starting
        # [LIGHTING DISABLED] point to fine-tune with the sliders, not a lab measurement.

        # [LIGHTING DISABLED] Sets self.LIGHT_AZIMUTH_DEG / self.LIGHT_ELEVATION_DEG, rebuilds
        # [LIGHTING DISABLED] the light basis + shadow tree, and returns (azimuth, elevation)
        # [LIGHTING DISABLED] so a caller (e.g. the GUI) can sync slider positions to match.
        # [LIGHTING DISABLED] Returns None if the room has no colors to estimate from.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] pts = np.asarray(self.room_pcd.points)
        # [LIGHTING DISABLED] if not self.room_pcd.has_colors() or len(pts) < 200:
            # [LIGHTING DISABLED] print("[light] Room has no usable colors/points to estimate from.")
            # [LIGHTING DISABLED] return None

        # [LIGHTING DISABLED] if not self.room_pcd.has_normals():
            # [LIGHTING DISABLED] radius = max(self.room_scale * 0.02, 1e-3)
            # [LIGHTING DISABLED] self.room_pcd.estimate_normals(
                # [LIGHTING DISABLED] search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))

        # [LIGHTING DISABLED] normals = np.asarray(self.room_pcd.normals)
        # [LIGHTING DISABLED] colors = np.asarray(self.room_pcd.colors)
        # [LIGHTING DISABLED] luminance = colors @ np.array([0.2126, 0.7152, 0.0722])

        # [LIGHTING DISABLED] def fit(n_arr, lum_arr):
            # [LIGHTING DISABLED] A = np.hstack([n_arr, np.ones((len(n_arr), 1))])
            # [LIGHTING DISABLED] sol, *_ = np.linalg.lstsq(A, lum_arr, rcond=None)
            # [LIGHTING DISABLED] L = sol[:3]
            # [LIGHTING DISABLED] norm = np.linalg.norm(L)
            # [LIGHTING DISABLED] return L / norm if norm > 1e-8 else None

        # [LIGHTING DISABLED] L0 = fit(normals, luminance)
        # [LIGHTING DISABLED] if L0 is None:
            # [LIGHTING DISABLED] print("[light] Estimation failed (degenerate normals/colors).")
            # [LIGHTING DISABLED] return None

        # Pass 2: refit using only the clearly front-lit points under L0,
        # which removes the terminator-clipping bias (see docstring).
        # [LIGHTING DISABLED] alignment = normals @ L0
        # [LIGHTING DISABLED] mask = alignment > np.percentile(alignment, 60)
        # [LIGHTING DISABLED] L1 = fit(normals[mask], luminance[mask]) if np.count_nonzero(mask) > 10 else L0
        # [LIGHTING DISABLED] L_toward_light = L1 if L1 is not None else L0

        # [LIGHTING DISABLED] light_travel = -L_toward_light  # our convention: direction the light TRAVELS
        # [LIGHTING DISABLED] az = float(np.degrees(np.arctan2(light_travel[1], light_travel[0])) % 360)
        # [LIGHTING DISABLED] el = float(np.degrees(np.arcsin(np.clip(-light_travel[2], -1, 1))))
        # [LIGHTING DISABLED] el = float(np.clip(el, 1.0, 89.0))  # keep it a sane, non-degenerate angle

        # [LIGHTING DISABLED] self.LIGHT_AZIMUTH_DEG = az
        # [LIGHTING DISABLED] self.LIGHT_ELEVATION_DEG = el
        # [LIGHTING DISABLED] self._build_light_basis()
        # [LIGHTING DISABLED] self._build_shadow_tree()
        # [LIGHTING DISABLED] print(f"[light] Estimated from scene: azimuth={az:.1f}°, elevation={el:.1f}°")
        # [LIGHTING DISABLED] return az, el

    # ── CHANGE 18 — a visible marker for the CURRENT light, whichever mode ──
    # [LIGHTING DISABLED] def build_light_gizmo(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Dispatches on self.LIGHT_MODE: a small glowing bulb at the point
        # [LIGHTING DISABLED] light's world position, or the original arrow for the directional
        # [LIGHTING DISABLED] "sun". Purely visual — rebuilt via _refresh_light_gizmo() any time
        # [LIGHTING DISABLED] the light moves/changes or the mode is toggled.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] if self.LIGHT_MODE == "point":
            # [LIGHTING DISABLED] return self._build_point_light_gizmo()
        # [LIGHTING DISABLED] return self._build_directional_light_gizmo()

    # [LIGHTING DISABLED] def _build_point_light_gizmo(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] A small bulb (sphere) at self.light_pos, plus a few short radiating
        # [LIGHTING DISABLED] line "rays" purely so it doesn't get lost against the room's own
        # [LIGHTING DISABLED] points — NOT the actual shadow rays used for shading (those go
        # [LIGHTING DISABLED] from every object point to the bulb; see _compute_point_light_darken).
        # [LIGHTING DISABLED] Returns a list of geometries; _refresh_light_gizmo() adds/removes
        # [LIGHTING DISABLED] every item in the list as one unit.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] radius = max(self.room_scale * 0.02, 1e-3)
        # [LIGHTING DISABLED] bulb = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=14)
        # [LIGHTING DISABLED] bulb.paint_uniform_color([1.0, 0.95, 0.55])
        # [LIGHTING DISABLED] bulb.compute_vertex_normals()
        # [LIGHTING DISABLED] bulb.translate(self.light_pos)

        # [LIGHTING DISABLED] dirs = np.array([
            # [LIGHTING DISABLED] [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0],
            # [LIGHTING DISABLED] [0, 0, 1], [0, 0, -1],
            # [LIGHTING DISABLED] [1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0],
        # [LIGHTING DISABLED] ], dtype=np.float64)
        # [LIGHTING DISABLED] dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        # [LIGHTING DISABLED] ray_len = radius * 3.0
        # [LIGHTING DISABLED] starts = self.light_pos[None, :] + dirs * radius * 1.3
        # [LIGHTING DISABLED] ends = starts + dirs * ray_len
        # [LIGHTING DISABLED] line_pts = np.vstack([starts, ends])
        # [LIGHTING DISABLED] n_dirs = len(dirs)
        # [LIGHTING DISABLED] lines = [[i, i + n_dirs] for i in range(n_dirs)]
        # [LIGHTING DISABLED] rays = o3d.geometry.LineSet(
            # [LIGHTING DISABLED] points=o3d.utility.Vector3dVector(line_pts),
            # [LIGHTING DISABLED] lines=o3d.utility.Vector2iVector(lines))
        # [LIGHTING DISABLED] rays.paint_uniform_color([1.0, 0.85, 0.2])

        # [LIGHTING DISABLED] return [bulb, rays]

    # [LIGHTING DISABLED] def _build_directional_light_gizmo(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Returns a TriangleMesh arrow pointing from the light's approximate
        # [LIGHTING DISABLED] origin toward the scene, along the current self.light_dir. Purely
        # [LIGHTING DISABLED] visual — rebuild and re-add it (via build_light_gizmo() again)
        # [LIGHTING DISABLED] any time the light direction changes, so it stays in sync with
        # [LIGHTING DISABLED] the azimuth/elevation sliders.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] center = self.room_pcd.get_axis_aligned_bounding_box().get_center()
        # [LIGHTING DISABLED] origin_dist = self.room_scale * 1.2
        # [LIGHTING DISABLED] arrow_len = self.room_scale * 0.5

        # [LIGHTING DISABLED] arrow = o3d.geometry.TriangleMesh.create_arrow(
            # [LIGHTING DISABLED] cylinder_radius=max(self.room_scale * 0.006, 1e-4),
            # [LIGHTING DISABLED] cone_radius=max(self.room_scale * 0.014, 1e-4),
            # [LIGHTING DISABLED] cylinder_height=arrow_len * 0.75,
            # [LIGHTING DISABLED] cone_height=arrow_len * 0.25)
        # [LIGHTING DISABLED] arrow.paint_uniform_color([1.0, 0.85, 0.0])
        # [LIGHTING DISABLED] arrow.compute_vertex_normals()

        # create_arrow() points along +Z by default; rotate it to point
        # along self.light_dir instead, then place its tail at the
        # light's approximate origin (back away from the room along the
        # direction the light travels FROM).
        # [LIGHTING DISABLED] z_axis = np.array([0.0, 0.0, 1.0])
        # [LIGHTING DISABLED] d = self.light_dir
        # [LIGHTING DISABLED] if np.linalg.norm(np.cross(z_axis, d)) < 1e-6:
            # [LIGHTING DISABLED] R = np.eye(3) if d[2] > 0 else Rotation.from_rotvec(
                # [LIGHTING DISABLED] np.pi * np.array([1.0, 0.0, 0.0])).as_matrix()
        # [LIGHTING DISABLED] else:
            # [LIGHTING DISABLED] axis = np.cross(z_axis, d)
            # [LIGHTING DISABLED] axis /= np.linalg.norm(axis)
            # [LIGHTING DISABLED] angle = np.arccos(np.clip(np.dot(z_axis, d), -1.0, 1.0))
            # [LIGHTING DISABLED] R = Rotation.from_rotvec(axis * angle).as_matrix()
        # [LIGHTING DISABLED] arrow.rotate(R, center=(0, 0, 0))

        # [LIGHTING DISABLED] origin = center - d * origin_dist
        # [LIGHTING DISABLED] arrow.translate(origin)
        # [LIGHTING DISABLED] return arrow

    # ── CHANGE 15 — pure computation, no viewer side-effects, no lag ───────
    # [LIGHTING DISABLED] def _compute_directional_darken(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Returns the per-point darken array for the object's CURRENT
        # [LIGHTING DISABLED] position, using a directional shadow test instead of the old
        # [LIGHTING DISABLED] overhead-count proxy. Does NOT touch obj_geom.colors or call
        # [LIGHTING DISABLED] update_geometry — safe to call as often or as rarely as you like.

        # [LIGHTING DISABLED] Only runs when explicitly asked to (H = preview, or automatically
        # [LIGHTING DISABLED] right before E/X export) — same lag-avoidance rule as before: this
        # [LIGHTING DISABLED] is deliberately NOT wired into the per-keystroke move/rotate
        # [LIGHTING DISABLED] handlers.

        # [LIGHTING DISABLED] Method: project every current object point into light space
        # [LIGHTING DISABLED] (u, v, depth), where smaller depth = closer to the light. For
        # [LIGHTING DISABLED] each point, do a radius search in (u, v) — i.e. "look along the
        # [LIGHTING DISABLED] light ray" — against the static room's KD-tree (built once).
        # [LIGHTING DISABLED] Any room point whose depth is smaller (closer to the light) than
        # [LIGHTING DISABLED] this point's own depth, by more than a small bias, is something
        # [LIGHTING DISABLED] already blocking the light before it reaches this point — count
        # [LIGHTING DISABLED] it as an occluder. The occluder count saturates into a soft
        # [LIGHTING DISABLED] darkening factor, the same way the original overhead-proxy's
        # [LIGHTING DISABLED] point count did, just now restricted to the correct light-space
        # [LIGHTING DISABLED] column instead of a raw vertical one.

        # [LIGHTING DISABLED] A KD-tree radius search (rather than binning into a fixed grid)
        # [LIGHTING DISABLED] is what makes this robust on sparse/uneven point clouds: it finds
        # [LIGHTING DISABLED] whatever real points are actually nearby instead of requiring
        # [LIGHTING DISABLED] them to land in one exact cell, so it doesn't leave "holes" that
        # [LIGHTING DISABLED] silently read as unoccluded.

        # [LIGHTING DISABLED] CHANGE 16 — self-shadow is OFF by default (ENABLE_SELF_SHADOW).
        # [LIGHTING DISABLED] A prior version always let the object shadow its own points too
        # [LIGHTING DISABLED] (for concave objects like a seat shadowing its own legs). That
        # [LIGHTING DISABLED] backfires badly on bushy/cluttered objects — a vase of flowers,
        # [LIGHTING DISABLED] foliage, anything with lots of thin overlapping geometry — because
        # [LIGHTING DISABLED] nearby points at slightly different depths trip the occluder
        # [LIGHTING DISABLED] count almost everywhere, saturating it regardless of light
        # [LIGHTING DISABLED] direction. That's what "changing the light angle does nothing"
        # [LIGHTING DISABLED] looks like: the object is darkening itself, not the room. Turn
        # [LIGHTING DISABLED] ENABLE_SELF_SHADOW back on only for solid, simple, genuinely
        # [LIGHTING DISABLED] concave objects where you've confirmed it helps.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] if self.obj_geom is None or self.obj_base_colors is None:
            # [LIGHTING DISABLED] return None

        # [LIGHTING DISABLED] pts = np.asarray(self.obj_geom.points)
        # [LIGHTING DISABLED] n = len(pts)
        # [LIGHTING DISABLED] u, v, depth = self._to_light_space(pts)
        # [LIGHTING DISABLED] uv = self._light_uv(u, v)

        # [LIGHTING DISABLED] radius = max(self.room_scale * self.SHADOW_RADIUS_FRAC, 1e-4)
        # [LIGHTING DISABLED] bias   = max(self.room_scale * self.SHADOW_BIAS_FRAC, 1e-5)

        # [LIGHTING DISABLED] room_hits = (self.room_shadow_tree.query_ball_point(uv, r=radius, workers=-1)
                     # [LIGHTING DISABLED] if self.room_shadow_tree is not None else [[] for _ in range(n)])

        # [LIGHTING DISABLED] if self.ENABLE_SELF_SHADOW and n > 1:
            # Self-occlusion uses the OBJECT's own scale for radius/bias,
            # not the room's — a vase is far smaller than the room, and
            # sizing its own self-shadow test off the room's scale is what
            # made it saturate almost instantly. Even with this fix,
            # leave it off for cluttered objects; see docstring above.
            # [LIGHTING DISABLED] obj_scale = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
            # [LIGHTING DISABLED] self_radius = max(obj_scale * self.SELF_SHADOW_RADIUS_FRAC, 1e-5)
            # [LIGHTING DISABLED] self_bias   = max(obj_scale * self.SELF_SHADOW_BIAS_FRAC, 1e-6)
            # [LIGHTING DISABLED] self_tree = cKDTree(uv)
            # [LIGHTING DISABLED] self_hits = self_tree.query_ball_point(uv, r=self_radius, workers=-1)
        # [LIGHTING DISABLED] else:
            # [LIGHTING DISABLED] self_hits = None
            # [LIGHTING DISABLED] self_bias = bias  # unused when self_hits is None

        # [LIGHTING DISABLED] darken = np.ones(n, dtype=np.float64)
        # [LIGHTING DISABLED] for i in range(n):
            # [LIGHTING DISABLED] occluding = 0
            # [LIGHTING DISABLED] room_idx = room_hits[i]
            # [LIGHTING DISABLED] if room_idx:
                # [LIGHTING DISABLED] occluding += int(np.count_nonzero(
                    # [LIGHTING DISABLED] self.room_light_depth[room_idx] < depth[i] - bias))
            # [LIGHTING DISABLED] if self_hits is not None:
                # [LIGHTING DISABLED] self_idx = self_hits[i]
                # [LIGHTING DISABLED] if self_idx:
                    # [LIGHTING DISABLED] occluding += int(np.count_nonzero(
                        # [LIGHTING DISABLED] depth[self_idx] < depth[i] - self_bias))
            # [LIGHTING DISABLED] if occluding:
                # [LIGHTING DISABLED] shadow_fraction = min(occluding / self.SHADOW_SATURATE_COUNT, 1.0)
                # [LIGHTING DISABLED] darken[i] = 1.0 - shadow_fraction * self.SHADOW_STRENGTH

        # [LIGHTING DISABLED] return darken[:, None]

    # ── CHANGE 18 — point-light counterpart to _compute_directional_darken ──
    # [LIGHTING DISABLED] def _compute_point_light_darken(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] The point-light analogue of _compute_directional_darken(). A point
        # [LIGHTING DISABLED] light has no single shared direction for the whole scene — every
        # [LIGHTING DISABLED] object point has its OWN direction and distance to self.light_pos —
        # [LIGHTING DISABLED] so there's no one (u, v) plane to project everything onto. Instead,
        # [LIGHTING DISABLED] each object point's shadow ray (itself → the bulb) is marched in
        # [LIGHTING DISABLED] world space and sampled at a handful of points along the way; any
        # [LIGHTING DISABLED] sample that lands near a real room point (via the plain 3D KD-tree
        # [LIGHTING DISABLED] built in _build_shadow_tree) means something in the room is sitting
        # [LIGHTING DISABLED] between that point and the light, so it counts as an occluder —
        # [LIGHTING DISABLED] the direct per-ray equivalent of the directional method's shared
        # [LIGHTING DISABLED] light-space column search.

        # [LIGHTING DISABLED] On top of occlusion, ENABLE_DISTANCE_FALLOFF applies inverse-square
        # [LIGHTING DISABLED] dimming with distance from the bulb — points near the light are
        # [LIGHTING DISABLED] brighter, points far from it are dimmer, exactly like a real lamp.
        # [LIGHTING DISABLED] A directional/sun light has no equivalent of this since it's
        # [LIGHTING DISABLED] treated as infinitely far away, so every point is "the same
        # [LIGHTING DISABLED] distance" from it.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] if self.obj_geom is None or self.obj_base_colors is None:
            # [LIGHTING DISABLED] return None

        # [LIGHTING DISABLED] pts = np.asarray(self.obj_geom.points)
        # [LIGHTING DISABLED] n = len(pts)
        # [LIGHTING DISABLED] vec = self.light_pos[None, :] - pts            # point → light
        # [LIGHTING DISABLED] dist = np.linalg.norm(vec, axis=1)
        # [LIGHTING DISABLED] dist_safe = np.maximum(dist, 1e-6)

        # [LIGHTING DISABLED] darken = np.ones(n, dtype=np.float64)

        # [LIGHTING DISABLED] if self.room_kdtree_3d is not None and n > 0:
            # [LIGHTING DISABLED] n_samp = max(int(self.POINT_SHADOW_SAMPLES), 1)
            # [LIGHTING DISABLED] bias = float(np.clip(self.POINT_SHADOW_BIAS_FRAC, 0.0, 0.45))
            # [LIGHTING DISABLED] ts = np.linspace(bias, 1.0 - bias, n_samp)       # skip both ray ends
            # samples[i, s, :] = pts[i] + ts[s] * vec[i]
            # [LIGHTING DISABLED] samples = pts[:, None, :] + ts[None, :, None] * vec[:, None, :]
            # [LIGHTING DISABLED] samples_flat = samples.reshape(-1, 3)

            # [LIGHTING DISABLED] radius = max(self.room_scale * self.POINT_SHADOW_RADIUS_FRAC, 1e-4)
            # [LIGHTING DISABLED] hits = self.room_kdtree_3d.query_ball_point(samples_flat, r=radius, workers=-1)
            # [LIGHTING DISABLED] hit_counts = np.fromiter((len(h) for h in hits), dtype=np.int64, count=len(hits))
            # [LIGHTING DISABLED] occluded_samples = (hit_counts.reshape(n, n_samp) > 0).sum(axis=1)

            # Soft shadow: fraction of the ray's SAMPLED length that's
            # blocked, same saturating-into-a-darkening-factor idea as the
            # directional method (just per-ray instead of per-count-vs-a-
            # fixed-threshold, since here every point has its own ray).
            # [LIGHTING DISABLED] shadow_fraction = np.minimum(occluded_samples / (n_samp * 0.5), 1.0)
            # [LIGHTING DISABLED] darken = 1.0 - shadow_fraction * self.SHADOW_STRENGTH

        # [LIGHTING DISABLED] if self.ENABLE_DISTANCE_FALLOFF:
            # [LIGHTING DISABLED] ref_dist = max(self.room_scale * 0.5, 1e-6)   # ~neutral at mid-room
            # [LIGHTING DISABLED] atten = np.clip((ref_dist / dist_safe) ** 2, 0.35, 1.8)
            # [LIGHTING DISABLED] darken = darken * atten

        # [LIGHTING DISABLED] return darken[:, None]

    # [LIGHTING DISABLED] def _apply_occlusion_shading(self, vis=None):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Computes shading for the CURRENT position and pushes it to both
        # [LIGHTING DISABLED] the viewer (obj_geom.colors) and the export cache
        # [LIGHTING DISABLED] (last_occlusion_darken). Only called explicitly now — see H key
        # [LIGHTING DISABLED] and _export_gaussians/_export_object_only — never from the
        # [LIGHTING DISABLED] per-keystroke move/rotate handlers. Accepts an optional unused
        # [LIGHTING DISABLED] `vis` arg so it can also be wired up directly as a key callback.

        # [LIGHTING DISABLED] CHANGE 18: dispatches on self.LIGHT_MODE — "point" uses the new
        # [LIGHTING DISABLED] in-room point light, "directional" uses the original sun-like
        # [LIGHTING DISABLED] light. Both keep working; the mode just picks which one is live.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] if self.LIGHT_MODE == "point":
            # [LIGHTING DISABLED] darken = self._compute_point_light_darken()
        # [LIGHTING DISABLED] else:
            # [LIGHTING DISABLED] darken = self._compute_directional_darken()
        # [LIGHTING DISABLED] if darken is None:
            # [LIGHTING DISABLED] return False
        # [LIGHTING DISABLED] self.last_occlusion_darken = darken
        # [LIGHTING DISABLED] shaded = np.clip(self.obj_base_colors * darken, 0.0, 1.0)
        # [LIGHTING DISABLED] self.obj_geom.colors = o3d.utility.Vector3dVector(shaded)
        # [LIGHTING DISABLED] self.vis.update_geometry(self.obj_geom)
        # [LIGHTING DISABLED] if self.LIGHT_MODE == "point":
            # [LIGHTING DISABLED] print(f"[shading] Re-lit object at current position "
                  # [LIGHTING DISABLED] f"(mean darken factor: {darken.mean():.2f}, "
                  # [LIGHTING DISABLED] f"point light at {np.round(self.light_pos, 2).tolist()}).")
        # [LIGHTING DISABLED] else:
            # [LIGHTING DISABLED] print(f"[shading] Re-lit object at current position "
                  # [LIGHTING DISABLED] f"(mean darken factor: {darken.mean():.2f}, "
                  # [LIGHTING DISABLED] f"light az={self.LIGHT_AZIMUTH_DEG:.0f}° el={self.LIGHT_ELEVATION_DEG:.0f}°).")
        # [LIGHTING DISABLED] return False

    # ── geometry transforms (pivot = object's own centroid) ────────────────
    def _apply_translation(self, delta):
        pts = np.asarray(self.obj_geom.points)
        pts = pts + delta
        self.obj_geom.points = o3d.utility.Vector3dVector(pts)
        self.centroid += delta
        # CHANGE 13: occlusion shading is now computed ONCE, at export time
        # (see _export_gaussians / _export_object_only), not on every
        # keypress — the KD-tree query over every object point was the
        # source of the GUI lag during live move/rotate. The viewer shows
        # the object at true (unshaded) color while you're positioning it;
        # shading is only baked in right before writing the .ply.
        self.vis.update_geometry(self.obj_geom)
        self._refresh_bbox()

    def _apply_rotation(self, axis, degrees):
        if self.obj_geom is None:
            return
        R = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array(axis) * np.radians(degrees))
        pts = np.asarray(self.obj_geom.points)
        pts = (pts - self.centroid) @ R.T + self.centroid
        self.obj_geom.points = o3d.utility.Vector3dVector(pts)
        # CHANGE 13: no live re-shade here either — see _apply_translation.
        self.vis.update_geometry(self.obj_geom)
        self._refresh_bbox()
        # Keep the accumulated rotation in sync so export can re-pose the
        # *actual* fitted Gaussians the same way this proxy was rotated.
        self.R_total = R @ self.R_total

    def _refresh_bbox(self):
        if getattr(self, "bbox_geom", None) is None:
            return
        new_box = self.obj_geom.get_axis_aligned_bounding_box()
        self.bbox_geom.min_bound = new_box.min_bound
        self.bbox_geom.max_bound = new_box.max_bound
        self.vis.update_geometry(self.bbox_geom)

    def _reset_object(self, vis=None):
        if self.obj_geom is None:
            return False
        self.obj_geom.points = o3d.utility.Vector3dVector(self.obj_base_points.copy())
        self.obj_geom.colors = o3d.utility.Vector3dVector(self.obj_base_colors.copy())
        self.centroid = self.base_centroid.copy()
        self.R_total = np.eye(3)
        self.last_occlusion_darken = None  # CHANGE 13: stale, matches old pose
        self.vis.update_geometry(self.obj_geom)
        self._refresh_bbox()
        return False

    # ── export the real fitted Gaussians (not the density proxy) ───────────
    def _export_gaussians(self, vis=None):
        if self.obj_geom is None or self.job.gauss is None:
            print("[export] Object hasn't finished splatting yet — wait for "
                  "the '[splat] Attached: …' message, then press E again.")
            return False
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "splat_exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"scene_splat_{ts}.ply")
        # [LIGHTING DISABLED] CHANGE 13: single shading pass here, at the object's
        # FINAL pose — this is the only place the KD-tree query now runs. Also
        # updates obj_geom.colors so the viewer's last-visible frame matches the
        # exported file, without having paid that cost on every move.
        # print("[export] Computing occlusion shading for final pose …")
        # self._apply_occlusion_shading()
        # self.vis.update_geometry(self.obj_geom)
        try:
            n_room, n_obj = export_scene_as_gaussian_splat(
                self.room_pcd, self.job, self.R_total,
                self.base_centroid, self.centroid, out_path,
                src=self.src, room_src_idx=self.room_src_idx,
                darken=None)  # [LIGHTING DISABLED] was: darken=self._export_darken()
            print(f"[export] {n_room:,} room + {n_obj:,} object gaussians "
                  f"({n_room + n_obj:,} total) → {out_path}")
            print("[export] Standard 3DGS layout (x,y,z / f_dc / opacity / "
                  "scale / rot) — open in SuperSplat, gsplat, Postshot, etc. "
                  "Untouched room/object points are written back with their "
                  "ORIGINAL scale/rot/opacity/SH when the source had them. "
                  "Object color includes the viewer's occlusion shading.")
        except Exception:
            import traceback
            traceback.print_exc()
        return False

    def _export_object_only(self, vis=None):
        if self.obj_geom is None or self.job.gauss is None:
            print("[export] Object hasn't finished splatting yet — wait for "
                  "the '[splat] Attached: …' message, then press X again.")
            return False
        out_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "splat_exports")
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"object_splat_{ts}.ply")
        # [LIGHTING DISABLED] CHANGE 13: same one-shot shading pass as scene export.
        # print("[export] Computing occlusion shading for final pose …")
        # self._apply_occlusion_shading()
        # self.vis.update_geometry(self.obj_geom)
        try:
            n = export_object_as_gaussian_splat(
                self.job, self.R_total, self.base_centroid, self.centroid, out_path,
                darken=None)  # [LIGHTING DISABLED] was: darken=self._export_darken()
            print(f"[export] {n:,} gaussians (object only) → {out_path}")
        except Exception:
            import traceback
            traceback.print_exc()
        return False

    # ── key bindings ─────────────────────────────────────────────────────────
    def _register_keys(self):
        STEP = 0.05   # meters per key press, tune to scene scale once known
        ROT  = 5.0    # degrees per key press

        def move(dx=0.0, dy=0.0, dz=0.0):
            def cb(vis):
                if self.obj_geom is not None:
                    self._apply_translation(np.array([dx, dy, dz]) * self._step())
                return False
            return cb

        def rotate(axis, sign):
            def cb(vis):
                if self.obj_geom is not None:
                    self._apply_rotation(axis, sign * ROT)
                return False
            return cb

        self.vis.register_key_callback(ord("J"), move(dx=-1))
        self.vis.register_key_callback(ord("L"), move(dx=+1))
        self.vis.register_key_callback(ord("I"), move(dy=+1))
        self.vis.register_key_callback(ord("K"), move(dy=-1))
        self.vis.register_key_callback(ord("U"), move(dz=-1))
        self.vis.register_key_callback(ord("O"), move(dz=+1))

        self.vis.register_key_callback(ord("N"), rotate([0, 1, 0], -1))
        self.vis.register_key_callback(ord("M"), rotate([0, 1, 0], +1))
        self.vis.register_key_callback(ord("T"), rotate([1, 0, 0], -1))
        self.vis.register_key_callback(ord("G"), rotate([1, 0, 0], +1))
        self.vis.register_key_callback(ord("P"), rotate([0, 0, 1], -1))
        self.vis.register_key_callback(ord("Y"), rotate([0, 0, 1], +1))

        self.vis.register_key_callback(ord("R"), self._reset_object)
        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("H"), self._apply_occlusion_shading)
        self.vis.register_key_callback(ord("E"), self._export_gaussians)
        self.vis.register_key_callback(ord("X"), self._export_object_only)

        # ── CHANGE 18 — move the point light around inside the room ────────
        # Only acts while LIGHT_MODE == "point" (harmless no-op otherwise).
        # Reuses self._step() so the light nudges by the same room-scaled
        # amount the object does with J/L/I/K/U/O.
        # [LIGHTING DISABLED] def nudge_light(dx=0.0, dy=0.0, dz=0.0):
            # [LIGHTING DISABLED] def cb(vis):
                # [LIGHTING DISABLED] if self.LIGHT_MODE == "point":
                    # [LIGHTING DISABLED] self.move_light(np.array([dx, dy, dz]) * self._step())
                    # [LIGHTING DISABLED] self._refresh_light_gizmo()
                    # [LIGHTING DISABLED] if self.obj_geom is not None:
                        # [LIGHTING DISABLED] self._apply_occlusion_shading()
                # [LIGHTING DISABLED] return False
            # [LIGHTING DISABLED] return cb

        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("A"), nudge_light(dx=-1))
        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("D"), nudge_light(dx=+1))
        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("S"), nudge_light(dy=-1))
        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("W"), nudge_light(dy=+1))
        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("F"), nudge_light(dz=-1))  # down
        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("V"), nudge_light(dz=+1))  # up

        # [LIGHTING DISABLED] def toggle_light_mode(vis):
            # [LIGHTING DISABLED] self.LIGHT_MODE = "directional" if self.LIGHT_MODE == "point" else "point"
            # [LIGHTING DISABLED] self._refresh_light_gizmo()
            # [LIGHTING DISABLED] if self.obj_geom is not None:
                # [LIGHTING DISABLED] self._apply_occlusion_shading()
            # [LIGHTING DISABLED] print(f"[light] Mode switched to '{self.LIGHT_MODE}'.")
            # [LIGHTING DISABLED] return False

        # [LIGHTING DISABLED] self.vis.register_key_callback(ord("B"), toggle_light_mode)

    # ── CHANGE 12 — safe accessor for export ────────────────────────────────
    # [LIGHTING DISABLED] def _export_darken(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Returns the darken array from the most recent occlusion-shading
        # [LIGHTING DISABLED] pass, IF its length still matches job.gauss (it always should,
        # [LIGHTING DISABLED] since obj_geom is job.splat_pcd itself and is never resampled/
        # [LIGHTING DISABLED] reordered — see SplatJob.run). Guards anyway so a future change to
        # [LIGHTING DISABLED] the fitting pipeline fails safe (full brightness) instead of
        # [LIGHTING DISABLED] silently mis-mapping colors to the wrong points on export.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] d = self.last_occlusion_darken
        # [LIGHTING DISABLED] if d is None or self.job.gauss is None:
            # [LIGHTING DISABLED] return None
        # [LIGHTING DISABLED] if len(d) != len(self.job.gauss["mean"]):
            # [LIGHTING DISABLED] print("[export] Occlusion-darken array length mismatch — "
                  # [LIGHTING DISABLED] "exporting at full brightness instead of guessing.")
            # [LIGHTING DISABLED] return None
        # [LIGHTING DISABLED] return d

    def _step(self):
        # Scale movement step to the room's size so it feels consistent
        # across small (table-top) and large (building) scans.
        ext = self.room_pcd.get_axis_aligned_bounding_box().get_extent()
        scale = float(np.linalg.norm(ext))
        return max(scale * 0.005, 0.01)

    # ── inserting the object once the background fit completes ─────────────
    def try_attach_object(self):
        if self.obj_added or not self.job.done.is_set():
            return
        self.obj_added = True
        if self.job.error is not None:
            print("[splat] Fitting failed — see traceback above. Room is still viewable.")
            return
        if self.job.splat_pcd is None or len(self.job.splat_pcd.points) == 0:
            print("[splat] Density cloud has 0 points — nothing to attach.")
            return

        self.obj_geom = self.job.splat_pcd
        self.obj_base_points = np.asarray(self.obj_geom.points).copy()
        self.obj_base_colors = np.asarray(self.obj_geom.colors).copy()  # CHANGE 11

        self.centroid      = self.job.centroid.copy()
        self.base_centroid = self.job.centroid.copy()

        # CHANGE 13: no shading pass here — object attaches at its true,
        # unshaded color. Occlusion shading is deferred to export time (E/X)
        # so positioning the object stays lag-free regardless of point count.
        self.vis.add_geometry(self.obj_geom)

        # Increase point size so the density cloud reads as soft blobs rather
        # than a sparse scatter. The splat cloud has ~25× more points than the
        # original cluster, so they render with natural density falloff.
        opt = self.vis.get_render_option()
        opt.point_size = max(opt.point_size, 2.0)

        # Yellow bbox so the object is always easy to locate in the room.
        bbox = self.obj_geom.get_axis_aligned_bounding_box()
        bbox.color = (1.0, 0.85, 0.0)
        self.bbox_geom = bbox
        self.vis.add_geometry(self.bbox_geom)

        n_gauss = len(self.job.gauss["mean"])
        n_pts   = len(self.obj_base_points)
        print(f"[splat] Attached: {n_gauss:,} gaussians → {n_pts:,} density points, "
              f"centroid={np.round(self.centroid, 3).tolist()}. "
              f"Move J/L/I/K/U/O  Rotate N/M/T/G/P/Y  Reset R  "
              f"Export scene E / object X")

    # ── manual run loop so we can poll the background thread ───────────────
    # ── CHANGE 17 — optional light-control side panel ───────────────────────
    # [LIGHTING DISABLED] def _build_light_panel(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] A small tkinter side window (same toolkit ObjectPickerApp already
        # [LIGHTING DISABLED] uses) with an "Estimate light from scene" button and two sliders
        # [LIGHTING DISABLED] for azimuth/elevation. Runs alongside the existing Open3D viewer —
        # [LIGHTING DISABLED] it does NOT take over the render loop; run() pumps it once per
        # [LIGHTING DISABLED] iteration via panel.update_idletasks()/update(), the same way
        # [LIGHTING DISABLED] vis.poll_events()/update_renderer() are already pumped manually.
        # [LIGHTING DISABLED] This avoids any cross-thread Tk/GLFW interaction entirely.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] root = tk.Tk()
        # [LIGHTING DISABLED] root.title("Light control")
        # [LIGHTING DISABLED] root.geometry("330x560")

        # ── CHANGE 18 — mode toggle: point light (inside the room) vs the
        # original directional "sun" (outside the room). Both stay fully
        # configured at all times; this just picks which one is live.
        # [LIGHTING DISABLED] tk.Label(root, text="Light type", font=("", 10, "bold")).pack(pady=(12, 2))
        # [LIGHTING DISABLED] mode_var = tk.StringVar(value=self.LIGHT_MODE)

        # [LIGHTING DISABLED] def on_mode_changed():
            # [LIGHTING DISABLED] self.LIGHT_MODE = mode_var.get()
            # [LIGHTING DISABLED] self._refresh_light_gizmo()
            # [LIGHTING DISABLED] if self.obj_geom is not None:
                # [LIGHTING DISABLED] self._apply_occlusion_shading()

        # [LIGHTING DISABLED] mode_frame = tk.Frame(root)
        # [LIGHTING DISABLED] mode_frame.pack()
        # [LIGHTING DISABLED] tk.Radiobutton(mode_frame, text="Point (movable, inside the room)",
                       # [LIGHTING DISABLED] variable=mode_var, value="point",
                       # [LIGHTING DISABLED] command=on_mode_changed).pack(anchor="w")
        # [LIGHTING DISABLED] tk.Radiobutton(mode_frame, text="Directional (sun, outside the room)",
                       # [LIGHTING DISABLED] variable=mode_var, value="directional",
                       # [LIGHTING DISABLED] command=on_mode_changed).pack(anchor="w")

        # ── Point light position — the new part: drag it anywhere inside
        # the room's own bounding box. ───────────────────────────────────
        # [LIGHTING DISABLED] point_frame = tk.LabelFrame(root, text="Point light position")
        # [LIGHTING DISABLED] point_frame.pack(fill="x", padx=10, pady=(12, 4))

        # [LIGHTING DISABLED] def on_pos_changed(_=None):
            # [LIGHTING DISABLED] self.light_pos = np.clip(
                # [LIGHTING DISABLED] np.array([x_var.get(), y_var.get(), z_var.get()]),
                # [LIGHTING DISABLED] self.LIGHT_POS_MIN, self.LIGHT_POS_MAX)
            # [LIGHTING DISABLED] self._refresh_light_gizmo()
            # [LIGHTING DISABLED] if self.obj_geom is not None:
                # [LIGHTING DISABLED] self._apply_occlusion_shading()

        # [LIGHTING DISABLED] x_var = tk.DoubleVar(value=round(float(self.light_pos[0]), 3))
        # [LIGHTING DISABLED] y_var = tk.DoubleVar(value=round(float(self.light_pos[1]), 3))
        # [LIGHTING DISABLED] z_var = tk.DoubleVar(value=round(float(self.light_pos[2]), 3))
        # [LIGHTING DISABLED] for label, var, lo, hi in (
            # [LIGHTING DISABLED] ("X", x_var, self.LIGHT_POS_MIN[0], self.LIGHT_POS_MAX[0]),
            # [LIGHTING DISABLED] ("Y", y_var, self.LIGHT_POS_MIN[1], self.LIGHT_POS_MAX[1]),
            # [LIGHTING DISABLED] ("Z  (height)", z_var, self.LIGHT_POS_MIN[2], self.LIGHT_POS_MAX[2]),
        # [LIGHTING DISABLED] ):
            # [LIGHTING DISABLED] tk.Label(point_frame, text=label).pack()
            # [LIGHTING DISABLED] step = max((hi - lo) / 200.0, 1e-4)
            # [LIGHTING DISABLED] tk.Scale(point_frame, from_=float(lo), to=float(hi), orient="horizontal",
                     # [LIGHTING DISABLED] variable=var, resolution=step, length=290,
                     # [LIGHTING DISABLED] command=on_pos_changed).pack()

        # [LIGHTING DISABLED] falloff_var = tk.BooleanVar(value=self.ENABLE_DISTANCE_FALLOFF)

        # [LIGHTING DISABLED] def on_falloff_changed():
            # [LIGHTING DISABLED] self.ENABLE_DISTANCE_FALLOFF = falloff_var.get()
            # [LIGHTING DISABLED] if self.obj_geom is not None:
                # [LIGHTING DISABLED] self._apply_occlusion_shading()

        # [LIGHTING DISABLED] tk.Checkbutton(point_frame, text="Dim with distance from the bulb",
                       # [LIGHTING DISABLED] variable=falloff_var,
                       # [LIGHTING DISABLED] command=on_falloff_changed).pack(pady=(2, 6))

        # ── Directional angle — the original sun controls, unchanged ──────
        # [LIGHTING DISABLED] dir_frame = tk.LabelFrame(root, text="Directional light angle")
        # [LIGHTING DISABLED] dir_frame.pack(fill="x", padx=10, pady=(4, 4))

        # [LIGHTING DISABLED] def on_light_changed(_=None):
            # [LIGHTING DISABLED] self.LIGHT_AZIMUTH_DEG = az_var.get()
            # [LIGHTING DISABLED] self.LIGHT_ELEVATION_DEG = el_var.get()
            # [LIGHTING DISABLED] self._build_light_basis()
            # [LIGHTING DISABLED] self._build_shadow_tree()
            # [LIGHTING DISABLED] self._refresh_light_gizmo()
            # [LIGHTING DISABLED] if self.obj_geom is not None:
                # [LIGHTING DISABLED] self._apply_occlusion_shading()

        # [LIGHTING DISABLED] tk.Label(dir_frame, text="Azimuth (°)").pack(pady=(6, 0))
        # [LIGHTING DISABLED] az_var = tk.DoubleVar(value=self.LIGHT_AZIMUTH_DEG)
        # [LIGHTING DISABLED] tk.Scale(dir_frame, from_=0, to=360, orient="horizontal",
                 # [LIGHTING DISABLED] variable=az_var, resolution=1, length=290,
                 # [LIGHTING DISABLED] command=on_light_changed).pack()

        # [LIGHTING DISABLED] tk.Label(dir_frame, text="Elevation (°)").pack(pady=(6, 0))
        # [LIGHTING DISABLED] el_var = tk.DoubleVar(value=self.LIGHT_ELEVATION_DEG)
        # [LIGHTING DISABLED] tk.Scale(dir_frame, from_=1, to=89, orient="horizontal",
                 # [LIGHTING DISABLED] variable=el_var, resolution=1, length=290,
                 # [LIGHTING DISABLED] command=on_light_changed).pack()

        # [LIGHTING DISABLED] def on_estimate():
            # [LIGHTING DISABLED] result = self.estimate_light_from_scene()
            # [LIGHTING DISABLED] if result is None:
                # [LIGHTING DISABLED] return
            # [LIGHTING DISABLED] az, el = result
            # [LIGHTING DISABLED] az_var.set(round(az, 1))
            # [LIGHTING DISABLED] el_var.set(round(el, 1))

            # Also give the POINT light a reasonable starting position along
            # the same estimated direction — walked in from the room center
            # toward the wall the light seems to come from, then clamped
            # inside the room, so switching to point mode afterward doesn't
            # start you at a default ceiling-center guess.
            # [LIGHTING DISABLED] center = np.asarray(self.room_pcd.get_axis_aligned_bounding_box().get_center())
            # [LIGHTING DISABLED] inward = -self.light_dir  # direction from that wall back toward the room
            # [LIGHTING DISABLED] candidate = center + inward * self.room_scale * 0.25
            # [LIGHTING DISABLED] candidate = np.clip(candidate, self.LIGHT_POS_MIN, self.LIGHT_POS_MAX)
            # [LIGHTING DISABLED] self.light_pos = candidate
            # [LIGHTING DISABLED] x_var.set(round(float(candidate[0]), 3))
            # [LIGHTING DISABLED] y_var.set(round(float(candidate[1]), 3))
            # [LIGHTING DISABLED] z_var.set(round(float(candidate[2]), 3))

            # [LIGHTING DISABLED] self._refresh_light_gizmo()
            # [LIGHTING DISABLED] if self.obj_geom is not None:
                # [LIGHTING DISABLED] self._apply_occlusion_shading()

        # [LIGHTING DISABLED] tk.Button(root, text="Estimate light from scene",
                  # [LIGHTING DISABLED] command=on_estimate).pack(pady=(14, 6))
        # [LIGHTING DISABLED] tk.Label(root, text="Point light also moves with A/D/S/W/F/V.\n"
                             # [LIGHTING DISABLED] "B toggles Point ↔ Directional.\n"
                             # [LIGHTING DISABLED] "Object still moves with J/L/I/K/U/O etc.",
                 # [LIGHTING DISABLED] fg="gray30", justify="center").pack(pady=(10, 0))

        # [LIGHTING DISABLED] root.protocol("WM_DELETE_WINDOW", lambda: None)  # closing the panel
                                                          # alone shouldn't
                                                          # kill the viewer
        # [LIGHTING DISABLED] return root

    # [LIGHTING DISABLED] def _refresh_light_gizmo(self):
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] Rebuilds the light gizmo (yellow bulb+rays for a point light, or
        # [LIGHTING DISABLED] the yellow arrow for a directional one) and re-adds it to the SAME
        # [LIGHTING DISABLED] viewer the object/room already live in, so it stays in sync
        # [LIGHTING DISABLED] whenever the light moves/changes (slider drag, key nudge, estimate,
        # [LIGHTING DISABLED] or a mode toggle).

        # [LIGHTING DISABLED] CHANGE 18: build_light_gizmo() can now return either a single
        # [LIGHTING DISABLED] geometry (the directional arrow) or a list of geometries (the
        # [LIGHTING DISABLED] point light's bulb + rays); this normalizes both cases so add/
        # [LIGHTING DISABLED] remove always happens per-item.
        # [LIGHTING DISABLED] """
        # [LIGHTING DISABLED] new_gizmo = self.build_light_gizmo()
        # [LIGHTING DISABLED] new_list = list(new_gizmo) if isinstance(new_gizmo, (list, tuple)) else [new_gizmo]

        # [LIGHTING DISABLED] old_gizmo = getattr(self, "light_gizmo", None)
        # [LIGHTING DISABLED] if old_gizmo is not None:
            # [LIGHTING DISABLED] old_list = list(old_gizmo) if isinstance(old_gizmo, (list, tuple)) else [old_gizmo]
            # [LIGHTING DISABLED] for g in old_list:
                # [LIGHTING DISABLED] self.vis.remove_geometry(g, reset_bounding_box=False)

        # [LIGHTING DISABLED] self.light_gizmo = new_list
        # [LIGHTING DISABLED] for g in new_list:
            # [LIGHTING DISABLED] self.vis.add_geometry(g, reset_bounding_box=False)

    def run(self, light_gui=False):
        print("Room loaded. Splatting selected object in the background …")
        panel = None
        # [LIGHTING DISABLED] light-control side panel is disabled — the panel
        # and its gizmo depended on the (now commented-out) light pipeline.
        # if light_gui:
        #     panel = self._build_light_panel()
        #     self.light_gizmo = None
        #     self._refresh_light_gizmo()
        #     print("[light] Light control panel opened — drag the sliders or "
        #           "click 'Estimate light from scene'.")
        try:
            while True:
                self.try_attach_object()
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
                if panel is not None:
                    panel.update_idletasks()
                    panel.update()
        finally:
            self.vis.destroy_window()
            if panel is not None:
                panel.destroy()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def pick_file_dialog():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select environment point cloud",
        filetypes=[("Point clouds", "*.ply *.pcd *.xyz *.xyzrgb *.pts"), ("All files", "*.*")])
    root.destroy()
    return path or None


def main():
    # CHANGE 17: --light-gui opens the tkinter light-control side panel
    # (sliders + "Estimate light from scene" button) alongside the normal
    # viewer. Stripped out here so it doesn't get mistaken for the filepath
    # positional argument.
    light_gui = "--light-gui" in sys.argv
    argv = [a for a in sys.argv if a != "--light-gui"]

    filepath = argv[1] if len(argv) > 1 else pick_file_dialog()
    if not filepath or not os.path.isfile(filepath):
        print("No valid file selected. Exiting.")
        return

    # ── CHANGE 9: load through SourceSplat, not pcg.load_pcd() ─────────────
    # SourceSplat reads every original vertex property (scale/rot/opacity/SH)
    # once, up front, and keeps them index-aligned in memory for the whole
    # run. `pcd_full` below is a disposable xyz+rgb VIEW of the same data,
    # built for pcg's clustering/ground-removal/viewer functions — those
    # never see (and can't accidentally drop) the real splat attributes,
    # because the real attributes never pass through an o3d.PointCloud.
    print(f"Loading '{filepath}' …")
    src = SourceSplat(filepath)
    pcd_full = src.view_pcd()

    # ── Clustering Workbench ────────────────────────────────────────────────
    # Ground removal on/off + threshold, DBSCAN eps/min_points sliders,
    # multi-cluster select, preview, and rectangle/lasso clean-up all live
    # in this one interactive window now — see ClusteringWorkbench above.
    # It owns pcd/labels/n_clusters/nonground_src_idx/ground_src_idx/
    # ground_pcd internally (they can change every time you re-cluster or
    # toggle ground removal), and hands the final versions back here via
    # on_continue once you press Continue.
    print("Opening clustering workbench — configure ground removal / DBSCAN, "
          "select one or more clusters (multi-select supported), preview and "
          "clean up if needed, then Continue.")

    def on_continue(pcd, labels, n_clusters, obj_mask,
                     nonground_src_idx, ground_src_idx, ground_pcd):
        room_pcd, obj_pcd, obj_mask = split_room_and_object(pcd, labels, obj_mask)

        # `nonground_src_idx` maps every row of `pcd` back to its row in
        # `src`. Slicing it by the same obj_mask/room_mask used to split
        # room vs. object gives the exact original indices for each half —
        # no re-matching, no re-fitting.
        assert len(nonground_src_idx) == len(obj_mask), (
            f"nonground_src_idx ({len(nonground_src_idx):,}) and obj_mask "
            f"({len(obj_mask):,}) length mismatch — ground-removal index "
            f"bookkeeping is out of sync with the clustered point cloud. "
            f"Try 'Run clustering' again before Continue.")
        obj_src_idx  = nonground_src_idx[obj_mask]
        room_src_idx = np.concatenate(
            [nonground_src_idx[~obj_mask], ground_src_idx])

        # Merge the floor back into the room cloud so the VIEWER shows it
        # (this is display-only — export uses room_src_idx, not room_pcd,
        # whenever src.has_splat_attrs). If ground removal was left off in
        # the workbench, ground_pcd is empty and this is a no-op merge
        # (room_pcd already contains the floor since it was never removed).
        room_pcd = _merge_pcds(room_pcd, ground_pcd)

        # Cover the hollow gap the object leaves behind so it doesn't show
        # once the object is picked up and moved elsewhere in the viewer.
        room_pcd = fill_object_hole(room_pcd, obj_pcd)

        print(f"Object selection: {len(obj_pcd.points):,} pts. "
              f"Room + floor: {len(room_pcd.points):,} pts.")

        # ── CHANGE 10: slice the object's ORIGINAL attributes here, before
        # any fitting happens, and hand them to the job so export can use
        # them instead of a from-scratch PCA fit. ──────────────────────────
        orig_attrs = None
        if src.has_splat_attrs:
            orig_attrs = dict(
                scale_raw=src.scale_raw[obj_src_idx],
                rot_raw=src.rot_raw[obj_src_idx],
                opacity_raw=src.opacity_raw[obj_src_idx],
                f_dc=src.f_dc[obj_src_idx],
                f_rest=src.f_rest[obj_src_idx],
            )

        job = SplatJob()
        # Parallel: fit Gaussians for the object on a background thread
        # (this fit is now ONLY used for the interactive viewer proxy —
        # export reads orig_attrs directly when available) …
        threading.Thread(target=job.run, args=(obj_pcd,), kwargs=dict(orig_attrs=orig_attrs),
                          daemon=True).start()
        # … while the room viewer opens and runs immediately.
        scene = CombinedScene(room_pcd, job, src=src, room_src_idx=room_src_idx)
        scene.run(light_gui=light_gui)

    workbench_root = tk.Tk()
    ClusteringWorkbench(workbench_root, pcd_full, on_continue)
    workbench_root.mainloop()


if __name__ == "__main__":
    main()