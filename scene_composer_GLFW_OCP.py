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
  H                  Preview occlusion shading at the object's CURRENT pose
                     (recomputes + shows it in the viewer without exporting)
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
# Step 1 — picker GUI (single-select "object" + Continue)
# ════════════════════════════════════════════════════════════════════════════

class ObjectPickerApp:
    """Minimal single-selection variant of point_cloud_gui's ClusterApp.
    Takes already-loaded/clustered data (shared with the labeled 3D overview
    so the scene isn't loaded/clustered twice)."""

    def __init__(self, root, pcd, labels, n_clusters, palette, cluster_meta, on_continue,
                 preselect_cid=None):
        self.root         = root
        self.pcd          = pcd
        self.labels       = labels
        self.n_clusters   = n_clusters
        self.palette      = palette
        self.cluster_meta = cluster_meta
        self.on_continue  = on_continue
        self.selected_cid = None
        self.preselect_cid = preselect_cid

        root.title("Pick the object to extract")
        root.configure(bg="#14151f")
        root.geometry("340x560")
        self._build_ui()
        self._populate()

    # ── UI ───────────────────────────────────────────────────────────────────
    def _build_ui(self):
        BG, FG = "#14151f", "#e0e4f5"
        f = tk.Frame(self.root, bg=BG, padx=14, pady=10)
        f.pack(fill="both", expand=True)

        tk.Label(f, text="CLUSTERS (pick one — IDs match the 3D overview)",
                 bg=BG, fg="#6a7fc1", font=("Segoe UI", 8, "bold")).pack(anchor="w")

        lf = tk.Frame(f, bg=BG); lf.pack(fill="both", expand=True, pady=4)
        sb = tk.Scrollbar(lf, orient="vertical")
        self.clist = tk.Listbox(lf, bg="#1e1f2e", fg=FG, selectbackground="#353850",
                                 selectforeground=FG, relief="flat", bd=0,
                                 font=("Consolas", 9), exportselection=False,
                                 yscrollcommand=sb.set)
        sb.config(command=self.clist.yview)
        self.clist.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        self.clist.bind("<<ListboxSelect>>", self._on_select)

        self.status = tk.Label(f, text="", bg=BG, fg="#7a80a0",
                                font=("Segoe UI", 9), anchor="w")
        self.status.pack(fill="x", pady=(4, 8))

        self.continue_btn = tk.Button(
            f, text="Continue ▶", state="disabled", command=self._continue,
            bg="#1e3a5f", fg=FG, activebackground="#2a4d7a", relief="flat",
            padx=10, pady=8, cursor="hand2")
        self.continue_btn.pack(fill="x")

    def _populate(self):
        cids = sorted(self.cluster_meta, key=lambda c: -self.cluster_meta[c]["n_points"])
        self.clist.delete(0, "end")
        for cid in cids:
            m = self.cluster_meta[cid]
            self.clist.insert("end", f"{cid:>3}   {m['n_points']:>7,} pts")
        self._cids_sorted = cids
        # Color each row to match its color in the labeled 3D overview.
        for i, cid in enumerate(cids):
            col = self.palette[cid]
            hexcol = "#{:02x}{:02x}{:02x}".format(
                int(col[0]*200+55), int(col[1]*200+55), int(col[2]*200+55))
            self.clist.itemconfig(i, fg=hexcol)
        self.status.config(text=f"{self.n_clusters} clusters. Select one, then Continue.")

        if self.preselect_cid is not None and self.preselect_cid in self._cids_sorted:
            row = self._cids_sorted.index(self.preselect_cid)
            self.clist.selection_clear(0, "end")
            self.clist.selection_set(row)
            self.clist.see(row)
            self.selected_cid = self.preselect_cid
            self.continue_btn.config(state="normal")
            m = self.cluster_meta[self.preselect_cid]
            self.status.config(
                text=f"Pre-selected cluster {self.preselect_cid} from your click "
                     f"({m['n_points']:,} pts). Ready, or pick a different one.")

    def _on_select(self, _evt):
        sel = self.clist.curselection()
        if not sel:
            return
        self.selected_cid = self._cids_sorted[sel[0]]
        self.continue_btn.config(state="normal")
        m = self.cluster_meta[self.selected_cid]
        self.status.config(text=f"Selected cluster {self.selected_cid} "
                                 f"({m['n_points']:,} pts). Ready.")

    def _continue(self):
        if self.selected_cid is None:
            return
        self.root.destroy()
        self.on_continue(self.pcd, self.labels, self.n_clusters, self.selected_cid)


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — split room vs. object
# ════════════════════════════════════════════════════════════════════════════

def split_room_and_object(pcd, labels, cid):
    """Returns (room_pcd_without_object, object_pcd, obj_mask).

    CHANGE 6: now also returns the boolean obj_mask. This is the same mask
    used to slice `pcd`/`cols` here, so it can be reused, unmodified, to
    slice SourceSplat's original scale/rot/opacity/f_dc/f_rest arrays —
    the room/object split no longer needs a second, separate pass just to
    figure out "which original point is this."
    """
    pts  = np.asarray(pcd.points)
    cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(pts)

    obj_mask  = labels == cid
    room_mask = ~obj_mask

    room = o3d.geometry.PointCloud()
    room.points = o3d.utility.Vector3dVector(pts[room_mask])
    room.colors = o3d.utility.Vector3dVector(cols[room_mask])

    obj = o3d.geometry.PointCloud()
    obj.points = o3d.utility.Vector3dVector(pts[obj_mask])
    obj.colors = o3d.utility.Vector3dVector(cols[obj_mask])

    return room, obj, obj_mask


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

        # ── CHANGE 11 — "under something" occlusion shading ─────────────────
        # Cheap, non-physical proxy for ambient occlusion: when the object
        # sits underneath other room geometry (a shelf, another object,
        # ceiling clutter, etc.) it should read as dimmer than something
        # sitting out in the open, the way real bounce/ambient light gets
        # blocked by whatever is overhead. Tunables below; radius/height are
        # FRACTIONS of the room's scale so behavior is consistent whether
        # this is a tabletop scan or a warehouse scan.
        self.OCCLUSION_RADIUS_FRAC   = 0.03  # how wide a column to look up through
        self.OCCLUSION_HEIGHT_FRAC   = 0.25  # how far up to look for overhead points
        self.OCCLUSION_SATURATE_COUNT = 15   # overhead-point count that = full shadow
        self.OCCLUSION_STRENGTH      = 0.65  # max darkening (0=no effect, 1=black)
        self.room_kdtree = None
        self.room_xy = None
        self.room_z = None
        self.last_occlusion_darken = None  # CHANGE 12: cached per-point darken
                                            # array from the most recent shading
                                            # pass, reused at export time so the
                                            # exported .ply matches the viewer.
        self.room_scale = float(np.linalg.norm(
            self.room_pcd.get_axis_aligned_bounding_box().get_extent()))
        self._build_room_kdtree()

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="Scene — room + movable splat object",
                                width=1280, height=720)
        self.vis.add_geometry(self.room_pcd)

        opt = self.vis.get_render_option()
        opt.background_color = np.array([0.08, 0.08, 0.12])
        opt.point_size = 2.0
        opt.show_coordinate_frame = True

        self._register_keys()

    # ── CHANGE 11 — build once; the room never moves in this tool ──────────
    def _build_room_kdtree(self):
        """
        2D (X,Y) KD-tree over the static room, used only to look straight
        UP (+Z, this pipeline's up-axis — see colorize_by_height in
        point_cloud_render.py / remove_ground in point_cloud_gui.py) from
        each object point for overhead room geometry. Built once in
        __init__ since the room itself is never edited or moved, so every
        subsequent move of the OBJECT can re-query this cheaply instead of
        rebuilding a tree every keypress.
        """
        pts = np.asarray(self.room_pcd.points)
        if len(pts) == 0:
            return
        self.room_xy = pts[:, :2]
        self.room_z = pts[:, 2]
        self.room_kdtree = cKDTree(self.room_xy)

    # ── CHANGE 12b — pure computation, no viewer side-effects, no lag ──────
    def _compute_occlusion_darken(self):
        """
        Returns the per-point darken array for the object's CURRENT
        position, or None if there's nothing to shade yet. Does NOT touch
        obj_geom.colors or call update_geometry — safe to call as often or
        as rarely as you like without affecting viewer responsiveness.

        This used to run on every single J/L/I/K/U/O/N/M/T/G/P/Y keypress
        (i.e. every frame while moving the object), which is what was
        causing the lag — a KD-tree query over every object point, every
        keystroke. It now only runs when explicitly asked to (H = preview,
        or automatically right before E/X export).
        """
        if self.obj_geom is None or self.room_kdtree is None or self.obj_base_colors is None:
            return None

        pts = np.asarray(self.obj_geom.points)
        xy = pts[:, :2]
        z = pts[:, 2]

        radius = max(self.room_scale * self.OCCLUSION_RADIUS_FRAC, 1e-3)
        height = max(self.room_scale * self.OCCLUSION_HEIGHT_FRAC, 1e-3)

        neighbor_lists = self.room_kdtree.query_ball_point(xy, r=radius, workers=-1)

        coverage = np.zeros(len(pts), dtype=np.float64)
        for i, nbrs in enumerate(neighbor_lists):
            if not nbrs:
                continue
            above = self.room_z[nbrs] - z[i]
            n_over = int(np.sum((above > 0.0) & (above < height)))
            coverage[i] = min(n_over / self.OCCLUSION_SATURATE_COUNT, 1.0)

        return 1.0 - coverage[:, None] * self.OCCLUSION_STRENGTH

    def _apply_occlusion_shading(self, vis=None):
        """
        Computes shading for the CURRENT position and pushes it to both
        the viewer (obj_geom.colors) and the export cache
        (last_occlusion_darken). Only called explicitly now — see H key
        and _export_gaussians/_export_object_only — never from the
        per-keystroke move/rotate handlers. Accepts an optional unused
        `vis` arg so it can also be wired up directly as a key callback.
        """
        darken = self._compute_occlusion_darken()
        if darken is None:
            return False
        self.last_occlusion_darken = darken
        shaded = np.clip(self.obj_base_colors * darken, 0.0, 1.0)
        self.obj_geom.colors = o3d.utility.Vector3dVector(shaded)
        self.vis.update_geometry(self.obj_geom)
        print(f"[shading] Re-lit object at current position "
              f"(mean darken factor: {darken.mean():.2f}).")
        return False

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
        # CHANGE 13: single shading pass here, at the object's FINAL pose —
        # this is the only place the KD-tree query now runs. Also updates
        # obj_geom.colors so the viewer's last-visible frame matches the
        # exported file, without having paid that cost on every move.
        print("[export] Computing occlusion shading for final pose …")
        self._apply_occlusion_shading()
        self.vis.update_geometry(self.obj_geom)
        try:
            n_room, n_obj = export_scene_as_gaussian_splat(
                self.room_pcd, self.job, self.R_total,
                self.base_centroid, self.centroid, out_path,
                src=self.src, room_src_idx=self.room_src_idx,
                darken=self._export_darken())
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
        # CHANGE 13: same one-shot shading pass as scene export.
        print("[export] Computing occlusion shading for final pose …")
        self._apply_occlusion_shading()
        self.vis.update_geometry(self.obj_geom)
        try:
            n = export_object_as_gaussian_splat(
                self.job, self.R_total, self.base_centroid, self.centroid, out_path,
                darken=self._export_darken())
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
        self.vis.register_key_callback(ord("H"), self._apply_occlusion_shading)
        self.vis.register_key_callback(ord("E"), self._export_gaussians)
        self.vis.register_key_callback(ord("X"), self._export_object_only)

    # ── CHANGE 12 — safe accessor for export ────────────────────────────────
    def _export_darken(self):
        """
        Returns the darken array from the most recent occlusion-shading
        pass, IF its length still matches job.gauss (it always should,
        since obj_geom is job.splat_pcd itself and is never resampled/
        reordered — see SplatJob.run). Guards anyway so a future change to
        the fitting pipeline fails safe (full brightness) instead of
        silently mis-mapping colors to the wrong points on export.
        """
        d = self.last_occlusion_darken
        if d is None or self.job.gauss is None:
            return None
        if len(d) != len(self.job.gauss["mean"]):
            print("[export] Occlusion-darken array length mismatch — "
                  "exporting at full brightness instead of guessing.")
            return None
        return d

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
    def run(self):
        print("Room loaded. Splatting selected object in the background …")
        try:
            while True:
                self.try_attach_object()
                if not self.vis.poll_events():
                    break
                self.vis.update_renderer()
        finally:
            self.vis.destroy_window()


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
    filepath = sys.argv[1] if len(sys.argv) > 1 else pick_file_dialog()
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

    # ── Ground removal disabled ─────────────────────────────────────────────
    # pcg.remove_ground() is no longer called. The full scene (floor
    # included) goes straight into clustering, so `pcd` is just `pcd_full`.
    # Since nothing was removed, the "nonground" index mapping back into
    # `src` is simply the identity (every row of pcd_full IS the
    # corresponding row of src, in the same order), and there's no separate
    # ground set left to carve out or merge back in later. ground_src_idx
    # and ground_pcd are kept as empty/no-op values purely so the rest of
    # the pipeline (room_src_idx concatenation, _merge_pcds call) doesn't
    # need any other changes.
    pcd = pcd_full
    nonground_src_idx = np.arange(src.n)
    ground_src_idx = np.array([], dtype=np.int64)
    ground_pcd = o3d.geometry.PointCloud()
    print("[ground] Ground removal is disabled — floor points remain part of the scene/clusters.")

    print("Clustering (DBSCAN) — this can take a while on huge scenes …")
    labels, n_clusters, eps = pcg.cluster_dbscan(pcd)
    palette = pcg.make_palette(n_clusters)
    cluster_meta = build_cluster_meta(pcd, labels, n_clusters)
    print(f"{n_clusters} clusters found (eps={eps:.4f}).")

    print("Opening cluster identification window — shift+click on your "
          "object's points, then press Q to close and continue.")
    candidate_cids = show_labeled_cluster_overview(pcd, labels, n_clusters, palette, cluster_meta)
    preselect_cid = candidate_cids[0] if candidate_cids else None

    def on_continue(pcd, labels, n_clusters, cid):
        room_pcd, obj_pcd, obj_mask = split_room_and_object(pcd, labels, cid)

        # `nonground_src_idx` maps every row of `pcd` back to its row in
        # `src`. Slicing it by the same obj_mask/room_mask used to split
        # room vs. object gives the exact original indices for each half —
        # no re-matching, no re-fitting.
        obj_src_idx  = nonground_src_idx[obj_mask]
        room_src_idx = np.concatenate(
            [nonground_src_idx[~obj_mask], ground_src_idx])

        # Merge the floor back into the room cloud so the VIEWER shows it
        # (this is display-only — export uses room_src_idx, not room_pcd,
        # whenever src.has_splat_attrs). With ground removal disabled,
        # ground_pcd is empty, so this is a no-op merge (room_pcd already
        # contains the floor since it was never removed).
        room_pcd = _merge_pcds(room_pcd, ground_pcd)

        print(f"Object cluster {cid}: {len(obj_pcd.points):,} pts. "
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
        scene.run()

    picker_root = tk.Tk()
    ObjectPickerApp(picker_root, pcd, labels, n_clusters, palette, cluster_meta, on_continue,
                    preselect_cid=preselect_cid)
    picker_root.mainloop()


if __name__ == "__main__":
    main()