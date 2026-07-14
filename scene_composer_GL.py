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
  Z / X              Move light source  -X / +X
  F / H              Move light source  +Y / -Y
  V / B              Move light source  -Z / +Z
  ESC / Q            Quit

Usage
-----
    python scene_composer.py <room.ply>
    python scene_composer.py                     # opens a file dialog
"""

import os
import sys
import copy
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import point_cloud_gui as pcg          # load_pcd, remove_ground, cluster_dbscan, make_palette, export_ply
import gaussian_splat_render as gsr    # estimate_gaussians (slow fallback), GaussianSplatWindow (optional true preview)


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
    """Returns (room_pcd_without_object, object_pcd)."""
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

    return room, obj


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
                             k: int = 100, scale: float = 1.0,
                             chunk_size: int = 50_000) -> dict:
    N = len(points)
    pts64 = points.astype(np.float64)
    tree = cKDTree(pts64)
    cov3d  = np.zeros((N, 6), dtype=np.float32)
    normal = np.zeros((N, 3), dtype=np.float32)
    centroid = pts64.mean(axis=0)

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

        vals, vecs = np.linalg.eigh(C)                                # batched, ascending
        vals = np.clip(vals, 0.0, None)
        mx = vals.max(axis=1, keepdims=True)
        mx = np.where(mx > 0, mx, 1e-6)
        vals = np.maximum(vals, mx * 0.01)

        # Surface normal = eigenvector of the smallest eigenvalue (column 0,
        # since eigh is ascending). Orient outward from the object centroid,
        # vectorized, so lighting doesn't flicker between flipped neighbors.
        n = vecs[:, :, 0]                                             # (n,3)
        outward = chunk - centroid
        flip = np.einsum('ni,ni->n', n, outward) < 0
        n[flip] *= -1
        normal[done:end] = n.astype(np.float32)

        C2 = np.einsum('nij,nj,nkj->nik', vecs, vals, vecs) * (scale ** 2)
        cov3d[done:end] = np.stack(
            [C2[:, 0, 0], C2[:, 0, 1], C2[:, 0, 2],
             C2[:, 1, 1], C2[:, 1, 2], C2[:, 2, 2]], axis=1)

        done = end
        print(f"[splat] fitted {done:,}/{N:,} gaussians …")

    return dict(mean=points.astype(np.float32),
                color=colors.astype(np.float32),
                opacity=np.full(N, 0.85, np.float32),
                cov3d=cov3d,
                normal=normal)


class SplatJob:
    """Shared state between the fitting thread and the combined GL window."""
    def __init__(self):
        self.done  = threading.Event()
        self.error = None
        self.gauss = None   # dict: mean, color, opacity, cov3d  (float32 arrays)

    def run(self, obj_pcd, k=100, scale=1.0):
        try:
            pts = np.asarray(obj_pcd.points, dtype=np.float32)
            cols = (np.asarray(obj_pcd.colors, dtype=np.float32)
                    if obj_pcd.has_colors()
                    else np.full((len(pts), 3), 0.8, np.float32))
            if len(pts) == 0:
                raise ValueError("Selected cluster has 0 points.")
            print(f"[splat] Fitting {len(pts):,} Gaussians  k={k}  scale={scale} …")
            self.gauss = estimate_gaussians_fast(pts, cols, k=k, scale=scale)
            print("[splat] Fit complete — splat will appear in the viewer.")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self.error = exc
        finally:
            self.done.set()




# ════════════════════════════════════════════════════════════════════════════
# Step 4 — Combined moderngl window: room as GL_POINTS + object as EWA splats
# ════════════════════════════════════════════════════════════════════════════
#
# This is the correct architecture for a "real Gaussian splat placed inside a
# point cloud scene":
#
#   • The room point cloud is uploaded to a GPU VBO and drawn each frame with
#     a trivial passthrough shader (gl_Position = proj * view * pos). No
#     Open3D involved at all for the final render.
#
#   • The selected object is rendered with the EXACT same EWA Gaussian splat
#     shader from gaussian_splat_render.py — instanced billboard quads, 2D
#     covariance projection, per-fragment Mahalanobis falloff, alpha blending
#     back-to-front. This is proper Gaussian splatting, not a proxy.
#
#   • Both use the SAME OrbitCamera and the SAME moderngl GL context, so they
#     composite correctly in one window with no context-switching artifacts.
#
#   • The object transform (translation + rotation about its own centroid) is
#     maintained as a 4×4 matrix applied to the original Gaussian means and
#     covariances every frame before depth-sort and EWA projection. Covariances
#     transform as  Σ' = R Σ Rᵀ  (second-order tensor rotation).
#
# Rendering order each frame:
#   1. Clear colour + depth.
#   2. Room: DEPTH_TEST on, depth write on, no blending → sets depth buffer.
#   3. Splat: depth TEST on (read-only, so splat is occluded by room walls),
#             blending on, back-to-front sorted → proper alpha composite on
#             top of the room without poking through walls.
#
# Controls  (same feel as gaussian_splat_render.py)
# ─────────────────────────────────────────────────
#   Left-drag          orbit camera      Right-drag  pan
#   Scroll             zoom
#   WASD               orbit (keyboard)  QE          zoom in/out
#   J / L              move object  −X / +X
#   I / K              move object  +Y / −Y
#   U / O              move object  −Z / +Z
#   N / M              yaw   object  (Y axis)
#   T / G              pitch object  (X axis)
#   P / Y              roll  object  (Z axis)
#   R                  reset object to original position & orientation
#   ESC                quit

import moderngl
import moderngl_window as mglw

# ── Room passthrough shader ───────────────────────────────────────────────────
ROOM_VERT = """
#version 330 core
in vec3 a_pos;
in vec3 a_color;
uniform mat4 u_view;
uniform mat4 u_proj;
out vec3 v_color;
void main() {
    gl_Position = u_proj * u_view * vec4(a_pos, 1.0);
    v_color = a_color;
}
"""

ROOM_FRAG = """
#version 330 core
in vec3 v_color;
out vec4 fragColor;
void main() {
    fragColor = vec4(v_color, 1.0);
}
"""

# ── Combined window ───────────────────────────────────────────────────────────

class CombinedGaussianWindow(mglw.WindowConfig):
    """
    One moderngl window that renders:
      • the room as a static GL_POINTS cloud (depth-buffered background)
      • the Gaussian-splatted object with full EWA shader (movable foreground)
    """
    title       = ("Scene Composer  |  WASD/drag=orbit  scroll=zoom  "
                   "J/L/I/K/U/O=move  N/M/T/G/P/Y=rotate  Z/X/F/H/V/B=light  "
                   "R=reset  ESC=quit")
    gl_version  = (3, 3)
    window_size = (1280, 720)
    resizable   = True

    # ── class-level slots filled by run_combined_window() before mglw.run_window_config()
    _room_pts  : np.ndarray = None   # (M, 3) float32
    _room_cols : np.ndarray = None   # (M, 3) float32
    _job       : "SplatJob" = None   # may still be running when window opens

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self._splat_ready = False   # True once GL buffers for the object are built
        self._splat_error = False

        # ── room geometry ────────────────────────────────────────────────────
        room_pts  = CombinedGaussianWindow._room_pts
        room_cols = CombinedGaussianWindow._room_cols
        self._n_room = len(room_pts)

        # Compute scene scale for movement step
        ext = room_pts.max(0) - room_pts.min(0)
        self._move_step = float(np.linalg.norm(ext)) * 0.005

        # ── camera centred on the room ───────────────────────────────────────
        centre = room_pts.mean(0).astype(np.float64)
        dists  = np.linalg.norm(room_pts - centre, axis=1)
        radius = float(np.percentile(dists, 90)) * 3.5
        W, H = self.window_size
        self.cam   = gsr.OrbitCamera(centre, radius, aspect=W/H)
        self._W, self._H = W, H

        # ── key light: fixed in room space (not attached to the camera), so
        # rotating the object with N/M/T/G visibly changes its shading — that
        # visible reaction is the "interacts with light" behavior. ──────────
        self._light_pos   = (centre + radius * np.array([0.55, 1.25, 0.85])).astype(np.float32)
        self._light_color = np.array([1.0, 0.97, 0.9], dtype=np.float32)
        self._ambient      = 0.28

        # ── object transform state ───────────────────────────────────────────
        self._obj_translation = np.zeros(3, np.float64)
        self._obj_rotation    = np.eye(3, dtype=np.float64)  # accumulated rotation
        self._obj_centroid    = None   # set when splat attaches

        # ── mouse state ──────────────────────────────────────────────────────
        self._mouse_pos   = (0, 0)
        self._mouse_left  = False
        self._mouse_right = False

        # ── depth-sort cache ─────────────────────────────────────────────────
        self._sort_order   = None
        self._last_cam_row = None

        # ── build GL resources ───────────────────────────────────────────────
        self._build_room_gl(room_pts, room_cols)

        print("[scene] Window open. Room rendered. "
              "Waiting for Gaussian fit to complete …")

    # ── GL setup ─────────────────────────────────────────────────────────────

    def _build_room_gl(self, pts, cols):
        self.room_prog = self.ctx.program(
            vertex_shader=ROOM_VERT, fragment_shader=ROOM_FRAG)
        data = np.hstack([pts.astype(np.float32),
                          cols.astype(np.float32)]).tobytes()
        self.room_vbo = self.ctx.buffer(data)
        self.room_vao = self.ctx.vertex_array(
            self.room_prog,
            [(self.room_vbo, "3f 3f", "a_pos", "a_color")])

    def _build_splat_gl(self, g: dict):
        """Called on the GL thread (from on_render) once the fit job completes."""
        self.means_orig   = g["mean"].astype(np.float32)
        self.colors_orig  = g["color"].astype(np.float32)
        self.opacities    = g["opacity"].astype(np.float32)
        self.cov3d_orig   = g["cov3d"].astype(np.float32)
        self.normals_orig = g.get("normal", np.zeros_like(self.means_orig)).astype(np.float32)
        self._N           = len(self.means_orig)
        self._obj_centroid = self.means_orig.mean(0).astype(np.float64)

        self.splat_prog = self.ctx.program(
            vertex_shader=gsr.VERT, fragment_shader=gsr.FRAG)

        quad = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())

        # Instance buffer: [mean(3) | color(3) | opacity(1) | cov2d(3) | normal(3)]
        self._ibuf = np.zeros((self._N, 13), dtype=np.float32)
        self._ibuf[:, 0:3] = self.means_orig
        self._ibuf[:, 3:6] = self.colors_orig
        self._ibuf[:, 6]   = self.opacities
        self.inst_vbo = self.ctx.buffer(self._ibuf.tobytes(), dynamic=True)

        self.splat_vao = self.ctx.vertex_array(
            self.splat_prog,
            [(self.quad_vbo, "2f",              "a_quad"),
             (self.inst_vbo, "3f 3f 1f 3f 3f /i",
              "a_mean", "a_color", "a_opacity", "a_cov2d", "a_normal")])

        print(f"[scene] Splat attached: {self._N:,} Gaussians. "
              f"Centroid = {np.round(self._obj_centroid, 3).tolist()}")

    # ── per-frame transform helpers ───────────────────────────────────────────

    def _transformed_means(self) -> np.ndarray:
        """Apply accumulated rotation (around centroid) then translation."""
        R  = self._obj_rotation
        c  = self._obj_centroid
        pts = (self.means_orig.astype(np.float64) - c) @ R.T + c
        pts += self._obj_translation
        return pts.astype(np.float32)

    def _transformed_normals(self) -> np.ndarray:
        """Rotate normals with the object (direction only — no translation)."""
        R = self._obj_rotation
        n = self.normals_orig.astype(np.float64) @ R.T
        return n.astype(np.float32)

    def _transformed_cov3d(self) -> np.ndarray:
        """Rotate each 3×3 covariance:  Σ' = R Σ Rᵀ  (vectorised)."""
        R = self._obj_rotation.astype(np.float64)
        c = self.cov3d_orig.astype(np.float64)
        N = len(c)
        C = np.zeros((N, 3, 3))
        C[:,0,0]=c[:,0]; C[:,0,1]=C[:,1,0]=c[:,1]; C[:,0,2]=C[:,2,0]=c[:,2]
        C[:,1,1]=c[:,3]; C[:,1,2]=C[:,2,1]=c[:,4]; C[:,2,2]=c[:,5]
        RC = np.einsum('ij,njk->nik', R, C)
        Cp = np.einsum('nij,kj->nik', RC, R)
        return np.stack([Cp[:,0,0],Cp[:,0,1],Cp[:,0,2],
                         Cp[:,1,1],Cp[:,1,2],Cp[:,2,2]], axis=1).astype(np.float32)

    def _cached_depth_sort(self, V, means):
        cam_row = V[2:3, :3].copy()
        if (self._sort_order is None or self._last_cam_row is None or
                not np.allclose(cam_row, self._last_cam_row, atol=1e-6)):
            self._sort_order   = gsr.depth_sort(means, V)
            self._last_cam_row = cam_row
        return self._sort_order

    # ── render loop ───────────────────────────────────────────────────────────

    def on_render(self, time: float, frame_time: float):
        # ── Poll for job completion (GL buffer init MUST be on GL thread) ────
        job = CombinedGaussianWindow._job
        if not self._splat_ready and not self._splat_error and job.done.is_set():
            if job.error:
                print(f"[splat] Fit failed: {job.error}")
                self._splat_error = True
            else:
                self._build_splat_gl(job.gauss)
                self._splat_ready = True

        V = self.cam.view()
        P = self.cam.proj()
        W, H = self._W, self._H

        # ── Pass 1: room — depth test + write ON, blending OFF ───────────────
        self.ctx.clear(0.04, 0.04, 0.06, 1.0)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.point_size = 2.0

        self.room_prog["u_view"].write(V.T.tobytes())
        self.room_prog["u_proj"].write(P.T.tobytes())
        self.room_vao.render(moderngl.POINTS)

        # ── Pass 2: Gaussian splats — depth TEST (read-only), blending ON ────
        if self._splat_ready:
            means   = self._transformed_means()
            cov3d   = self._transformed_cov3d()
            normals = self._transformed_normals()
            order   = self._cached_depth_sort(V, means)
            c2d     = gsr.project_cov2d(means, cov3d, V, P, W, H)

            buf = self._ibuf
            buf[order, 0:3]   = means[order]
            buf[order, 3:6]   = self.colors_orig[order]
            buf[order, 6]     = self.opacities[order]
            buf[order, 7:10]  = c2d[order]
            buf[order, 10:13] = normals[order]
            self.inst_vbo.write(buf.tobytes())

            # Read-only depth: splat is occluded by room walls but not by
            # other splats (they alpha-composite among themselves correctly).
            self.ctx.depth_mask = False
            self.ctx.enable(moderngl.BLEND)
            self.ctx.blend_func = (moderngl.SRC_ALPHA,
                                   moderngl.ONE_MINUS_SRC_ALPHA)

            self.splat_prog["u_view"].write(V.T.tobytes())
            self.splat_prog["u_proj"].write(P.T.tobytes())
            self.splat_prog["u_viewport"].value = (float(W), float(H))
            self.splat_prog["u_light_pos"].value   = tuple(self._light_pos.tolist())
            self.splat_prog["u_view_pos"].value    = tuple(self.cam.position.astype(np.float32).tolist())
            self.splat_prog["u_light_color"].value = tuple(self._light_color.tolist())
            self.splat_prog["u_ambient"].value     = self._ambient
            self.splat_vao.render(moderngl.TRIANGLE_STRIP,
                                   vertices=4, instances=self._N)
            self.ctx.depth_mask = True   # restore

    # ── object movement / rotation helpers ───────────────────────────────────

    def _rotate_obj(self, axis: list, degrees: float):
        if self._obj_centroid is None:
            return
        R = gsr.OrbitCamera.__new__(gsr.OrbitCamera)   # just need the math
        rad  = np.radians(degrees)
        a    = np.array(axis, np.float64)
        a   /= np.linalg.norm(a)
        # Rodrigues' formula
        K    = np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        Rdel = np.eye(3) + np.sin(rad)*K + (1-np.cos(rad))*(K@K)
        self._obj_rotation    = Rdel @ self._obj_rotation
        self._sort_order      = None   # invalidate depth-sort cache

    # ── input handlers ────────────────────────────────────────────────────────

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        k   = self.wnd.keys
        ROT = 5.0

        # camera
        if   key == k.ESCAPE: self.wnd.close()
        elif key == k.A:      self.cam.orbit(-15, 0)
        elif key == k.D:      self.cam.orbit( 15, 0)
        elif key == k.W:      self.cam.orbit(0,  15)
        elif key == k.S:      self.cam.orbit(0, -15)
        elif key == k.Q:      self.cam.zoom( 1)
        elif key == k.E:      self.cam.zoom(-1)

        # object translate
        elif key == k.J:      self._obj_translation[0] -= self._move_step
        elif key == k.L:      self._obj_translation[0] += self._move_step
        elif key == k.I:      self._obj_translation[1] += self._move_step
        elif key == k.K:      self._obj_translation[1] -= self._move_step
        elif key == k.U:      self._obj_translation[2] -= self._move_step
        elif key == k.O:      self._obj_translation[2] += self._move_step

        # object rotate
        elif key == k.N:      self._rotate_obj([0,1,0], -ROT)
        elif key == k.M:      self._rotate_obj([0,1,0],  ROT)
        elif key == k.T:      self._rotate_obj([1,0,0], -ROT)
        elif key == k.G:      self._rotate_obj([1,0,0],  ROT)
        elif key == k.P:      self._rotate_obj([0,0,1], -ROT)
        elif key == k.Y:      self._rotate_obj([0,0,1],  ROT)

        # reset
        elif key == k.R:
            self._obj_translation[:] = 0.0
            self._obj_rotation = np.eye(3, dtype=np.float64)
            self._sort_order   = None

        # light source — move it and watch the splat shading react live
        elif key == k.Z:      self._light_pos[0] -= self._move_step * 4
        elif key == k.X:      self._light_pos[0] += self._move_step * 4
        elif key == k.F:      self._light_pos[1] += self._move_step * 4
        elif key == k.H:      self._light_pos[1] -= self._move_step * 4
        elif key == k.V:      self._light_pos[2] -= self._move_step * 4
        elif key == k.B:      self._light_pos[2] += self._move_step * 4

    def on_mouse_press_event(self, x, y, button):
        self._mouse_pos = (x, y)
        if button == 1: self._mouse_left  = True
        else:           self._mouse_right = True

    def on_mouse_release_event(self, x, y, button):
        self._mouse_left = self._mouse_right = False

    def on_mouse_drag_event(self, x, y, dx, dy):
        self._mouse_pos = (x, y)
        if self._mouse_left:  self.cam.orbit(dx, dy)
        if self._mouse_right: self.cam.pan(dx, dy)

    def on_mouse_position_event(self, x, y, dx, dy):
        px, py = self._mouse_pos
        self._mouse_pos = (x, y)
        cdx, cdy = x-px, y-py
        if self._mouse_left:  self.cam.orbit(cdx, cdy)
        if self._mouse_right: self.cam.pan(cdx, cdy)

    def on_mouse_scroll_event(self, x_offset, y_offset):
        self.cam.zoom(y_offset)

    def on_resize(self, w, h):
        self._W, self._H = w, h
        self.cam.aspect  = w / max(h, 1)
        self.ctx.viewport = (0, 0, w, h)


def run_combined_window(room_pcd: o3d.geometry.PointCloud, job: "SplatJob"):
    """
    Starts the fitting job on a background thread, then immediately opens
    the combined moderngl window. The room appears right away; the splat
    object fades in as soon as the fit completes (polled in on_render).
    """
    room_pts  = np.asarray(room_pcd.points, dtype=np.float32)
    room_cols = (np.asarray(room_pcd.colors, dtype=np.float32)
                 if room_pcd.has_colors()
                 else np.full((len(room_pts), 3), 0.5, np.float32))

    CombinedGaussianWindow._room_pts  = room_pts
    CombinedGaussianWindow._room_cols = room_cols
    CombinedGaussianWindow._job       = job

    mglw.run_window_config(CombinedGaussianWindow)


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def pick_file_dialog():
    root = tk.Tk(); root.withdraw()
    path = filedialog.askopenfilename(
        title="Select environment point cloud",
        filetypes=[("Point clouds", "*.ply *.pcd *.xyz *.xyzrgb *.pts"),
                   ("All files", "*.*")])
    root.destroy()
    return path or None


def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else pick_file_dialog()
    if not filepath or not os.path.isfile(filepath):
        print("No valid file selected. Exiting.")
        return

    print(f"Loading '{filepath}' …")
    pcd = pcg.load_pcd(filepath)
    pcd = pcg.remove_ground(pcd)
    print("Clustering (DBSCAN) — this can take a while on huge scenes …")
    labels, n_clusters, eps = pcg.cluster_dbscan(pcd)
    palette      = pcg.make_palette(n_clusters)
    cluster_meta = build_cluster_meta(pcd, labels, n_clusters)
    print(f"{n_clusters} clusters found (eps={eps:.4f}).")

    print("Opening cluster identification window — shift+click on your "
          "object's points, then press Q to close and continue.")
    candidate_cids = show_labeled_cluster_overview(
        pcd, labels, n_clusters, palette, cluster_meta)
    preselect_cid = candidate_cids[0] if candidate_cids else None

    def on_continue(pcd, labels, n_clusters, cid):
        room_pcd, obj_pcd = split_room_and_object(pcd, labels, cid)
        print(f"Object cluster {cid}: {len(obj_pcd.points):,} pts  |  "
              f"Room: {len(room_pcd.points):,} pts.")

        job = SplatJob()
        # Gaussian fit runs on a background thread; the combined window opens
        # immediately and shows the room while it waits.
        threading.Thread(target=job.run, args=(obj_pcd,), daemon=True).start()
        run_combined_window(room_pcd, job)

    picker_root = tk.Tk()
    ObjectPickerApp(picker_root, pcd, labels, n_clusters, palette,
                    cluster_meta, on_continue, preselect_cid=preselect_cid)
    picker_root.mainloop()


if __name__ == "__main__":
    main()