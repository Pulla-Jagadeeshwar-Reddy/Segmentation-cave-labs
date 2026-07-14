"""
Point Cloud Renderer — supports standard AND Gaussian Splat PLY files
----------------------------------------------------------------------
Supports: .ply (standard + 3DGS), .pcd, .xyz, .xyzn, .xyzrgb, .pts

Usage:
    python render_point_cloud.py <path_to_file>
    python render_point_cloud.py   # Opens a file dialog
"""

import sys, os

def install(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

for pkg in ["open3d", "plyfile"]:
    try:
        __import__("open3d" if pkg == "open3d" else "plyfile")
    except ImportError:
        print(f"Installing {pkg} …")
        install(pkg)

import open3d as o3d
import numpy as np
from plyfile import PlyData

SUPPORTED = {".ply", ".pcd", ".xyz", ".xyzn", ".xyzrgb", ".pts"}
SH_C0 = 0.28209479177387814   # zeroth-order spherical harmonic constant


# ── file picker ───────────────────────────────────────────────────────────────

def pick_file():
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk(); root.withdraw()
        path = filedialog.askopenfilename(
            title="Select a point cloud file",
            filetypes=[("Point cloud files", "*.ply *.pcd *.xyz *.xyzn *.xyzrgb *.pts"),
                       ("All files", "*.*")])
        root.destroy()
        return path or None
    except Exception:
        p = input("Enter path to point cloud file: ").strip()
        return p if p else None


# ── Gaussian Splat detection & loading ───────────────────────────────────────

def is_gaussian_splat(filepath: str) -> bool:
    """Check if a PLY file is a 3D Gaussian Splatting file (has f_dc_0 property)."""
    if not filepath.lower().endswith(".ply"):
        return False
    try:
        ply = PlyData.read(filepath)
        props = {p.name for p in ply['vertex'].properties}
        return "f_dc_0" in props
    except Exception:
        return False


def load_gaussian_splat(filepath: str) -> o3d.geometry.PointCloud:
    """
    Load a 3DGS PLY and convert Spherical Harmonics DC coefficients → RGB.

    The DC (degree-0) SH coefficient encodes the view-independent base colour.
    Conversion: RGB = clip(0.5 + SH_C0 * f_dc, 0, 1)
    """
    print("  Detected 3D Gaussian Splatting PLY — converting SH coefficients to RGB …")
    ply  = PlyData.read(filepath)
    v    = ply['vertex']

    xyz = np.stack([np.array(v['x']), np.array(v['y']), np.array(v['z'])], axis=1)

    # SH DC → linear RGB
    r = np.clip(0.5 + SH_C0 * np.array(v['f_dc_0']), 0, 1)
    g = np.clip(0.5 + SH_C0 * np.array(v['f_dc_1']), 0, 1)
    b = np.clip(0.5 + SH_C0 * np.array(v['f_dc_2']), 0, 1)
    colors = np.stack([r, g, b], axis=1)

    # Filter by opacity if present (removes near-invisible splats)
    if 'opacity' in {p.name for p in v.properties}:
        raw_opacity  = np.array(v['opacity'])
        opacity      = 1.0 / (1.0 + np.exp(-raw_opacity))   # sigmoid
        mask         = opacity > 0.1
        pct_kept     = mask.sum() / len(mask) * 100
        print(f"  Opacity filter: keeping {mask.sum():,} / {len(mask):,} points ({pct_kept:.1f}%)")
        xyz    = xyz[mask]
        colors = colors[mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(f"  RGB mean after conversion — R:{r.mean():.3f}  G:{g.mean():.3f}  B:{b.mean():.3f}")
    return pcd


# ── standard load ─────────────────────────────────────────────────────────────

def load_standard(filepath: str) -> o3d.geometry.PointCloud:
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SUPPORTED:
        print(f"  Warning: '{ext}' not a known extension. Attempting anyway …")
    pcd = o3d.io.read_point_cloud(filepath)
    if not pcd.has_points():
        raise ValueError(f"No points loaded from '{filepath}'.")
    return pcd


def fix_standard_colors(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """Fix integer-range or very dark colors in standard point clouds."""
    if not pcd.has_colors():
        return pcd
    colors = np.asarray(pcd.colors).copy().astype(np.float64)
    print(f"  Color range: min={colors.min():.4f}  max={colors.max():.4f}  mean={colors.mean():.4f}")
    if colors.max() > 1.5:
        print("  → Integer colors (0-255) detected — dividing by 255")
        colors /= 255.0
    if colors.mean() < 0.25:
        print("  → Dark colors — applying histogram stretch + gamma")
        for c in range(3):
            lo, hi = np.percentile(colors[:, c], 1), np.percentile(colors[:, c], 99)
            if hi - lo > 1e-6:
                colors[:, c] = (colors[:, c] - lo) / (hi - lo)
        colors = np.clip(colors, 0, 1) ** 0.5
    pcd.colors = o3d.utility.Vector3dVector(np.clip(colors, 0, 1))
    return pcd


def colorize_by_height(pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    pts   = np.asarray(pcd.points)
    z     = pts[:, 2]
    z_n   = (z - z.min()) / ((z.max() - z.min()) + 1e-9)
    cols  = np.zeros((len(z_n), 3))
    cols[:, 0] = np.clip(1.5 - np.abs(z_n - 0.75) * 4, 0, 1)
    cols[:, 1] = np.clip(1.5 - np.abs(z_n - 0.50) * 4, 0, 1)
    cols[:, 2] = np.clip(1.5 - np.abs(z_n - 0.25) * 4, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(cols)
    return pcd


# ── scene helpers ─────────────────────────────────────────────────────────────

def get_scene_scale(pcd):
    return float(np.linalg.norm(pcd.get_axis_aligned_bounding_box().get_extent()))

def auto_point_size(scale):
    if scale < 1.0:  return 3.0
    if scale < 5.0:  return 2.0
    if scale < 20.0: return 1.5
    return 1.0

def print_info(pcd, filepath, scale):
    ext = pcd.get_axis_aligned_bounding_box().get_extent()
    print(f"\n  File        : {os.path.basename(filepath)}")
    print(f"  Points      : {len(np.asarray(pcd.points)):,}")
    print(f"  Bounding box: {ext[0]:.2f} × {ext[1]:.2f} × {ext[2]:.2f}")
    print(f"  Scene scale : {scale:.2f}\n")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    filepath = sys.argv[1] if len(sys.argv) > 1 else pick_file()
    if not filepath:
        print("No file selected. Exiting."); sys.exit(0)
    if not os.path.isfile(filepath):
        print(f"Error: file not found – '{filepath}'"); sys.exit(1)

    print(f"\nLoading '{filepath}' …")

    # Load
    if is_gaussian_splat(filepath):
        pcd = load_gaussian_splat(filepath)
    else:
        pcd = load_standard(filepath)
        if not pcd.has_colors():
            print("  No color data — applying height colormap.")
            pcd = colorize_by_height(pcd)
        else:
            pcd = fix_standard_colors(pcd)

    scale = get_scene_scale(pcd)

    # Normals
    if not pcd.has_normals():
        print("Estimating normals …")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=max(scale * 0.01, 0.05), max_nn=30))

    print_info(pcd, filepath, scale)

    # Viewer
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"Point Cloud – {os.path.basename(filepath)}",
        width=1280, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color      = np.array([0.08, 0.08, 0.12])
    opt.point_size            = auto_point_size(scale)
    opt.show_coordinate_frame = True
    opt.light_on              = True

    print(f"Point size: {opt.point_size}")
    vis.reset_view_point(True)
    print("Viewer open. Close the window or press Q / Esc to quit.\n")
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    main()