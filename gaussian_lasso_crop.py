#!/usr/bin/env python3
"""
gaussian_lasso_crop.py

Interactively lasso/polygon-select a region of a 3D Gaussian Splatting PLY
in Open3D, and export ONLY the selected splats to a new PLY file, with all
original Gaussian attributes (SH coefficients, opacity, scale, rotation,
etc.) preserved.

Why not just use o3d.io.read_point_cloud + draw_geometries_with_editing
directly? Two reasons:
  1. Open3D's PointCloud object only understands xyz / rgb / normals. A
     Gaussian splat PLY has many more fields per point, and those would be
     silently dropped if we cropped and saved through Open3D alone. This
     script keeps the full data around and only uses Open3D to pick
     *which indices* you want.
  2. draw_geometries_with_editing's "S" save key opens a file dialog where
     YOU choose the filename/location, and what it saves is Open3D's own
     xyz/rgb-only geometry -- never the full Gaussian data. Relying on that
     save is what caused indices to go missing before. This version instead
     pulls the crop directly out of Open3D's memory with
     get_cropped_geometry(), so there's no intermediate file and no
     filename guessing at all.

Install dependencies:
    pip install open3d plyfile scipy numpy

Usage:
    python gaussian_lasso_crop.py input.ply output.ply

Controls in the Open3D window:
    Y            - (optional) press twice to align view to an axis, makes
                   drawing a clean selection easier
    K            - lock the screen and enter selection mode
    Ctrl + Left-click (dragged out as you click) - draw a polygon (lasso)
                   selection point by point
    (or just click-drag with the left mouse button for a rectangle selection)
    C            - crop: keep the selected area, discard the rest
    F            - unlock / go back to free orbit view
    Q / Esc      - close the window when you're done

IMPORTANT: do NOT press 'S'. You don't need to save anything from inside
the Open3D window -- once you press 'C' to crop and then close the window,
this script pulls the crop out automatically and writes the real,
full-attribute output file for you.

You can press K -> select -> C multiple times if you want to refine the
region before closing the window; only the final cropped state is used.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import open3d as o3d
except ImportError:
    sys.exit("Open3D is required. Install with: pip install open3d")

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    sys.exit("plyfile is required. Install with: pip install plyfile")

try:
    from scipy.spatial import cKDTree
except ImportError:
    sys.exit("scipy is required. Install with: pip install scipy")


SH_C0 = 0.28209479177387814  # constant for degree-0 spherical harmonics


def load_gaussian_ply(path: Path):
    """Load a Gaussian splat PLY, returning the raw PlyData object and the
    structured numpy array of vertex properties (keeps every field)."""
    ply = PlyData.read(str(path))
    if "vertex" not in ply:
        sys.exit("No 'vertex' element found in the PLY file.")
    vertex = ply["vertex"]
    return ply, vertex.data


def preview_colors(data: np.ndarray) -> np.ndarray:
    """Build an RGB color per point for the viewer, using the SH DC term if
    present (this is what the splat 'looks like' at zeroth order). Falls
    back to flat gray if the fields aren't there."""
    names = data.dtype.names
    if all(n in names for n in ("f_dc_0", "f_dc_1", "f_dc_2")):
        r = 0.5 + SH_C0 * data["f_dc_0"]
        g = 0.5 + SH_C0 * data["f_dc_1"]
        b = 0.5 + SH_C0 * data["f_dc_2"]
        colors = np.stack([r, g, b], axis=1)
        return np.clip(colors, 0.0, 1.0).astype(np.float64)
    n = len(data)
    return np.tile(np.array([0.6, 0.6, 0.6]), (n, 1))


def run_lasso_selection(pcd: "o3d.geometry.PointCloud"):
    """Open the Open3D editing window and return the cropped point cloud
    directly from memory (no file save/dialog involved)."""
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Lasso-select the region to KEEP, then press C, then close the window")
    vis.add_geometry(pcd)
    vis.run()  # blocks until the user closes the window
    cropped = vis.get_cropped_geometry()
    vis.destroy_window()
    return cropped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="Path to the input Gaussian PLY")
    parser.add_argument("output", type=Path, help="Path to write the cropped Gaussian PLY")
    args = parser.parse_args()

    print(f"Loading {args.input} ...")
    ply, data = load_gaussian_ply(args.input)
    xyz = np.stack([data["x"], data["y"], data["z"]], axis=1).astype(np.float64)
    n_points = xyz.shape[0]
    print(f"Loaded {n_points:,} splats, fields: {data.dtype.names}")

    colors = preview_colors(data)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print(
        "\nOpen3D window controls:\n"
        "  Y (x2)      : optional, align view to an axis for easier drawing\n"
        "  K           : lock screen / enter selection mode\n"
        "  Ctrl+Left click (repeated) : draw a lasso/polygon selection\n"
        "  Left-drag   : (alternative) rectangle selection\n"
        "  C           : crop -- keep the selection, discard the rest\n"
        "  F           : unlock / back to free orbit view\n"
        "  Q / Esc     : close the window when finished\n"
        "  Do NOT press 'S' -- just close the window after pressing 'C'.\n"
    )

    cropped = run_lasso_selection(pcd)
    cropped_xyz = np.asarray(cropped.points)

    if len(cropped_xyz) == 0:
        sys.exit(
            "\nNo crop was captured -- it looks like the window was closed "
            "without pressing 'K' to select and then 'C' to crop. "
            "Nothing was exported. Run again and make sure to press 'C' "
            "before closing the window."
        )

    if len(cropped_xyz) == n_points:
        print(
            "\nWarning: the crop contains ALL points, same count as the "
            "original file. If you meant to select a smaller region, make "
            "sure you pressed 'C' after drawing your selection, before "
            "closing the window."
        )

    print(f"Selection has {len(cropped_xyz):,} points. Matching back to the original data...")

    # Match selected points back to the original array by nearest neighbor
    # (robust to any float32/float64 rounding introduced along the way).
    tree = cKDTree(xyz)
    dist, idx = tree.query(cropped_xyz, k=1)

    # Sanity check: matches should be essentially exact (crop doesn't move
    # points). Flag anything suspicious instead of silently accepting it.
    bad = dist > 1e-4
    if np.any(bad):
        print(f"Warning: {np.sum(bad):,} of {len(dist):,} matched points had "
              f"a larger-than-expected distance to their nearest original "
              f"point (max dist = {dist.max():.6g}). Proceeding anyway, but "
              f"double check the output if this number is large.")

    matched_indices = np.unique(idx)
    print(f"Matched {len(matched_indices):,} unique original splats.")

    selected_data = data[matched_indices]

    # --- Verify no fields were lost before we write anything ---
    original_fields = list(data.dtype.names)
    selected_fields = list(selected_data.dtype.names)
    if selected_fields != original_fields:
        sys.exit(
            f"Field mismatch detected! Refusing to write a corrupted export.\n"
            f"  original fields: {original_fields}\n"
            f"  selected fields: {selected_fields}"
        )

    new_vertex = PlyElement.describe(selected_data, "vertex")
    PlyData([new_vertex], text=ply.text, byte_order=ply.byte_order).write(str(args.output))

    # --- Re-read the file we just wrote and verify it matches, for real ---
    check = PlyData.read(str(args.output))["vertex"].data
    assert list(check.dtype.names) == original_fields, "Written file is missing fields!"
    assert len(check) == len(matched_indices), "Written file has the wrong point count!"
    for name in original_fields:
        if not np.array_equal(check[name], selected_data[name]):
            sys.exit(f"Verification failed: field '{name}' does not match after writing.")

    print(f"\nDone. Saved cropped Gaussian PLY -> {args.output}")
    print(f"({len(matched_indices):,} / {n_points:,} splats kept, "
          f"{100 * len(matched_indices) / n_points:.1f}%)")
    print(f"Verified all {len(original_fields)} original fields are intact in the output:")
    print(f"  {original_fields}")


if __name__ == "__main__":
    main()