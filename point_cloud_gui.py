"""
Point Cloud Cluster GUI  —  v3 (tkinter + Open3D visualizer)
-------------------------------------------------------------
Uses tkinter for the control panel and Open3D's stable draw_geometries
visualizer for 3D rendering — avoids Filament/WGL black screen on Windows.

Controls (in 3D window):
  Left-drag      Orbit
  Right-drag     Zoom
  Middle-drag    Pan
  Q / Escape     Close viewer

Requires:
    pip install open3d plyfile numpy
"""

import sys, os, threading, tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np

def _ensure(pkg):
    import importlib, subprocess
    try:
        importlib.import_module(pkg)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

_ensure("open3d"); _ensure("plyfile"); _ensure("numpy")

import open3d as o3d
from plyfile import PlyData, PlyElement

SH_C0 = 0.28209479177387814


# ══════════════════════════════════════════════════════════════════════════════
# Loaders
# ══════════════════════════════════════════════════════════════════════════════

def _is_gaussian_splat(fp):
    if not fp.lower().endswith(".ply"):
        return False
    try:
        props = {p.name for p in PlyData.read(fp)["vertex"].properties}
        return "f_dc_0" in props
    except Exception:
        return False

def _load_gaussian(fp):
    v   = PlyData.read(fp)["vertex"]
    xyz = np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], 1)
    r   = np.clip(0.5 + SH_C0 * np.array(v["f_dc_0"]), 0, 1)
    g   = np.clip(0.5 + SH_C0 * np.array(v["f_dc_1"]), 0, 1)
    b   = np.clip(0.5 + SH_C0 * np.array(v["f_dc_2"]), 0, 1)
    col = np.stack([r, g, b], 1)
    if "opacity" in {p.name for p in v.properties}:
        op  = 1.0 / (1.0 + np.exp(-np.array(v["opacity"])))
        msk = op > 0.1
        xyz, col = xyz[msk], col[msk]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(col)
    return pcd

def _fix_colors(pcd):
    if not pcd.has_colors():
        return pcd
    c = np.asarray(pcd.colors).copy().astype(np.float64)
    if c.max() > 1.5:
        c /= 255.0
    if c.mean() < 0.25:
        for ch in range(3):
            lo, hi = np.percentile(c[:, ch], 1), np.percentile(c[:, ch], 99)
            if hi - lo > 1e-6:
                c[:, ch] = (c[:, ch] - lo) / (hi - lo)
        c = np.clip(c, 0, 1) ** 0.5
    pcd.colors = o3d.utility.Vector3dVector(np.clip(c, 0, 1))
    return pcd

def _colorize_height(pcd):
    z  = np.asarray(pcd.points)[:, 2]
    zn = (z - z.min()) / (z.ptp() + 1e-9)
    c  = np.zeros((len(zn), 3))
    c[:, 0] = np.clip(1.5 - np.abs(zn - 0.75) * 4, 0, 1)
    c[:, 1] = np.clip(1.5 - np.abs(zn - 0.50) * 4, 0, 1)
    c[:, 2] = np.clip(1.5 - np.abs(zn - 0.25) * 4, 0, 1)
    pcd.colors = o3d.utility.Vector3dVector(c)
    return pcd

def load_pcd(fp):
    if _is_gaussian_splat(fp):
        return _load_gaussian(fp)
    pcd = o3d.io.read_point_cloud(fp)
    if not pcd.has_points():
        raise ValueError(f"No points in '{fp}'")
    return _fix_colors(pcd) if pcd.has_colors() else _colorize_height(pcd)


# ══════════════════════════════════════════════════════════════════════════════
# Clustering
# ══════════════════════════════════════════════════════════════════════════════

def remove_ground(pcd, thresh=0.1):
    try:
        _, inliers = pcd.segment_plane(thresh, 3, 1000)
        return pcd.select_by_index(inliers, invert=True)
    except Exception:
        return pcd

def _auto_eps(pcd, k=25):
    pts  = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    step = max(1, len(pts) // 4000)
    ds   = [np.sqrt(tree.search_knn_vector_3d(pts[i], k + 1)[2][-1])
            for i in range(0, len(pts), step)]
    return float(np.percentile(ds, 95))

def cluster_dbscan(pcd, eps=None, min_pts=60):
    if eps is None:
        eps = _auto_eps(pcd)
    labels = np.array(
        pcd.cluster_dbscan(eps=eps, min_points=min_pts, print_progress=False))
    return labels, int(labels.max()) + 1, float(eps)


# ══════════════════════════════════════════════════════════════════════════════
# Palette
# ══════════════════════════════════════════════════════════════════════════════

def make_palette(n):
    phi  = 0.618033988749895
    cols = []
    for i in range(n):
        h  = (i * phi) % 1.0
        hi = int(h * 6) % 6
        f  = h * 6 - int(h * 6)
        p, q, t = 0.9*(1-.75), 0.9*(1-.75*f), 0.9*(1-.75*(1-f))
        v  = 0.9
        cols.append([(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][hi])
    return np.array(cols)


# ══════════════════════════════════════════════════════════════════════════════
# Export
# ══════════════════════════════════════════════════════════════════════════════

def export_ply(pcd, indices, out_path):
    pts    = np.asarray(pcd.points)[indices]
    colors = (np.asarray(pcd.colors)[indices]
              if pcd.has_colors() else np.ones((len(indices), 3)))
    v = np.zeros(len(pts), dtype=[
        ("x","f4"),("y","f4"),("z","f4"),
        ("red","u1"),("green","u1"),("blue","u1")])
    v["x"], v["y"], v["z"] = pts[:,0], pts[:,1], pts[:,2]
    v["red"]   = (colors[:,0]*255).clip(0,255).astype(np.uint8)
    v["green"] = (colors[:,1]*255).clip(0,255).astype(np.uint8)
    v["blue"]  = (colors[:,2]*255).clip(0,255).astype(np.uint8)
    PlyData([PlyElement.describe(v, "vertex")]).write(out_path)


# ══════════════════════════════════════════════════════════════════════════════
# Build colored point cloud from cluster state
# ══════════════════════════════════════════════════════════════════════════════

def build_display_pcd(pcd, labels, n_clusters, palette, selected, isolate):
    pts  = np.asarray(pcd.points)
    cols = np.zeros((len(pts), 3))

    # Noise
    noise = labels == -1
    if not isolate:
        cols[noise] = [0.25, 0.25, 0.25]

    for cid in range(n_clusters):
        mask = labels == cid
        if isolate and cid not in selected:
            continue
        col = palette[cid]
        if cid in selected:
            c = [min(col[0]+0.2,1), min(col[1]+0.2,1), min(col[2]+0.2,1)]
        else:
            c = list(col)
        cols[mask] = c

    keep = np.zeros(len(pts), dtype=bool)
    if not isolate:
        keep |= noise
    for cid in range(n_clusters):
        if isolate and cid not in selected:
            continue
        keep |= (labels == cid)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts[keep])
    out.colors = o3d.utility.Vector3dVector(cols[keep])
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Main App
# ══════════════════════════════════════════════════════════════════════════════

class ClusterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Cloud Cluster GUI")
        self.root.configure(bg="#14151f")
        self.root.geometry("360x780")
        self.root.resizable(False, True)

        self.filepath     = None
        self.pcd          = None
        self.labels       = None
        self.n_clusters   = 0
        self.palette      = None
        self.cluster_meta = {}
        self.selected     = set()
        self._vis_thread  = None

        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        BG, FG, ACC = "#14151f", "#e0e4f5", "#6a7fc1"
        BTN = {"bg":"#252736","fg":FG,"activebackground":"#353850",
               "activeforeground":FG,"relief":"flat","bd":0,
               "padx":10,"pady":5,"cursor":"hand2"}
        LBL = {"bg":BG,"fg":FG}
        SEC = {"bg":BG,"fg":ACC,"font":("Segoe UI",8,"bold")}

        f = tk.Frame(self.root, bg=BG, padx=14, pady=10)
        f.pack(fill="both", expand=True)

        def sec(text):
            tk.Label(f, text=text, **SEC).pack(anchor="w", pady=(10,2))

        def btn(text, cmd, color=None):
            b = tk.Button(f, text=text, command=cmd, **BTN)
            if color:
                b.configure(bg=color)
            b.pack(fill="x", pady=2)
            return b

        def stat_row(label):
            row = tk.Frame(f, bg=BG)
            row.pack(fill="x")
            tk.Label(row, text=label+":", bg=BG, fg="#7a80a0",
                     font=("Segoe UI",9)).pack(side="left")
            v = tk.Label(row, text="—", bg=BG, fg=FG, font=("Segoe UI",9))
            v.pack(side="right")
            return v

        # FILE
        sec("FILE")
        btn("Open File…", self._on_open)
        btn("Re-cluster…", self._on_recluster)

        # SCENE STATS
        sec("SCENE")
        self.lbl_file  = stat_row("File")
        self.lbl_pts   = stat_row("Points")
        self.lbl_cls   = stat_row("Clusters")
        self.lbl_sel   = stat_row("Selected")
        self.lbl_noise = stat_row("Noise pts")
        self.lbl_eps   = stat_row("eps used")

        # SELECTION
        sec("SELECTION")
        row = tk.Frame(f, bg=BG); row.pack(fill="x", pady=2)
        for text, cmd in [("All",self._select_all),("None",self._deselect_all),("Invert",self._invert_sel)]:
            tk.Button(row, text=text, command=cmd, **BTN).pack(side="left", expand=True, fill="x", padx=2)

        # CLUSTERS LIST
        sec("CLUSTERS")
        lf = tk.Frame(f, bg=BG); lf.pack(fill="x")
        sb = tk.Scrollbar(lf, orient="vertical")
        self.clist = tk.Listbox(lf, bg="#1e1f2e", fg=FG, selectbackground="#353850",
                                 selectforeground=FG, relief="flat", bd=0,
                                 font=("Consolas",9), height=12,
                                 yscrollcommand=sb.set, exportselection=False)
        sb.config(command=self.clist.yview)
        self.clist.pack(side="left", fill="x", expand=True)
        sb.pack(side="right", fill="y")
        self.clist.bind("<<ListboxSelect>>", self._on_list_sel)

        # LABEL
        sec("LABEL")
        self.label_var = tk.StringVar()
        tk.Entry(f, textvariable=self.label_var, bg="#1e1f2e", fg=FG,
                 insertbackground=FG, relief="flat", font=("Segoe UI",10)).pack(fill="x", pady=2)
        btn("Apply to selected", self._apply_label)

        # VIEW
        sec("VIEW")
        self.isolate_var = tk.BooleanVar()
        tk.Checkbutton(f, text="Show only selected", variable=self.isolate_var,
                       bg=BG, fg=FG, selectcolor="#252736",
                       activebackground=BG, activeforeground=FG,
                       command=self._refresh_vis).pack(anchor="w")

        sec("VISUALIZE")
        btn("▶  Open 3D Viewer", self._open_viewer, color="#1e3a5f")

        # EXPORT
        sec("EXPORT")
        btn("Export selected clusters", self._export_sel)
        btn("Export all clusters",      self._export_all)
        self.lbl_export = tk.Label(f, text="", bg=BG, fg="#3dda8a",
                                    font=("Segoe UI",9))
        self.lbl_export.pack(anchor="w", pady=2)

        # STATUS BAR
        self.status = tk.Label(self.root, text="Ready", bg="#0d0e17", fg="#555a7a",
                                font=("Segoe UI",8), anchor="w", padx=8)
        self.status.pack(fill="x", side="bottom")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _on_open(self):
        fp = filedialog.askopenfilename(
            title="Open point cloud",
            filetypes=[("Point clouds","*.ply *.pcd *.xyz *.xyzrgb *.pts"),
                       ("All files","*.*")])
        if fp:
            self._load(fp)

    def _load(self, fp, eps=None, min_pts=50, ground_thresh=0.02, skip_ground=False):
        self.filepath = fp
        self._set_status("Loading…")
        self.lbl_file.config(text="Loading…")

        def worker():
            try:
                pcd = load_pcd(fp)
                if not skip_ground:
                    pcd = remove_ground(pcd, thresh=ground_thresh)
                labels, n, eps_used = cluster_dbscan(pcd, eps=eps, min_pts=min_pts)
                self.root.after(0, lambda: self._on_loaded(pcd, labels, n, eps_used))
            except Exception as exc:
                self.root.after(0, lambda: messagebox.showerror("Load error", str(exc)))

        threading.Thread(target=worker, daemon=True).start()

    def _on_loaded(self, pcd, labels, n_clusters, eps):
        self.pcd        = pcd
        self.labels     = labels
        self.n_clusters = n_clusters
        self.palette    = make_palette(n_clusters)
        self.selected   = set()
        self._build_meta()
        self._populate_list()
        self._update_stats(eps)
        self._set_status(f"Loaded {n_clusters} clusters. Click 'Open 3D Viewer' to view.")

    def _build_meta(self):
        pts = np.asarray(self.pcd.points)
        self.cluster_meta = {}
        for cid in range(self.n_clusters):
            mask = self.labels == cid
            idx  = np.where(mask)[0]
            self.cluster_meta[cid] = {
                "label"   : f"cluster_{cid}",
                "n_points": int(mask.sum()),
                "indices" : idx,
                "centroid": pts[mask].mean(axis=0),
            }

    # ── Visualizer ────────────────────────────────────────────────────────────

    def _open_viewer(self):
        if self.pcd is None:
            messagebox.showinfo("No data", "Open a point cloud file first.")
            return
        display = build_display_pcd(
            self.pcd, self.labels, self.n_clusters,
            self.palette, self.selected, self.isolate_var.get())
        threading.Thread(target=self._run_viewer, args=(display,), daemon=True).start()

    def _run_viewer(self, display):
        self._set_status("3D viewer open…")
        o3d.visualization.draw_geometries(
            [display],
            window_name="Point Cloud Viewer",
            width=1100, height=750,
            point_show_normal=False)
        self._set_status("3D viewer closed.")

    def _refresh_vis(self):
        # Just updates status; user reopens viewer to see changes
        self._set_status("Settings changed. Re-open viewer to update.")

    # ── List / Stats ──────────────────────────────────────────────────────────

    def _sorted_cids(self):
        return sorted(self.cluster_meta, key=lambda c: -self.cluster_meta[c]["n_points"])

    def _populate_list(self):
        self.clist.delete(0, "end")
        for cid in self._sorted_cids():
            m   = self.cluster_meta[cid]
            dot = "●" if cid in self.selected else "○"
            self.clist.insert("end",
                f"{dot} {cid:>3}  {m['label']:<18}  {m['n_points']:>7,}")
        self._color_list()

    def _color_list(self):
        for i, cid in enumerate(self._sorted_cids()):
            col = self.palette[cid] if self.palette is not None else (0.5,0.5,0.5)
            hex_col = "#{:02x}{:02x}{:02x}".format(
                int(col[0]*200+55), int(col[1]*200+55), int(col[2]*200+55))
            self.clist.itemconfig(i, fg=hex_col if cid not in self.selected else "#ffffff")

    def _update_stats(self, eps=None):
        if self.pcd is None: return
        noise = int((self.labels == -1).sum()) if self.labels is not None else 0
        self.lbl_file.config(text=os.path.basename(self.filepath or "—")[:28])
        self.lbl_pts.config(text=f"{len(np.asarray(self.pcd.points)):,}")
        self.lbl_cls.config(text=str(self.n_clusters))
        self.lbl_sel.config(text=f"{len(self.selected)} / {self.n_clusters}")
        self.lbl_noise.config(text=f"{noise:,}")
        if eps is not None:
            self.lbl_eps.config(text=f"{eps:.4f}")

    def _set_status(self, msg):
        self.status.config(text=msg)

    # ── Selection ─────────────────────────────────────────────────────────────

    def _on_list_sel(self, event):
        sel = self.clist.curselection()
        if not sel: return
        cids = self._sorted_cids()
        cid  = cids[sel[0]]
        if cid in self.selected:
            self.selected.discard(cid)
        else:
            self.selected.add(cid)
        self._populate_list()
        self._update_stats()

    def _select_all(self):
        self.selected = set(self.cluster_meta.keys())
        self._populate_list(); self._update_stats()

    def _deselect_all(self):
        self.selected.clear()
        self._populate_list(); self._update_stats()

    def _invert_sel(self):
        self.selected = set(self.cluster_meta.keys()) - self.selected
        self._populate_list(); self._update_stats()

    # ── Label ─────────────────────────────────────────────────────────────────

    def _apply_label(self):
        if not self.selected:
            messagebox.showinfo("No selection", "Select at least one cluster first.")
            return
        text = self.label_var.get().strip()
        if not text:
            messagebox.showinfo("Empty label", "Type a label before applying.")
            return
        for i, cid in enumerate(sorted(self.selected)):
            suffix = f"_{i}" if len(self.selected) > 1 else ""
            self.cluster_meta[cid]["label"] = text + suffix
        self.label_var.set("")
        self._populate_list()

    # ── Re-cluster ────────────────────────────────────────────────────────────

    def _on_recluster(self):
        if not self.filepath:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("Re-cluster")
        dlg.configure(bg="#14151f")
        dlg.resizable(False, False)
        dlg.grab_set()

        f = tk.Frame(dlg, bg="#14151f", padx=16, pady=12)
        f.pack()

        LBL = {"bg":"#14151f","fg":"#e0e4f5","font":("Segoe UI",9)}

        def row(label, default):
            r = tk.Frame(f, bg="#14151f"); r.pack(fill="x", pady=3)
            tk.Label(r, text=label, **LBL).pack(side="left")
            v = tk.StringVar(value=str(default))
            tk.Entry(r, textvariable=v, bg="#1e1f2e", fg="#e0e4f5",
                     insertbackground="#e0e4f5", width=10, relief="flat").pack(side="right")
            return v

        eps_v   = row("eps (0 = auto)", 0)
        minpt_v = row("min points",    50)
        gth_v   = row("ground thresh", 0.02)
        gnd_v   = tk.BooleanVar(value=True)
        tk.Checkbutton(f, text="Remove ground plane", variable=gnd_v,
                       bg="#14151f", fg="#e0e4f5", selectcolor="#252736",
                       activebackground="#14151f").pack(anchor="w", pady=4)

        def do_run():
            try:
                eps = float(eps_v.get()) or None
                mpt = int(minpt_v.get())
                gth = float(gth_v.get())
            except ValueError:
                messagebox.showerror("Invalid input", "Check your values.", parent=dlg)
                return
            dlg.destroy()
            self._load(self.filepath, eps=eps, min_pts=mpt,
                       ground_thresh=gth, skip_ground=not gnd_v.get())

        br = tk.Frame(f, bg="#14151f"); br.pack(fill="x", pady=(8,0))
        tk.Button(br, text="Cancel", command=dlg.destroy,
                  bg="#252736", fg="#e0e4f5", relief="flat", padx=10, pady=4).pack(side="left")
        tk.Button(br, text="Run", command=do_run,
                  bg="#1e3a5f", fg="#e0e4f5", relief="flat", padx=10, pady=4).pack(side="right")

    # ── Export ────────────────────────────────────────────────────────────────

    def _do_export(self, cids):
        if not cids:
            messagebox.showinfo("Nothing to export", "Select at least one cluster.")
            return
        folder = filedialog.askdirectory(title="Choose export folder")
        if not folder: return
        base  = os.path.splitext(os.path.basename(self.filepath))[0]
        count = 0
        for cid in cids:
            m     = self.cluster_meta[cid]
            label = m["label"].replace(" ", "_")
            export_ply(self.pcd, m["indices"],
                       os.path.join(folder, f"{base}_{label}.ply"))
            count += 1
        self.lbl_export.config(text=f"✓ {count} file(s) saved")
        messagebox.showinfo("Export complete", f"Saved {count} PLY file(s) to:\n{folder}")

    def _export_sel(self): self._do_export(list(self.selected))
    def _export_all(self): self._do_export(list(self.cluster_meta.keys()))


# ══════════════════════════════════════════════════════════════════════════════
# Entry
# ══════════════════════════════════════════════════════════════════════════════

def main():
    root = tk.Tk()
    app  = ClusterApp(root)
    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        root.after(200, lambda: app._load(sys.argv[1]))
    root.mainloop()

if __name__ == "__main__":
    main()