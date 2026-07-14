"""
3D Gaussian Splatting renderer from PLY point cloud.

Controls:
  Left-drag        → orbit (around current pivot)
  Right-drag        → pan
  Scroll            → zoom
  WASD              → orbit (keyboard fallback)
  QE                → zoom in/out
  C, then left-click → pick a new orbit pivot at the clicked Gaussian
  R                 → reset pivot/camera to the point-cloud centroid
  ESC               → quit

The centroid of all Gaussians is rendered as a bright yellow marker.
"""

import os, sys, math, ctypes, argparse
import numpy as np

# ── Force NVIDIA discrete GPU ─────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"]      = "0"
os.environ["__NV_PRIME_RENDER_OFFLOAD"] = "1"
os.environ["__GLX_VENDOR_LIBRARY_NAME"] = "nvidia"
try:    ctypes.windll.LoadLibrary("nvapi64.dll")
except: pass

import open3d as o3d

try:
    import torch
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        torch.cuda.set_device(0)
        print(f"[CUDA] {torch.cuda.get_device_name(0)}")
    else:
        DEVICE = torch.device("cpu")
        print("[CUDA] not available – CPU fallback")
    TORCH = True
except ImportError:
    TORCH  = False
    DEVICE = None
    print("[CUDA] PyTorch not installed – CPU fallback")

import moderngl
import moderngl_window as mglw


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Gaussian estimation via PCA
# ═══════════════════════════════════════════════════════════════════════════════

def estimate_gaussians(points: np.ndarray, colors: np.ndarray,
                       k: int = 16, scale: float = 1.0) -> dict:
    N   = len(points)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    kd  = o3d.geometry.KDTreeFlann(pcd)

    cov3d    = np.zeros((N, 6), dtype=np.float32)
    normal   = np.zeros((N, 3), dtype=np.float32)
    centroid = points.mean(axis=0)

    for i in range(N):
        _, idx, _ = kd.search_knn_vector_3d(points[i], k + 1)
        nbrs = points[np.asarray(idx[1:], np.int32)]
        diff = nbrs - points[i]
        C    = (diff.T @ diff) / max(len(nbrs) - 1, 1)

        # Clamp eigenvalues: min axis = 1% of max axis (prevents blobs / needles)
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 0.0, None)
        mx   = vals.max() if vals.max() > 0 else 1e-6
        vals = np.maximum(vals, mx * 0.01)

        # Surface normal = eigenvector of the SMALLEST eigenvalue (the local
        # "flat" direction of the point neighbourhood). eigh returns
        # eigenvalues ascending, so column 0 is it. Orient it outward from
        # the object's centroid so neighbouring splats don't shade with
        # randomly flipped normals (PCA normals have no inherent sign).
        n = vecs[:, 0]
        if np.dot(n, points[i] - centroid) < 0:
            n = -n
        normal[i] = n

        C    = (vecs * vals) @ vecs.T * (scale ** 2)
        cov3d[i] = [C[0,0], C[0,1], C[0,2], C[1,1], C[1,2], C[2,2]]

    return dict(mean    = points.astype(np.float32),
                color   = colors.astype(np.float32),
                opacity = np.full(N, 0.85, np.float32),
                cov3d   = cov3d,
                normal  = normal)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Camera
# ═══════════════════════════════════════════════════════════════════════════════

class OrbitCamera:
    """
    Orbit camera with a *pivot* point that rotation happens around, and a
    separate *target* that pan moves. Initially pivot == target (point-cloud
    centroid). When the user picks a new rotation axis point (C + click),
    `pivot` is updated but the camera's current world position is preserved
    by recomputing (theta, phi, radius) relative to the new pivot so there
    is no visual jump.
    """
    def __init__(self, centre, radius, fov=50.0, aspect=16/9,
                 near=0.001, far=10000.0):
        self.pivot  = np.array(centre, np.float64)   # rotation centre
        self.radius = float(radius)
        self.theta  = math.pi / 4
        self.phi    = math.pi / 3
        self.fov    = fov
        self.aspect = aspect
        self.near   = near
        self.far    = far

    @property
    def position(self):
        sp, cp = math.sin(self.phi),   math.cos(self.phi)
        st, ct = math.sin(self.theta), math.cos(self.theta)
        return self.pivot + self.radius * np.array([sp*ct, cp, sp*st])

    def view(self) -> np.ndarray:
        eye = self.position
        f   = self.pivot - eye;         f  /= np.linalg.norm(f)
        r   = np.cross(f, [0.,1.,0.]); r  /= np.linalg.norm(r)
        u   = np.cross(r, f)
        M   = np.eye(4, dtype=np.float32)
        M[0,:3]=r;  M[0,3]=-float(r@eye)
        M[1,:3]=u;  M[1,3]=-float(u@eye)
        M[2,:3]=-f; M[2,3]= float(f@eye)
        return M

    def proj(self) -> np.ndarray:
        t = math.tan(math.radians(self.fov)/2)
        n, f = self.near, self.far
        P = np.zeros((4,4), dtype=np.float32)
        P[0,0] = 1/(self.aspect*t)
        P[1,1] = 1/t
        P[2,2] = -(f+n)/(f-n)
        P[2,3] = -2*f*n/(f-n)
        P[3,2] = -1.0
        return P

    def orbit(self, dx: float, dy: float):
        """Rotate around the current pivot."""
        self.theta -= dx * 0.006
        self.phi    = float(np.clip(self.phi - dy*0.006, 0.04, math.pi-0.04))

    def pan(self, dx: float, dy: float):
        """
        Pan moves both the pivot and the implicit camera position together,
        i.e. it slides the whole view sideways without changing what you're
        looking at relative to the scene's depth.
        """
        V = self.view()
        s = self.radius * 0.0012
        self.pivot -= V[0,:3].astype(np.float64)*dx*s
        self.pivot += V[1,:3].astype(np.float64)*dy*s

    def zoom(self, d: float):
        self.radius = max(0.001, self.radius*(1.0 - d*0.12))

    def set_pivot_preserve_view(self, new_pivot: np.ndarray):
        """
        Change the rotation pivot to `new_pivot` while keeping the camera's
        current world-space eye position fixed, so the view doesn't jump.
        Recomputes (theta, phi, radius) so that `position` stays the same.
        """
        eye = self.position                      # current eye, before pivot change
        self.pivot = np.array(new_pivot, np.float64)

        offset = eye - self.pivot
        radius = float(np.linalg.norm(offset))
        if radius < 1e-9:
            # Degenerate: picked point coincides with camera, keep old radius
            return
        self.radius = radius

        # Recover (theta, phi) from offset = radius*[sin(phi)cos(theta), cos(phi), sin(phi)sin(theta)]
        cp = np.clip(offset[1] / radius, -1.0, 1.0)
        self.phi = math.acos(cp)
        sp = math.sin(self.phi)
        if abs(sp) < 1e-9:
            self.theta = 0.0
        else:
            self.theta = math.atan2(offset[2] / (radius*sp), offset[0] / (radius*sp))


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  EWA covariance projection  (pixel space, CUDA or NumPy)
# ═══════════════════════════════════════════════════════════════════════════════

def _build_cov2d(means, cov3d, V, P, W, H, use_cuda=False):
    N  = len(means)
    R  = V[:3,:3]
    fx = float(P[0,0]) * W * 0.5
    fy = float(P[1,1]) * H * 0.5

    if use_cuda:
        import torch
        dev  = DEVICE
        t_m  = torch.tensor(means,  dtype=torch.float64, device=dev)
        ones = torch.ones((N,1),    dtype=torch.float64, device=dev)
        Vt   = torch.tensor(V,      dtype=torch.float64, device=dev)
        Rt   = torch.tensor(R,      dtype=torch.float64, device=dev)

        cam  = (Vt @ torch.cat([t_m, ones], 1).T).T[:,:3]
        tz   = cam[:,2].clone()
        bad  = tz >= -1e-4
        tz[bad] = -1e-4
        itz  = 1.0/tz;  itz2 = itz*itz
        z0   = torch.zeros(N, dtype=torch.float64, device=dev)

        J0 = torch.stack([ fx*itz, z0,      -fx*cam[:,0]*itz2 ], 1)
        J1 = torch.stack([ z0,     fy*itz,  -fy*cam[:,1]*itz2 ], 1)

        tc = torch.tensor(cov3d, dtype=torch.float64, device=dev)
        Sg = torch.stack([
            torch.stack([tc[:,0],tc[:,1],tc[:,2]],1),
            torch.stack([tc[:,1],tc[:,3],tc[:,4]],1),
            torch.stack([tc[:,2],tc[:,4],tc[:,5]],1),
        ],1)
        RS  = torch.einsum("ij,njk->nik", Rt, Sg)
        M   = torch.einsum("nij,kj->nik", RS, Rt)

        s00 = (J0 * torch.einsum("nij,nj->ni", M, J0)).sum(1) + 0.3
        s01 = (J0 * torch.einsum("nij,nj->ni", M, J1)).sum(1)
        s11 = (J1 * torch.einsum("nij,nj->ni", M, J1)).sum(1) + 0.3

        out = torch.stack([s00,s01,s11],1).float()
        out[bad] = 0.0
        return out.cpu().numpy()

    else:
        R   = R.astype(np.float64)
        ones = np.ones((N,1), np.float64)
        cam  = (V.astype(np.float64) @ np.hstack([means,ones]).T).T[:,:3]
        tz   = cam[:,2].copy()
        bad  = tz >= -1e-4
        tz[bad] = -1e-4
        itz  = 1/tz;  itz2 = itz*itz

        J0 = np.stack([ fx*itz,        np.zeros(N), -fx*cam[:,0]*itz2 ], 1)
        J1 = np.stack([ np.zeros(N),   fy*itz,      -fy*cam[:,1]*itz2 ], 1)

        c  = cov3d.astype(np.float64)
        Sg = np.stack([
            np.stack([c[:,0],c[:,1],c[:,2]],1),
            np.stack([c[:,1],c[:,3],c[:,4]],1),
            np.stack([c[:,2],c[:,4],c[:,5]],1),
        ],1)
        RS  = np.einsum("ij,njk->nik", R, Sg)
        M   = np.einsum("nij,kj->nik", RS, R)

        s00 = (J0 * np.einsum("nij,nj->ni", M, J0)).sum(1) + 0.3
        s01 = (J0 * np.einsum("nij,nj->ni", M, J1)).sum(1)
        s11 = (J1 * np.einsum("nij,nj->ni", M, J1)).sum(1) + 0.3

        out = np.stack([s00,s01,s11],1).astype(np.float32)
        out[bad] = 0.0
        return out


def project_cov2d(means, cov3d, V, P, W, H):
    return _build_cov2d(means, cov3d, V, P, W, H,
                        use_cuda=TORCH and DEVICE.type=="cuda")


def depth_sort(means, V):
    if TORCH and DEVICE.type == "cuda":
        import torch
        t = torch.tensor(means,       dtype=torch.float32, device=DEVICE)
        r = torch.tensor(V[2:3,:3],   dtype=torch.float32, device=DEVICE)
        return torch.argsort((r @ t.T)[0]).cpu().numpy()
    return np.argsort((V[2:3,:3] @ means.T)[0])


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  GLSL — pixel-space EWA Gaussian splat
# ═══════════════════════════════════════════════════════════════════════════════

VERT = """
#version 330 core

in vec2  a_quad;        // billboard corner  (-1..+1, -1..+1)

in vec3  a_mean;
in vec3  a_color;
in float a_opacity;
in vec3  a_cov2d;       // [s00, s01, s11]  pixel² covariance
in vec3  a_normal;      // world-space surface normal, for lighting

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport;

// Passed to fragment for per-pixel Gaussian evaluation + shading
out vec2  v_pix_off;    // pixel offset from splat centre (at this corner)
out vec3  v_color;
out float v_opacity;
out vec3  v_cov2d;      // forwarded as-is
out vec3  v_world_pos;
out vec3  v_normal;

void main() {
    vec4 cam4 = u_view * vec4(a_mean, 1.0);
    vec4 clip = u_proj * cam4;

    if (cam4.z > -0.001) {
        gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
        v_pix_off = vec2(0.0); v_color = vec3(0.0);
        v_opacity = 0.0; v_cov2d = vec3(0.0);
        v_world_pos = vec3(0.0); v_normal = vec3(0.0);
        return;
    }

    float s00 = a_cov2d.x, s01 = a_cov2d.y, s11 = a_cov2d.z;

    // Eigendecomposition of 2x2 pixel covariance
    float trace = s00 + s11;
    float disc  = sqrt(max((s00-s11)*(s00-s11)*0.25 + s01*s01, 0.0));
    float lam1  = trace*0.5 + disc;
    float lam2  = max(trace*0.5 - disc, 0.0);

    vec2 e1;
    if (abs(s01) > 1e-5) {
        e1 = normalize(vec2(lam1 - s11, s01));
    } else {
        e1 = (s00 >= s11) ? vec2(1.0,0.0) : vec2(0.0,1.0);
    }
    vec2 e2 = vec2(-e1.y, e1.x);

    // 3-sigma half-axes in pixels, capped to avoid filling the screen
    float r1 = min(sqrt(lam1) * 3.0, 512.0);
    float r2 = min(sqrt(lam2) * 3.0, 512.0);

    // Pixel offset for this corner
    vec2 off_px  = a_quad.x * e1 * r1 + a_quad.y * e2 * r2;

    // Convert pixel offset to NDC offset (viewport in pixels)
    vec2 ndc_ctr = clip.xy / clip.w;
    vec2 ndc_pos = ndc_ctr + off_px * (2.0 / u_viewport);

    gl_Position = vec4(ndc_pos * clip.w, clip.z, clip.w);

    v_pix_off   = off_px;
    v_color     = a_color;
    v_opacity   = a_opacity;
    v_cov2d     = a_cov2d;
    v_world_pos = a_mean;
    v_normal    = a_normal;
}
"""

FRAG = """
#version 330 core

in vec2  v_pix_off;
in vec3  v_color;
in float v_opacity;
in vec3  v_cov2d;
in vec3  v_world_pos;
in vec3  v_normal;

uniform vec3  u_light_pos;    // world-space point light
uniform vec3  u_view_pos;     // camera eye, world-space
uniform vec3  u_light_color;
uniform float u_ambient;

out vec4 fragColor;

void main() {
    float s00 = v_cov2d.x, s01 = v_cov2d.y, s11 = v_cov2d.z;

    // Inverse of 2x2 covariance (pixel² units)
    float det = s00*s11 - s01*s01;
    if (det < 1e-6) discard;

    float inv_det = 1.0 / det;
    float i00 =  s11 * inv_det;
    float i01 = -s01 * inv_det;
    float i11 =  s00 * inv_det;

    // Mahalanobis² using the actual pixel offset from the splat centre
    float dx = v_pix_off.x, dy = v_pix_off.y;
    float mah2  = dx*(i00*dx + i01*dy) + dy*(i01*dx + i11*dy);
    float power = -0.5 * mah2;

    if (power < -4.5) discard;     // beyond 3-sigma

    float gauss_alpha = exp(power) * v_opacity;
    if (gauss_alpha < 1.0/255.0) discard;

    // ── Blinn-Phong shading — this is what makes the splat react to light ──
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_pos - v_world_pos);
    vec3 V = normalize(u_view_pos  - v_world_pos);
    vec3 H = normalize(L + V);

    float diff = max(dot(N, L), 0.0);
    float spec = pow(max(dot(N, H), 0.0), 40.0);
    // Fresnel-style rim so silhouette edges catch extra light (adds depth,
    // stops the splat cloud from reading as flat painted dots)
    float rim  = pow(1.0 - max(dot(N, V), 0.0), 3.0) * 0.25;

    vec3 lit = v_color * (u_ambient + diff * 0.85) * u_light_color
             + vec3(1.0) * spec * 0.5
             + v_color * rim;
    lit = clamp(lit, 0.0, 1.0);

    fragColor = vec4(lit, gauss_alpha);
}
"""

# ── Marker shader: crisp solid-colour disc for centroid / pivot indicators ──
# Independent from the Gaussian splat shader since markers need a fixed
# pixel-radius circle, not a covariance-driven blur.
MARKER_VERT = """
#version 330 core

in vec2  a_quad;        // (-1..1, -1..1) quad corner
in vec3  a_center;      // world-space marker position
in vec3  a_color;
in float a_radius_px;   // marker radius in screen pixels

uniform mat4 u_view;
uniform mat4 u_proj;
uniform vec2 u_viewport;

out vec2  v_uv;
out vec3  v_color;

void main() {
    vec4 cam4 = u_view * vec4(a_center, 1.0);
    vec4 clip = u_proj * cam4;

    vec2 ndc_ctr = clip.xy / clip.w;
    vec2 off_px  = a_quad * a_radius_px;
    vec2 ndc_pos = ndc_ctr + off_px * (2.0 / u_viewport);

    // Push slightly toward camera in depth so markers draw on top of splats
    gl_Position = vec4(ndc_pos * clip.w, clip.z - 0.001 * clip.w, clip.w);

    v_uv    = a_quad;
    v_color = a_color;
}
"""

MARKER_FRAG = """
#version 330 core

in vec2 v_uv;
in vec3 v_color;

out vec4 fragColor;

void main() {
    float d = length(v_uv);
    if (d > 1.0) discard;          // circular disc
    // Thin dark outline for visibility against any background colour
    float edge = smoothstep(1.0, 0.85, d);
    vec3 col = mix(vec3(0.0), v_color, edge);
    fragColor = vec4(col, 1.0);
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Window
# ═══════════════════════════════════════════════════════════════════════════════

class GaussianSplatWindow(mglw.WindowConfig):
    title       = "3D Gaussian Splatting"
    gl_version  = (3, 3)
    window_size = (1280, 720)
    resizable   = True

    _g: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        g              = GaussianSplatWindow._g
        self.means     = g["mean"]
        self.colors    = g["color"]
        self.opacities = g["opacity"]
        self.cov3d     = g["cov3d"]
        self.normals   = g.get("normal", np.zeros_like(g["mean"]))
        self.N         = len(self.means)

        centre = self.means.mean(0)
        dists  = np.linalg.norm(self.means - centre, axis=1)
        radius = float(np.percentile(dists, 90)) * 3.5

        # Centroid of all Gaussians — rendered as a fixed bright-yellow marker
        self.centroid = centre.astype(np.float64)

        # ── key light: fixed above/beside the object so rotating or orbiting
        # around it visibly changes the shading (this is what makes the
        # splat "interact" with a light source rather than look flat-shaded) ─
        self.light_pos   = (centre + radius * np.array([0.55, 1.25, 0.85])).astype(np.float32)
        self.light_color = np.array([1.0, 0.97, 0.9], dtype=np.float32)  # warm white
        self.ambient      = 0.28

        W, H = self.window_size
        self.cam   = OrbitCamera(centre, radius, aspect=W/H)
        self._W, self._H = W, H

        # Rotation-axis (pivot) picking state.
        # Pressing 'C' arms pick mode; the next left-click selects the
        # nearest Gaussian centre to the click ray as the new pivot.
        self._pick_mode = False
        self.pivot_point = self.centroid.copy()   # default pivot = centroid

        # Mouse state — manual fallback flags used only when wnd.mouse_states
        # is unavailable (very old moderngl-window builds).
        self._mouse_pos   = (0, 0)
        self._mouse_left  = False
        self._mouse_right = False
        self._mouse_mid   = False

        self._fps_t = 0.0
        self._fps_n = 0

        # depth-sort cache: skip re-sort when camera hasn't moved
        self._sort_order   = None
        self._last_cam_row = None

        self._build_gl()

    def _build_gl(self):
        self.prog = self.ctx.program(vertex_shader=VERT, fragment_shader=FRAG)

        quad = np.array([[-1,-1],[-1,1],[1,-1],[1,1]], dtype=np.float32)
        self.quad_vbo = self.ctx.buffer(quad.tobytes())

        self._ibuf = np.zeros((self.N, 13), dtype=np.float32)
        self._ibuf[:,0:3]   = self.means
        self._ibuf[:,3:6]   = self.colors
        self._ibuf[:,6]     = self.opacities
        self._ibuf[:,10:13] = self.normals
        self.inst_vbo = self.ctx.buffer(self._ibuf.tobytes(), dynamic=True)

        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (self.quad_vbo, "2f",             "a_quad"),
                (self.inst_vbo, "3f 3f 1f 3f 3f /i",
                 "a_mean", "a_color", "a_opacity", "a_cov2d", "a_normal"),
            ],
        )

        # ── Marker pipeline (centroid + pivot indicator dots) ────────────────
        self.marker_prog = self.ctx.program(vertex_shader=MARKER_VERT,
                                            fragment_shader=MARKER_FRAG)
        self.marker_quad_vbo = self.ctx.buffer(quad.tobytes())

        # Up to 2 markers: [0]=centroid (always shown), [1]=pivot (shown only
        # when it differs from the centroid). Layout per row:
        # center(3) color(3) radius_px(1) = 7 floats
        self._marker_buf = np.zeros((2, 7), dtype=np.float32)
        self.marker_vbo  = self.ctx.buffer(self._marker_buf.tobytes(), dynamic=True)

        self.marker_vao = self.ctx.vertex_array(
            self.marker_prog,
            [
                (self.marker_quad_vbo, "2f",          "a_quad"),
                (self.marker_vbo,      "3f 3f 1f /i",
                 "a_center", "a_color", "a_radius_px"),
            ],
        )

    def _cached_depth_sort(self, V):
        """Re-sort only when the camera view row has changed."""
        cam_row = V[2:3, :3].copy()
        if (self._sort_order is None or
                self._last_cam_row is None or
                not np.allclose(cam_row, self._last_cam_row, atol=1e-6)):
            self._sort_order   = depth_sort(self.means, V)
            self._last_cam_row = cam_row
        return self._sort_order

    # ── render ────────────────────────────────────────────────────────────────

    def on_render(self, time: float, frame_time: float):
        self.ctx.clear(0.04, 0.04, 0.06)
        self.ctx.enable(moderngl.BLEND)
        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        W, H = self._W, self._H
        V    = self.cam.view()
        P    = self.cam.proj()

        order = self._cached_depth_sort(V)
        c2d   = project_cov2d(self.means, self.cov3d, V, P, W, H)

        buf = self._ibuf
        buf[order, 0:3]   = self.means[order]
        buf[order, 3:6]   = self.colors[order]
        buf[order, 6]     = self.opacities[order]
        buf[order, 7:10]  = c2d[order]
        buf[order, 10:13] = self.normals[order]
        self.inst_vbo.write(buf.tobytes())

        self.prog["u_view"].write(V.T.tobytes())
        self.prog["u_proj"].write(P.T.tobytes())
        self.prog["u_viewport"].value = (float(W), float(H))
        self.prog["u_light_pos"].value   = tuple(self.light_pos.tolist())
        self.prog["u_view_pos"].value    = tuple(self.cam.position.astype(np.float32).tolist())
        self.prog["u_light_color"].value = tuple(self.light_color.tolist())
        self.prog["u_ambient"].value     = self.ambient
        self.vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=self.N)

        # ── Draw markers: centroid (always) + pivot (only if user picked one) ──
        markers = self._marker_buf
        markers[0, 0:3] = self.centroid.astype(np.float32)
        markers[0, 3:6] = (1.0, 1.0, 0.0)        # bright yellow
        markers[0, 6]   = 7.0                    # px radius

        showing_pivot = not np.allclose(self.pivot_point, self.centroid, atol=1e-6)
        if showing_pivot:
            markers[1, 0:3] = self.pivot_point.astype(np.float32)
            markers[1, 3:6] = (0.1, 1.0, 0.3)    # bright green pivot marker
            markers[1, 6]   = 6.0
            n_markers = 2
        else:
            n_markers = 1

        self.marker_vbo.write(markers.tobytes())
        self.marker_prog["u_view"].write(V.T.tobytes())
        self.marker_prog["u_proj"].write(P.T.tobytes())
        self.marker_prog["u_viewport"].value = (float(W), float(H))
        self.marker_vao.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=n_markers)

        self._fps_t += frame_time
        self._fps_n += 1
        if self._fps_t >= 0.5:
            fps = self._fps_n / self._fps_t
            dev = (torch.cuda.get_device_name(0)
                   if TORCH and DEVICE.type=="cuda" else "CPU")
            self.wnd.title = (f"3D Gaussian Splatting | {self.N:,} splats | "
                              f"{fps:.1f} FPS | {dev}")
            self._fps_t = self._fps_n = 0

    def _screen_to_ray(self, x: int, y: int):
        """
        Convert a screen-space pixel coordinate into a world-space ray
        (origin, direction) using the current view/projection matrices.
        """
        W, H = self._W, self._H
        # NDC in [-1,1], Y flipped (screen Y grows downward, NDC Y grows up)
        ndc_x = (2.0 * x / W) - 1.0
        ndc_y = 1.0 - (2.0 * y / H)

        V = self.cam.view().astype(np.float64)
        P = self.cam.proj().astype(np.float64)

        inv_VP = np.linalg.inv(P @ V)

        near_clip = np.array([ndc_x, ndc_y, -1.0, 1.0])
        far_clip  = np.array([ndc_x, ndc_y,  1.0, 1.0])

        near_world = inv_VP @ near_clip
        far_world  = inv_VP @ far_clip
        near_world /= near_world[3]
        far_world  /= far_world[3]

        origin = near_world[:3]
        direction = far_world[:3] - near_world[:3]
        direction /= np.linalg.norm(direction)
        return origin, direction

    def _pick_pivot_at(self, x: int, y: int):
        """
        Find the Gaussian centre closest to the click ray (perpendicular
        distance) and use it as the new orbit pivot, preserving the current
        camera viewpoint so the scene doesn't jump.
        """
        origin, direction = self._screen_to_ray(x, y)

        pts = self.means.astype(np.float64)              # (N,3)
        to_pts = pts - origin                             # (N,3)
        t = to_pts @ direction                             # projection length along ray
        t = np.maximum(t, 0.0)                             # ignore points behind the ray origin
        closest_on_ray = origin + t[:, None] * direction   # (N,3)
        perp_dist = np.linalg.norm(pts - closest_on_ray, axis=1)

        idx = int(np.argmin(perp_dist))
        picked = pts[idx]

        self.pivot_point = picked.copy()
        self.cam.set_pivot_preserve_view(picked)

    # ── mouse ─────────────────────────────────────────────────────────────────
    # THE PROBLEM: button integers are not normalised across backends.
    # GLFW passes raw GLFW_MOUSE_BUTTON_* values which moderngl-window may
    # forward as-is or shift by 1 depending on version — any integer-decoding
    # scheme breaks silently on at least one backend/version combination.
    #
    # FIX: read self.wnd.mouse_states — a struct maintained by moderngl-window
    # itself with .left / .right / .middle bool attributes that are always
    # correct. Manual _mouse_* flags are kept only as a last-resort fallback.

    def _btn_left(self):
        ms = getattr(self.wnd, "mouse_states", None)
        if ms is not None:
            return bool(ms.left)
        return self._mouse_left

    def _btn_right(self):
        ms = getattr(self.wnd, "mouse_states", None)
        if ms is not None:
            return bool(ms.right)
        return self._mouse_right

    def _btn_mid(self):
        ms = getattr(self.wnd, "mouse_states", None)
        if ms is not None:
            return bool(ms.middle)
        return self._mouse_mid

    def on_mouse_press_event(self, x: int, y: int, button: int):
        self._mouse_pos = (x, y)

        if button == 1 and self._pick_mode:
            # Consume this click as a pivot pick instead of starting an orbit drag.
            self._pick_pivot_at(x, y)
            self._pick_mode = False
            dev = (torch.cuda.get_device_name(0)
                   if TORCH and DEVICE.type=="cuda" else "CPU")
            self.wnd.title = (f"3D Gaussian Splatting | {self.N:,} splats | "
                              f"pivot set | {dev}")
            return

        # Fallback flags — only used when mouse_states is unavailable.
        # button==1 is left on every backend; anything else treated as right.
        if button == 1:
            self._mouse_left  = True
        else:
            self._mouse_right = True

    def on_mouse_release_event(self, x: int, y: int, button: int):
        self._mouse_left  = False
        self._mouse_right = False
        self._mouse_mid   = False

    def on_mouse_drag_event(self, x: int, y: int, dx: int, dy: int):
        """Fired when the mouse moves while a button is held (most backends)."""
        self._mouse_pos = (x, y)
        if self._btn_left():
            self.cam.orbit(dx, dy)
        if self._btn_right() or self._btn_mid():
            self.cam.pan(dx, dy)

    def on_mouse_position_event(self, x: int, y: int, dx: int, dy: int):
        """
        Fired on every mouse move regardless of button state.
        Belt-and-suspenders: some backends skip mouse_drag_event entirely.
        We compute our own delta so this is always correct.
        """
        px, py = self._mouse_pos
        self._mouse_pos = (x, y)
        cdx, cdy = x - px, y - py
        if self._btn_left():
            self.cam.orbit(cdx, cdy)
        if self._btn_right() or self._btn_mid():
            self.cam.pan(cdx, cdy)

    def on_mouse_scroll_event(self, x_offset: float, y_offset: float):
        self.cam.zoom(y_offset)

    # ── keyboard ──────────────────────────────────────────────────────────────

    def on_key_event(self, key, action, modifiers):
        if action != self.wnd.keys.ACTION_PRESS:
            return
        k = self.wnd.keys
        if   key == k.ESCAPE: self.wnd.close()
        elif key == k.A:      self.cam.orbit(-15, 0)
        elif key == k.D:      self.cam.orbit( 15, 0)
        elif key == k.W:      self.cam.orbit(0,  15)
        elif key == k.S:      self.cam.orbit(0, -15)
        elif key == k.Q:      self.cam.zoom( 1)
        elif key == k.E:      self.cam.zoom(-1)
        elif key == k.C:
            # Arm pivot-picking mode. Next left-click sets the new rotation axis.
            self._pick_mode = True
            self.wnd.title = (f"3D Gaussian Splatting | {self.N:,} splats | "
                              f"CLICK A POINT to set rotation axis (Esc-free, "
                              f"any left-click picks)")
        elif key == k.R:
            centre = self.means.mean(0)
            dists  = np.linalg.norm(self.means - centre, axis=1)
            self.pivot_point = centre.astype(np.float64)
            self.cam.set_pivot_preserve_view(centre.astype(np.float64))
            self.cam.theta  = math.pi / 4
            self.cam.phi    = math.pi / 3
            self.cam.radius = float(np.percentile(dists, 90)) * 3.5

    # ── resize ────────────────────────────────────────────────────────────────

    def on_resize(self, w: int, h: int):
        self._W, self._H  = w, h
        self.cam.aspect   = w / max(h, 1)
        self.ctx.viewport = (0, 0, w, h)


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  PLY loader
# ═══════════════════════════════════════════════════════════════════════════════

def load_ply(path: str):
    pcd = o3d.io.read_point_cloud(path)
    pts = np.asarray(pcd.points, dtype=np.float32)
    if len(pts) == 0:
        raise ValueError(f"No points in {path}")
    clr = (np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors()
           else np.full((len(pts),3), 0.75, np.float32))
    return pts, clr


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description="3D Gaussian Splatting viewer")
    ap.add_argument("--ply", default="splat_exports/scene_splat_20260703_092632.ply", help="Input .ply point cloud")
    ap.add_argument("--k",     type=int,   default=100,
                    help="k-NN for covariance PCA  (default 16)")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Covariance scale  (default 0.5; smaller = tighter splats)")
    args = ap.parse_args()

    print(f"Loading: {args.ply}")
    pts, clr = load_ply(args.ply)
    print(f"  {len(pts):,} points")

    print(f"Estimating Gaussians  k={args.k}  scale={args.scale} ...")
    g = estimate_gaussians(pts, clr, k=args.k, scale=args.scale)
    print("  Done.")

    GaussianSplatWindow._g = g
    mglw.run_window_config(GaussianSplatWindow)


if __name__ == "__main__":
    main()