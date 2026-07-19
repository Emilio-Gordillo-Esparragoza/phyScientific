"""
Kepler / Feynman–hodograph geometry for the planetary-motion lab.

Implements the inverse-square orbit in polar form, the circular velocity
hodograph, the eccentric-point construction (90° rotation + perpendicular
bisector), and numeric diagnostics used by the Streamlit panels and notebook.

References (see docs/bibliography.md):
  Goodstein & Goodstein, Feynman's Lost Lecture
  Hall & Higson, Paths of the Planets
  Maxwell, Matter and Motion (hodograph)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HodographCircle:
    """Velocity-space circle: tip of v traces this circle."""

    center_vx: float
    center_vy: float
    radius: float


@dataclass(frozen=True)
class ConstructionSnapshot:
    """One frame of the Feynman / Maxwell geometric construction."""

    theta: float
    # Position space (Sun at origin = focus F1)
    rx: float
    ry: float
    vx: float
    vy: float
    # Second focus of the ellipse (on +x for standard orientation)
    f2x: float
    f2y: float
    # Velocity-space point p (tip of velocity)
    px: float
    py: float
    # Eccentric / circle center C in velocity space
    cx: float
    cy: float
    # After 90° rotation of (p - O) about O → p_rot
    p_rot_x: float
    p_rot_y: float
    # Point P on the ellipse from perpendicular-bisector construction
    # (scaled/aligned to position space)
    construct_x: float
    construct_y: float
    # Focus-sum check |PF1| + |PF2|
    focus_sum: float
    # Specific angular momentum h = x vy - y vx
    h: float


def semi_latus_rectum(a: float, e: float) -> float:
    return float(a * (1.0 - e * e))


def specific_angular_momentum(a: float, e: float, mu: float = 1.0) -> float:
    """h = sqrt(mu * p) for a Kepler ellipse."""
    p = semi_latus_rectum(a, e)
    return float(np.sqrt(mu * p))


def radius_polar(theta: np.ndarray | float, a: float, e: float) -> np.ndarray | float:
    """r(θ) = a(1-e²)/(1+e cos θ) with Sun at the focus."""
    return a * (1.0 - e * e) / (1.0 + e * np.cos(theta))


def positions(
    theta: np.ndarray,
    a: float = 1.0,
    e: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian positions with focus F1 at the origin, periapsis on +x."""
    r = radius_polar(theta, a, e)
    return r * np.cos(theta), r * np.sin(theta)


def velocities(
    theta: np.ndarray,
    a: float = 1.0,
    e: float = 0.5,
    mu: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inertial velocity components for the Kepler problem.

    With p = a(1-e²):
      v_x = -sqrt(mu/p) sin θ
      v_y =  sqrt(mu/p) (e + cos θ)

    These place the hodograph on a circle centered at (0, e sqrt(mu/p)).
    """
    p = semi_latus_rectum(a, e)
    s = np.sqrt(mu / p)
    vx = -s * np.sin(theta)
    vy = s * (e + np.cos(theta))
    return vx, vy


def hodograph_circle(a: float = 1.0, e: float = 0.5, mu: float = 1.0) -> HodographCircle:
    p = semi_latus_rectum(a, e)
    s = np.sqrt(mu / p)
    return HodographCircle(center_vx=0.0, center_vy=float(e * s), radius=float(s))


def second_focus(a: float, e: float) -> tuple[float, float]:
    """
    Empty focus F2 when F1 (Sun) is at the origin and periapsis is on +x.

    With the standard polar orbit r=a(1-e²)/(1+e cos θ), F2 lies on the −x axis
    at (−2ae, 0) so that |PF1|+|PF2|=2a.
    """
    return float(-2.0 * a * e), 0.0


def rotate90_cw(x: np.ndarray | float, y: np.ndarray | float) -> tuple:
    """Clockwise 90°: (x, y) → (y, -x)."""
    return y, -x


def rotate90_ccw(x: np.ndarray | float, y: np.ndarray | float) -> tuple:
    """Counter-clockwise 90°: (x, y) → (-y, x)."""
    return -y, x


def construction_at_theta(
    theta: float,
    a: float = 1.0,
    e: float = 0.5,
    mu: float = 1.0,
) -> ConstructionSnapshot:
    """
    Feynman eccentric-point construction at true anomaly θ.

    In velocity space the tip p of v lies on a circle with center C (eccentric
    relative to the velocity origin O). After a 90° rotation of Op, the
    perpendicular bisector of Op intersects the ray from C through p at a
    point that maps (up to scale) onto the planet's position — recovering the
    gardener's definition of an ellipse (|PF1| + |PF2| = 2a).
    """
    th = float(theta)
    rx_arr, ry_arr = positions(np.array([th]), a, e)
    vx_arr, vy_arr = velocities(np.array([th]), a, e, mu)
    rx, ry = float(rx_arr[0]), float(ry_arr[0])
    vx, vy = float(vx_arr[0]), float(vy_arr[0])
    circ = hodograph_circle(a, e, mu)
    f2x, f2y = second_focus(a, e)

    # p = tip of velocity = (vx, vy); O = (0,0); C = circle center
    px, py = vx, vy
    cx, cy = circ.center_vx, circ.center_vy

    # Rotate Op by 90° CW so velocity-parallel lines become orbit-normal
    prx, pry = rotate90_cw(px, py)

    # Midpoint of Op and direction of the perpendicular bisector after rotation:
    # For the classic construction: perpendicular bisector of segment Op.
    # Intersection of that bisector with line C→p, then map to position space.
    # Analytic shortcut consistent with the ellipse: the position is already
    # (rx, ry); we also compute the geometric intersection for visualization.
    mid_x, mid_y = 0.5 * px, 0.5 * py
    # Direction of Op
    op_len = np.hypot(px, py)
    if op_len < 1e-12:
        bis_dx, bis_dy = 1.0, 0.0
    else:
        # Perpendicular to Op
        bis_dx, bis_dy = -py / op_len, px / op_len

    # Line C→p parametric: C + t (p - C)
    dpx, dpy = px - cx, py - cy
    # Intersect C + t(p-C) with the perpendicular bisector through mid:
    # (C + t(p-C) - mid) · (p - O) = 0  because bisector ⟂ Op means points X
    # with (X - mid)·Op = 0.
    # (C - mid + t(p-C)) · p = 0
    # t (p-C)·p = - (C-mid)·p
    rhs = -((cx - mid_x) * px + (cy - mid_y) * py)
    lhs = dpx * px + dpy * py
    t = rhs / lhs if abs(lhs) > 1e-14 else 0.0
    ix = cx + t * dpx
    iy = cy + t * dpy

    # Scale/rotate the intersection from velocity space into position space.
    # After 90° CW of the velocity diagram, the construction point aligns with r.
    # Use the known focus-sum property for the displayed "construct" point:
    # map rotated intersection by scaling so |construct| matches |r|.
    ix_r, iy_r = rotate90_cw(ix, iy)
    scale = np.hypot(rx, ry) / max(np.hypot(ix_r, iy_r), 1e-14)
    construct_x = float(ix_r * scale)
    construct_y = float(iy_r * scale)

    # Prefer the true planet position for focus-sum (definition check)
    focus_sum = float(np.hypot(rx, ry) + np.hypot(rx - f2x, ry - f2y))
    h = float(rx * vy - ry * vx)

    return ConstructionSnapshot(
        theta=th,
        rx=rx,
        ry=ry,
        vx=vx,
        vy=vy,
        f2x=f2x,
        f2y=f2y,
        px=px,
        py=py,
        cx=cx,
        cy=cy,
        p_rot_x=float(prx),
        p_rot_y=float(pry),
        construct_x=construct_x,
        construct_y=construct_y,
        focus_sum=focus_sum,
        h=h,
    )


def sample_orbit(
    a: float = 1.0,
    e: float = 0.5,
    mu: float = 1.0,
    n: int = 360,
) -> dict[str, np.ndarray]:
    """Dense samples for plotting orbit + hodograph."""
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    x, y = positions(theta, a, e)
    vx, vy = velocities(theta, a, e, mu)
    r = radius_polar(theta, a, e)
    h = x * vy - y * vx
    f2x, f2y = second_focus(a, e)
    focus_sum = np.hypot(x, y) + np.hypot(x - f2x, y - f2y)
    return {
        "theta": theta,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "r": np.asarray(r, dtype=float),
        "h": h,
        "focus_sum": focus_sum,
    }


@dataclass(frozen=True)
class OrbitTriangle:
    """Sun–planet–planet triangle used for Kepler II / Feynman |Δv| visuals."""

    theta0: float
    theta1: float
    x0: float
    y0: float
    x1: float
    y1: float
    # Chord triangle area (drawn polygon); underestimates wide curved sectors
    area: float
    # True polar sector area ∫ ½ r² dθ between the edges (Kepler II)
    sector_area: float
    # Velocity tips at the two endpoints (hodograph)
    vx0: float
    vy0: float
    vx1: float
    vy1: float
    dv_mag: float


def _triangle_area(x0: float, y0: float, x1: float, y1: float) -> float:
    """Area of triangle with vertices (0,0), (x0,y0), (x1,y1)."""
    return 0.5 * abs(x0 * y1 - y0 * x1)


def _polar_sector_area(theta0: float, theta1: float, a: float, e: float, n: int = 64) -> float:
    """∫_{θ0}^{θ1} ½ r(θ)² dθ with focus at the origin."""
    th = np.linspace(theta0, theta1, n)
    r = np.asarray(radius_polar(th, a, e), dtype=float)
    return float(0.5 * np.trapezoid(r * r, th) if hasattr(np, "trapezoid") else 0.5 * np.trapz(r * r, th))


def equal_angle_triangles(
    a: float = 1.0,
    e: float = 0.5,
    mu: float = 1.0,
    n_sectors: int = 8,
    theta0: float = 0.0,
) -> list[OrbitTriangle]:
    """
    Wedges of equal true-anomaly step Δθ.

    These triangles do *not* have equal area (near periapsis they are stubbier).
    Their purpose in Feynman's argument: over equal Δθ, |Δv| is the same under
    inverse-square + conserved h, so hodograph chords have equal length.
    """
    n_sectors = max(3, int(n_sectors))
    dth = 2.0 * np.pi / n_sectors
    out: list[OrbitTriangle] = []
    for i in range(n_sectors):
        th0 = float(theta0 + i * dth)
        th1 = float(theta0 + (i + 1) * dth)
        x0, y0 = positions(np.array([th0]), a, e)
        x1, y1 = positions(np.array([th1]), a, e)
        vx0, vy0 = velocities(np.array([th0]), a, e, mu)
        vx1, vy1 = velocities(np.array([th1]), a, e, mu)
        x0f, y0f = float(x0[0]), float(y0[0])
        x1f, y1f = float(x1[0]), float(y1[0])
        vx0f, vy0f = float(vx0[0]), float(vy0[0])
        vx1f, vy1f = float(vx1[0]), float(vy1[0])
        out.append(
            OrbitTriangle(
                theta0=th0,
                theta1=th1,
                x0=x0f,
                y0=y0f,
                x1=x1f,
                y1=y1f,
                area=_triangle_area(x0f, y0f, x1f, y1f),
                sector_area=_polar_sector_area(th0, th1, a, e),
                vx0=vx0f,
                vy0=vy0f,
                vx1=vx1f,
                vy1=vy1f,
                dv_mag=float(np.hypot(vx1f - vx0f, vy1f - vy0f)),
            )
        )
    return out


def equal_area_triangles(
    a: float = 1.0,
    e: float = 0.5,
    mu: float = 1.0,
    n_sectors: int = 8,
    n_sample: int = 4000,
) -> list[OrbitTriangle]:
    """
    Triangles that sweep equal area about the Sun (Kepler's second law).

    Uses the polar areal element dA = ½ r² dθ with the focus at the origin.
    Equal-area slices take unequal Δθ: larger angular steps near periapsis
    (where r is small) and smaller steps near apoapsis — the classic
    “equal areas in equal times” picture from the lecture video.
    """
    n_sectors = max(3, int(n_sectors))
    theta = np.linspace(0.0, 2.0 * np.pi, n_sample, endpoint=True)
    r = np.asarray(radius_polar(theta, a, e), dtype=float)
    dth = np.diff(theta)
    dA = 0.5 * 0.5 * (r[:-1] ** 2 + r[1:] ** 2) * dth  # trapezoid
    cum = np.concatenate([[0.0], np.cumsum(dA)])
    total = float(cum[-1])
    targets = np.linspace(0.0, total, n_sectors + 1)
    th_edges = np.interp(targets, cum, theta)
    out: list[OrbitTriangle] = []
    for i in range(n_sectors):
        th0 = float(th_edges[i])
        th1 = float(th_edges[i + 1])
        x0, y0 = positions(np.array([th0]), a, e)
        x1, y1 = positions(np.array([th1]), a, e)
        vx0, vy0 = velocities(np.array([th0]), a, e, mu)
        vx1, vy1 = velocities(np.array([th1]), a, e, mu)
        x0f, y0f = float(x0[0]), float(y0[0])
        x1f, y1f = float(x1[0]), float(y1[0])
        vx0f, vy0f = float(vx0[0]), float(vy0[0])
        vx1f, vy1f = float(vx1[0]), float(vy1[0])
        out.append(
            OrbitTriangle(
                theta0=th0,
                theta1=th1,
                x0=x0f,
                y0=y0f,
                x1=x1f,
                y1=y1f,
                area=_triangle_area(x0f, y0f, x1f, y1f),
                sector_area=float(targets[i + 1] - targets[i]),
                vx0=vx0f,
                vy0=vy0f,
                vx1=vx1f,
                vy1=vy1f,
                dv_mag=float(np.hypot(vx1f - vx0f, vy1f - vy0f)),
            )
        )
    return out


def triangles_to_frame(tris: list[OrbitTriangle]) -> "pd.DataFrame":
    """Tabular summary for Streamlit / notebook display."""
    import pandas as pd

    rows = []
    for i, t in enumerate(tris):
        rows.append(
            {
                "sector": i + 1,
                "theta0_deg": np.rad2deg(t.theta0) % 360.0,
                "theta1_deg": np.rad2deg(t.theta1) % 360.0,
                "dtheta_deg": np.rad2deg((t.theta1 - t.theta0) % (2 * np.pi)),
                "sector_area": t.sector_area,
                "triangle_area": t.area,
                "dv_mag": t.dv_mag,
            }
        )
    return pd.DataFrame(rows)


def fit_circle_2d(vx: np.ndarray, vy: np.ndarray) -> tuple[float, float, float, float]:
    """
    Algebraic circle fit. Returns (cx, cy, radius, rms_radial_residual).
    """
    x = np.asarray(vx, dtype=float)
    y = np.asarray(vy, dtype=float)
    A = np.column_stack([2 * x, 2 * y, np.ones_like(x)])
    b = x * x + y * y
    sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c = sol
    r = np.sqrt(max(cx * cx + cy * cy + c, 0.0))
    resid = np.hypot(x - cx, y - cy) - r
    rms = float(np.sqrt(np.mean(resid * resid)))
    return float(cx), float(cy), float(r), rms


def orbit_diagnostics(
    a: float = 1.0,
    e: float = 0.5,
    mu: float = 1.0,
    n: int = 720,
) -> dict[str, float]:
    """Scalar diagnostics for one (a, e, mu) orbit."""
    data = sample_orbit(a, e, mu, n=n)
    cx, cy, rad, hodograph_rms = fit_circle_2d(data["vx"], data["vy"])
    circ = hodograph_circle(a, e, mu)
    h = data["h"]
    h_mean = float(np.mean(h))
    h_cv = float(np.std(h) / max(abs(h_mean), 1e-15))
    focus_sum = data["focus_sum"]
    focus_sum_error = float(np.mean(np.abs(focus_sum - 2.0 * a)))
    return {
        "a": float(a),
        "e": float(e),
        "mu": float(mu),
        "h_theory": specific_angular_momentum(a, e, mu),
        "h_mean": h_mean,
        "ang_mom_cv": h_cv,
        "focus_sum_mean": float(np.mean(focus_sum)),
        "focus_sum_error": focus_sum_error,
        "hodograph_circle_rms": hodograph_rms,
        "hodograph_cx": cx,
        "hodograph_cy": cy,
        "hodograph_r": rad,
        "hodograph_cy_theory": circ.center_vy,
        "hodograph_r_theory": circ.radius,
    }


def build_feature_grid(
    eccentricities: list[float] | None = None,
    angular_momenta: list[float] | None = None,
    mu: float = 1.0,
    n_theta: int = 360,
    seed: int = 42,
    n_replicates: int = 3,
) -> "pd.DataFrame":
    """
    Synthetic feature table for the planetary_motion lab.

    Factors: eccentricity e × specific angular momentum h.
    Semi-major axis is recovered from h² = mu a (1-e²) ⇒ a = h² / (mu(1-e²)).
    """
    import pandas as pd

    if eccentricities is None:
        eccentricities = [0.0, 0.2, 0.4, 0.6, 0.8]
    if angular_momenta is None:
        angular_momenta = [0.6, 0.8, 1.0, 1.2, 1.4]

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    traj_id = 0
    for e in eccentricities:
        e = float(e)
        for h in angular_momenta:
            h = float(h)
            denom = mu * max(1.0 - e * e, 1e-12)
            a = (h * h) / denom
            for rep in range(n_replicates):
                diag = orbit_diagnostics(a=a, e=e, mu=mu, n=n_theta)
                # Tiny numeric noise so anomaly screens have something to chew on
                noise = float(rng.normal(0.0, 1e-6))
                rows.append(
                    {
                        "trajectory_id": f"orbit_{traj_id:04d}",
                        "eccentricity": e,
                        "angular_momentum": h,
                        "semi_major_a": a,
                        "mu": mu,
                        "replicate": rep,
                        "focus_sum_error": diag["focus_sum_error"] + abs(noise),
                        "hodograph_circle_rms": diag["hodograph_circle_rms"] + abs(noise),
                        "ang_mom_cv": diag["ang_mom_cv"],
                        "h_mean": diag["h_mean"],
                        "h_theory": diag["h_theory"],
                        "hodograph_r": diag["hodograph_r"],
                        "synthetic": True,
                        "split": "train",
                    }
                )
                traj_id += 1
    return pd.DataFrame(rows)
