"""Build notebooks/feynman_lost_lecture.ipynb (elementary + formal proofs)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "feynman_lost_lecture.ipynb"


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


cells = [
    md(
        """# Feynman’s Lost Lecture — The Motion of Planets Around the Sun

> I am going to give what I will call an elementary demonstration. But elementary does not mean easy to understand. Elementary means that very little is required to know ahead of time in order to understand it, except to have an infinite amount of intelligence. There may be a large number of steps that hard to follow, but to each does not require already knowing the calculus or Fourier transforms. Yeah, that’s all, infinite intelligence. I think you’re up to that, don’t you?
>
> — **Richard Feynman**, 1964

**Goal.** From Newton’s inverse-square law and Kepler’s second law (equal areas), prove that a bound orbit is an **ellipse** with the Sun at a focus.

This notebook has two parts:

- **Part A — Elementary demonstration** (triangles, equal areas, hodograph, eccentric point)
- **Part B — Formal demonstration** (vector hodograph, Runge–Lenz, optional Binet)

Docs: `docs/celestial/feynman_elementary.md`, `docs/celestial/feynman_formal.md`, `docs/bibliography.md`.  
Video: https://youtu.be/xdIjYBtnvZU  
Lab module: `src/orbit_feynman.py` · App: Streamlit sidebar → `planetary_motion`
"""
    ),
    md("## Setup"),
    code(
        """\
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
if not (ROOT / "src").exists():
    ROOT = Path.cwd().parent
sys.path.insert(0, str(ROOT))

from src.orbit_feynman import (
    construction_at_theta,
    equal_angle_triangles,
    equal_area_triangles,
    fit_circle_2d,
    hodograph_circle,
    orbit_diagnostics,
    sample_orbit,
    second_focus,
    triangles_to_frame,
)

# Canonical demo orbit
A, E, MU = 1.0, 0.55, 1.0
N_SEC = 8
data = sample_orbit(a=A, e=E, mu=MU, n=400)
circ = hodograph_circle(a=A, e=E, mu=MU)
area_tris = equal_area_triangles(a=A, e=E, mu=MU, n_sectors=N_SEC)
angle_tris = equal_angle_triangles(a=A, e=E, mu=MU, n_sectors=N_SEC)
area_df = triangles_to_frame(area_tris)
angle_df = triangles_to_frame(angle_tris)

print("h theory:", orbit_diagnostics(A, E, MU)["h_theory"])
print("hodograph center, radius:", circ)
print(area_df.round(4).to_string(index=False))
"""
    ),
    md(
        """## Part A — Elementary demonstration

### A.1 Kepler II: equal-area triangles (as in the lecture video)

The Sun–planet line sweeps **equal areas in equal times**. On an eccentric ellipse that forces **unequal angles**: near periapsis $r$ is small and the planet is fast, so a wider $\\Delta\\theta$ is needed to keep area $=\tfrac12 h\\,\\Delta t$ fixed; near apoapsis the wedges are skinny.

Below: shaded **equal-area** triangles vs **equal-angle** wedges on the same orbit.
"""
    ),
    code(
        """\
COLORS = ["#5F6B45", "#9A7340", "#3D5A56", "#7A3E32", "#8A9470", "#C4A06A", "#5C574E", "#4A5D3A"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

def draw_orbit_with_tris(ax, tris, title):
    ax.plot(np.append(data["x"], data["x"][0]), np.append(data["y"], data["y"][0]),
            color="#1A1814", lw=1.8, zorder=3)
    for i, t in enumerate(tris):
        ax.fill([0, t.x0, t.x1], [0, t.y0, t.y1], color=COLORS[i % len(COLORS)],
                alpha=0.45, ec="#1A1814", lw=0.8)
        # label sector mid-angle
        xm, ym = 0.55 * (t.x0 + t.x1) / 2, 0.55 * (t.y0 + t.y1) / 2
        ax.text(xm, ym, str(i + 1), fontsize=8, ha="center", va="center")
    ax.scatter([0], [0], c="#9A7340", s=90, marker="*", zorder=5, label="Sun $F_1$")
    f2x, f2y = second_focus(A, E)
    ax.scatter([f2x], [f2y], c="#5C574E", s=45, marker="x", label="$F_2$")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(loc="upper right", fontsize=8)

draw_orbit_with_tris(axes[0], area_tris, "Equal-area triangles (Kepler II)")
draw_orbit_with_tris(axes[1], angle_tris, "Equal-angle wedges (Feynman clock)")
fig.suptitle("How areas fit the law — same orbit, two partitions", y=1.02)
plt.tight_layout()
plt.show()

print("Equal-area:  sector_area CV =", (area_df["sector_area"].std() / area_df["sector_area"].mean()))
print("Equal-angle: sector_area CV =", (angle_df["sector_area"].std() / angle_df["sector_area"].mean()))
print("Equal-angle: |Δv| CV =", (angle_df["dv_mag"].std() / angle_df["dv_mag"].mean()))
"""
    ),
    md(
        """### A.2 Tables: areas vs $|\\Delta v|$

| Partition | What stays constant? | Why |
|-----------|----------------------|-----|
| Equal **area** | polar sector area $\\int\\tfrac12 r^{2}\\,d\\theta$ (∝ time) | Kepler II / conserved $h$ |
| Equal **angle** | $|\\Delta v|$ on the hodograph | inverse-square cancels $r^{2}$ in $\\Delta t$ |

That cancellation is the heart of the video: equal angles ⇒ equal velocity chords ⇒ **circle** in velocity space. Drawn chord triangles approximate the sectors; for wide periapsis wedges the chord triangle undercounts — use `sector_area` for the law.
"""
    ),
    code(
        r"""
cmp = pd.DataFrame({
    "sector": area_df["sector"],
    "eq_area_sector": area_df["sector_area"],
    "eq_angle_sector": angle_df["sector_area"],
    "eq_angle_dtheta_deg": angle_df["dtheta_deg"],
    "eq_area_dtheta_deg": area_df["dtheta_deg"],
    "eq_angle_dv": angle_df["dv_mag"],
    "eq_area_dv": area_df["dv_mag"],
})
print(cmp.round(4).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
x = np.arange(1, N_SEC + 1)
w = 0.35
axes[0].bar(x - w/2, area_df["sector_area"], width=w, color="#5F6B45", label="equal area")
axes[0].bar(x + w/2, angle_df["sector_area"], width=w, color="#9A7340", label="equal angle")
axes[0].set_xlabel("sector")
axes[0].set_ylabel("sector area")
axes[0].set_title("Polar areas: flat only for equal-area partition")
axes[0].legend(fontsize=8)

axes[1].bar(x - w/2, area_df["dv_mag"], width=w, color="#5F6B45", label="equal area")
axes[1].bar(x + w/2, angle_df["dv_mag"], width=w, color="#7A3E32", label="equal angle")
axes[1].set_xlabel("sector")
axes[1].set_ylabel(r"$|\Delta v|$")
axes[1].set_title(r"Equal $\Delta\theta$ $\Rightarrow$ flat $|\Delta v|$")
axes[1].legend(fontsize=8)
plt.tight_layout()
plt.show()
"""
    ),
    md(
        """### A.3 Position space + velocity hodograph

Translate every $\\mathbf{v}$ to a common origin. Equal-angle $|\\Delta v|$ chords (red) have equal length → the tips lie on a circle centered at the **eccentric point** $C$, offset from the velocity origin $O$.
"""
    ),
    code(
        r"""
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Position space
ax = axes[0]
for i, t in enumerate(area_tris):
    ax.fill([0, t.x0, t.x1], [0, t.y0, t.y1], color=COLORS[i % len(COLORS)], alpha=0.35, ec="#1A1814", lw=0.6)
ax.plot(np.append(data["x"], data["x"][0]), np.append(data["y"], data["y"][0]),
        color="#5F6B45", lw=2, label="orbit")
ax.scatter([0], [0], c="#9A7340", s=80, marker="*", zorder=5, label="Sun $F_1$")
f2x, f2y = second_focus(A, E)
ax.scatter([f2x], [f2y], c="#5C574E", s=50, marker="x", label="$F_2$")
th = np.deg2rad(50)
snap = construction_at_theta(th, a=A, e=E, mu=MU)
ax.scatter([snap.rx], [snap.ry], c="#1A1814", s=40, label="planet")
ax.plot([0, snap.rx], [0, snap.ry], ls=":", color="#5C574E", lw=1)
ax.set_aspect("equal")
ax.set_title("Position space (equal-area triangles)")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel("x")
ax.set_ylabel("y")

# Velocity space
ax = axes[1]
ax.plot(np.append(data["vx"], data["vx"][0]), np.append(data["vy"], data["vy"][0]),
        color="#5F6B45", lw=2, label="hodograph")
phi = np.linspace(0, 2 * np.pi, 200)
ax.plot(
    circ.center_vx + circ.radius * np.cos(phi),
    circ.center_vy + circ.radius * np.sin(phi),
    ls="--", color="#9A7340", label="velocity circle",
)
for i, t in enumerate(angle_tris):
    ax.plot([t.vx0, t.vx1], [t.vy0, t.vy1], color="#7A3E32", lw=2,
            label=r"equal-$\angle$ $|\Delta v|$" if i == 0 else None)
ax.scatter([0], [0], c="#1A1814", s=40, label="O")
ax.scatter([circ.center_vx], [circ.center_vy], c="#9A7340", s=50, marker="D", label="eccentric C")
ax.plot([0, snap.px], [0, snap.py], color="#5C574E", lw=1.5)
ax.scatter([snap.px], [snap.py], c="#1A1814", s=30)
ax.set_aspect("equal")
ax.set_title("Velocity space (hodograph)")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlabel(r"$v_x$")
ax.set_ylabel(r"$v_y$")

fig.suptitle("Elementary picture: equal areas <-> equal |Dv| chords <-> circle", y=1.02)
plt.tight_layout()
plt.show()

print(f"|PF1|+|PF2| = {snap.focus_sum:.6f}  (2a = {2*A:.6f})")
"""
    ),
    md(
        """### A.4 Eccentric point and $90^{\\circ}$ construction frames

The velocity origin $O$ is offset from circle center $C$. As $p$ moves, the geometric construction tracks the elliptical orbit (gardener’s definition $|PF_1|+|PF_2|=2a$).
"""
    ),
    code(
        """\
thetas = np.linspace(0, 2 * np.pi, 8, endpoint=False)
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(np.append(data["x"], data["x"][0]), np.append(data["y"], data["y"][0]),
        color="#5F6B45", lw=2)
ax.scatter([0], [0], c="#9A7340", s=80, marker="*")
ax.scatter([f2x], [f2y], c="#5C574E", s=50, marker="x")

rows = []
for th in thetas:
    s = construction_at_theta(float(th), a=A, e=E, mu=MU)
    ax.scatter([s.rx], [s.ry], c="#1A1814", s=25)
    ax.scatter([s.construct_x], [s.construct_y], c="#8A9470", s=20, marker="D", alpha=0.8)
    rows.append({
        "theta_deg": np.rad2deg(th) % 360,
        "focus_sum": s.focus_sum,
        "h": s.h,
        "vx": s.vx,
        "vy": s.vy,
    })

ax.set_aspect("equal")
ax.set_title("Construction points track the elliptical orbit")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.show()
print(pd.DataFrame(rows).round(4).to_string(index=False))
"""
    ),
    md(
        """### A.5 Elementary conclusion

Equal-area triangles encode Kepler II. Equal-angle $|\\Delta v|$ chords force a circular hodograph. The eccentric-point construction then yields $|PF_1|+|PF_2|=2a$ — an **ellipse** with the Sun at a focus.
"""
    ),
    md(
        """## Part B — Formal demonstration

### B.1 Circular hodograph from $d\\mathbf{v}/d\\theta$

$$
\\frac{d\\mathbf{v}}{d\\theta} = -\\frac{\\mu}{h}\\hat{\\mathbf{r}}.
$$

Only for $a\\propto 1/r^{2}$ does the $r^{2}$ in $dt/d\\theta$ cancel to leave a constant-magnitude turn — a circle in velocity space.
"""
    ),
    code(
        """\
cx, cy, rad, rms = fit_circle_2d(data["vx"], data["vy"])
diag = orbit_diagnostics(A, E, MU)
summary = pd.DataFrame([
    {"quantity": "hodograph RMS", "value": rms},
    {"quantity": "fitted R", "value": rad},
    {"quantity": "theory R", "value": circ.radius},
    {"quantity": "fitted cy", "value": cy},
    {"quantity": "theory cy", "value": circ.center_vy},
    {"quantity": "ang_mom CV", "value": diag["ang_mom_cv"]},
    {"quantity": "focus_sum error", "value": diag["focus_sum_error"]},
])
print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(5, 5))
phi = np.linspace(0, 2 * np.pi, 200)
ax.scatter(data["vx"], data["vy"], s=8, c="#5F6B45", alpha=0.7, label="samples")
ax.plot(circ.center_vx + circ.radius * np.cos(phi),
        circ.center_vy + circ.radius * np.sin(phi),
        "--", color="#9A7340", label="theory circle")
ax.scatter([0], [0], c="k", s=40, label="O")
ax.scatter([circ.center_vx], [circ.center_vy], c="#9A7340", marker="D", s=50, label="C")
ax.set_aspect("equal")
ax.set_title("Circle fit to velocity tips")
ax.legend(fontsize=8)
ax.set_xlabel(r"$v_x$")
ax.set_ylabel(r"$v_y$")
plt.show()
"""
    ),
    md(
        """### B.2 Runge–Lenz vector ⇒ focused conic

$$
\\mathbf{A} = \\mathbf{v}\\times\\mathbf{h} - \\mu\\,\\hat{\\mathbf{r}},
\\qquad \\dot{\\mathbf{A}}=\\mathbf{0}
\\quad\\Rightarrow\\quad
r = \\frac{h^{2}/\\mu}{1+e\\cos\\theta}.
$$

Bound orbits ($e<1$) are ellipses.
"""
    ),
    code(
        """\
x, y, vx, vy = data["x"], data["y"], data["vx"], data["vy"]
r = np.hypot(x, y)
hx = x * vy - y * vx
Ax = vy * hx - MU * (x / r)
Ay = -vx * hx - MU * (y / r)
A_mag = np.hypot(Ax, Ay)

rl = pd.DataFrame({
    "theta_deg": np.rad2deg(data["theta"]),
    "A_x": Ax,
    "A_y": Ay,
    "|A|": A_mag,
    "e_est": A_mag / MU,
})
print(rl.iloc[::50].round(5).to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(data["theta"], A_mag, color="#5F6B45")
axes[0].axhline(E * MU, color="#9A7340", ls="--", label=rf"$e\\mu={E*MU:.3f}$")
axes[0].set_xlabel(r"$\\theta$")
axes[0].set_ylabel(r"$|A|$")
axes[0].set_title("Runge–Lenz magnitude (conserved)")
axes[0].legend()

axes[1].plot(Ax, Ay, color="#5F6B45")
axes[1].scatter([Ax.mean()], [Ay.mean()], c="#9A7340", s=60, zorder=5)
axes[1].set_aspect("equal")
axes[1].set_title(r"$A$ tip stays fixed (periapsis on $+x$)")
axes[1].set_xlabel(r"$A_x$")
axes[1].set_ylabel(r"$A_y$")
plt.tight_layout()
plt.show()

print(f"|A| mean={A_mag.mean():.6f}, std={A_mag.std():.3e}")
print(f"e from |A|/mu = {A_mag.mean()/MU:.6f}  (input e={E})")
"""
    ),
    md(
        """### B.3 Optional Binet sketch

With $u=1/r$,
$$
\\frac{d^{2}u}{d\\theta^{2}}+u=\\frac{\\mu}{h^{2}},
$$
solved by $u=(\\mu/h^{2})(1+e\\cos\\theta)$ — the same focused conic.
"""
    ),
    code(
        """\
# Compare polar r(θ) to Binet/closed form
th = data["theta"]
h = diag["h_theory"]
r_binet = (h * h / MU) / (1 + E * np.cos(th))
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(th, data["r"], color="#5F6B45", lw=2, label="orbit sampler")
ax.plot(th, r_binet, "--", color="#9A7340", label="Binet / conic formula")
ax.set_xlabel(r"$\\theta$")
ax.set_ylabel("r")
ax.set_title("Polar radius vs closed-form conic")
ax.legend()
plt.show()
print("max |r - r_binet| =", np.max(np.abs(data["r"] - r_binet)))

for k, v in diag.items():
    print(f"{k:24s} {v}")
"""
    ),
    md(
        """## Pointers

| Resource | Path |
|----------|------|
| Elementary write-up | `docs/celestial/feynman_elementary.md` |
| Formal write-up | `docs/celestial/feynman_formal.md` |
| Bibliography | `docs/bibliography.md` |
| Streamlit lab | sidebar → `planetary_motion` |
| Implementation | `src/orbit_feynman.py` |

**Sources.** Goodstein & Goodstein, *Feynman’s Lost Lecture*; Maxwell, *Matter and Motion*; Newton, *Principia*; Hall & Higson notes; Cariñena–Rañada–Santander (2016).
"""
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print(f"Wrote {OUT}")
