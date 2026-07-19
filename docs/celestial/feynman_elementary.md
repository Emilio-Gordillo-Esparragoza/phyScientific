# Feynman’s elementary demonstration

> “Elementary does not mean easy to understand… except to have an infinite amount of intelligence.” — R. P. Feynman (1964)

**Companion code:** `src/orbit_feynman.py` · **UI:** Streamlit lab `planetary_motion` → *Elementary demonstration* · **Notebook:** `notebooks/feynman_lost_lecture.ipynb` (Part A) · **Video:** <https://youtu.be/xdIjYBtnvZU>

## Goal

Prove **Kepler’s first law** (bound orbits are ellipses with the Sun at a focus) from:

1. Newton’s inverse-square law of gravitation, and  
2. Kepler’s second law (equal areas in equal times ⇔ central force ⇔ conserved angular momentum),

using **plane geometry** and a **velocity-space (hodograph)** diagram — no differential equations required (Goodstein & Goodstein, 1996; after Maxwell, 1877).

## Why this mathematics

Newton’s *Principia* already contains a geometric proof, but it uses conic properties that are opaque to modern readers. Feynman rebuilt a **hodograph** argument (in the spirit of Maxwell) so that a freshman audience could see *why* $1/r^{2}$ is special: equal angular steps produce equal $|\Delta\mathbf{v}|$, forcing a circle in velocity space; an off-center (“eccentric”) origin then generates an ellipse in position space.

## Step-by-step

### 1. Position space

Sun fixed at focus $F_1$. Planet at $\mathbf{r}(t)$ with velocity $\mathbf{v}=\dot{\mathbf{r}}$. The curve traced by the planet is the unknown orbit.

### 2. Velocity space (hodograph)

Plot every velocity vector with its tail at a common origin $O$. The tip $p$ traces the **hodograph**.

Central force $\Rightarrow$ $\Delta\mathbf{v}$ is always parallel to $-\hat{\mathbf{r}}$.

### 3. Inverse-square + equal areas ⇒ circular hodograph

- Acceleration magnitude $\propto 1/r^{2}$.  
- Equal areas $\Rightarrow$ time to sweep a fixed angle $\propto r^{2}$.  

**Triangles in the lecture video.** Partition the orbit into Sun–planet–planet triangles:

1. **Equal-area triangles** (Kepler II): same area $\Leftrightarrow$ same time. On an eccentric orbit their angular widths $\Delta\theta$ are *unequal* — wider near periapsis, skinnier near apoapsis.  
2. **Equal-angle wedges** (Feynman’s clock): same $\Delta\theta$. Their areas vary, but under inverse-square + conserved $h$, $|\Delta v|$ is the same — equal chord lengths on the velocity diagram.

Therefore, over equal orbital angles, $|\Delta\mathbf{v}|$ is **constant**. Equal chord lengths at successive equal angles imply that $p$ lies on a **circle**.

Analytically (for orientation; not required in the elementary path), with $h=|\mathbf{r}\times\mathbf{v}|$ and $\mu=GM$,

$$
v_x = -\sqrt{\frac{\mu}{p}}\sin\theta,\quad
v_y = \sqrt{\frac{\mu}{p}}(e+\cos\theta),\quad
p=a(1-e^{2}),
$$

so the tips lie on a circle of radius $\sqrt{\mu/p}$ centered at $(0,\,e\sqrt{\mu/p})$.

The Streamlit lab and notebook draw both partitions side by side, with tables of area and $|\Delta v|$ per sector (`equal_area_triangles` / `equal_angle_triangles` in `src/orbit_feynman.py`).

### 4. Eccentric point and $90^{\circ}$ rotation

Call the circle’s center $C$. The velocity origin $O$ is **offset** from $C$ — the **eccentric point**. Rotating the velocity diagram by $90^{\circ}$ aligns velocity-parallel lines with the tangent geometry of the orbit. The perpendicular bisector of $Op$, meeting the ray $Cp$, maps to a point $P$ in position space.

### 5. Finale

As $p$ runs around the circle, the constructed points $P$ satisfy the gardener’s definition

$$
|PF_1| + |PF_2| = 2a
$$

for a second focus $F_2$. Hence the orbit is an **ellipse** with the Sun at $F_1$.

## Assumptions

- Point-mass Newtonian gravity; planar motion  
- Bound orbit (the elementary lecture emphasizes the closed elliptical case)  
- Inverse-square specifically (other central forces yield other hodographs)

## Sources

Goodstein & Goodstein (1996); Feynman (1964); Maxwell (1877); Hall & Higson notes; Cariñena–Rañada–Santander (2016). See [bibliography.md](../bibliography.md).

**Formal companion:** [feynman_formal.md](feynman_formal.md).
