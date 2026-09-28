#!/usr/bin/env python3
"""
fp_bosc_desktop.py  --  recompute and plot the fixed-point frequency bound from
the exported CSV.  Standalone: needs only numpy, scipy and matplotlib.

For every row of fixed_points.csv (rate constants, Gamma, fixed point) the
script refines the fixed point by Newton iteration, recomputes the Jacobian,
its spectrum, sigma_ps, B_osc and B_osc_max, checks the frequency bound, and
draws the figure:
  (a) stable focus of the ideal WR network (reference rates, drive 0.8) in 3D,
      trajectories starting in and off the invariant plane of the leading pair;
  (b) max|Im zeta|^2 against sigma_ps B_osc_max, ideal and nonideal in random
      order, bound y = x at 45 degrees, star at the panel-(a) fixed point.

    python3 fp_bosc_desktop.py --csv fixed_points.csv --out fp_bosc_final

Figure style is collected in the STYLE block below for easy changes.
"""
import argparse
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import sqrtm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "Liberation Serif",
                   "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 1.0,
})
from matplotlib.lines import Line2D

# ------------------------------------------------------------------ STYLE
STYLE = dict(
    figsize=(8.0, 4.3),                  # width matches the reference figures (8 in)
    fs_tick=20, fs_label=22, fs_legend=17, fs_panel=22, fs_note=18,
    # panel positions [left, bottom, width, height] in figure fractions
    pos_a=[0.07, 0.0, 0.44, 1.0], pos_b=[0.66, 0.165, 0.305, 0.74],
    col_ideal="#3b6ea8", col_nonideal="#e08a2c", col_star="#c0392b",
    marker_size=1.5, alpha=0.35,
    xlim=(1e-1, 1e3), ylim=(1e-1, 1e3),   # equal decades: square panel, bound at 45 deg
    view=(22, -125),                      # 3D elevation, azimuth
)

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="fixed_points.csv")
ap.add_argument("--out", default="fp_bosc_final")
ap.add_argument("--no-trails", action="store_true", help="drop rows with source == 1")
ap.add_argument("--seed", type=int, default=1, help="random drawing order")
args = ap.parse_args()

# ------------------------------------------------------------------ network
NU_P = np.array([[1., 1., 0., 1., 0.], [0., 1., 1., 0., 0.], [0., 0., 0., 1., 1.]])
NU_M = np.array([[2., 0., 0., 0., 0.], [0., 2., 0., 0., 0.], [0., 0., 0., 0., 2.]])
S = NU_M - NU_P
NBAR = 0.5 * (NU_P + NU_M)
K_REF = np.array([30., .25, 1., 1e-4, 10., 1e-3, 1., 1e-4, 16.5, .5]) / 30.0


# ------------------------------------------------------------------ batched kernels
# shapes: x (n,3), k (n,10), G (n,3,3)
def fluxes(x, k, G):
    lna = np.log(x) + np.einsum("nij,nj->ni", G, x)
    jp = k[:, 0::2] * np.exp(lna @ NU_P)
    jm = k[:, 1::2] * np.exp(lna @ NU_M)
    return jp, jm


def metric(x, G):
    return G + np.einsum("ni,ij->nij", 1.0 / x, np.eye(3))


def jacobian(x, k, G):
    jp, jm = fluxes(x, k, G)
    A = jp[:, :, None] * NU_P.T[None] - jm[:, :, None] * NU_M.T[None]   # (n,5,3)
    return np.einsum("ir,nrj,njk->nik", S, A, metric(x, G))


def newton(x, k, G, iters=30):
    y = np.log(x)
    for _ in range(iters):
        xx = np.exp(y)
        jp, jm = fluxes(xx, k, G)
        F = (jp - jm) @ S.T
        Jy = jacobian(xx, k, G) * xx[:, None, :]
        dy = np.linalg.solve(Jy, -F[..., None])[..., 0]
        dy = np.clip(dy, -1.0, 1.0)
        y = y + dy
        if np.max(np.abs(dy)) < 1e-14:
            break
    xx = np.exp(y)
    jp, jm = fluxes(xx, k, G)
    res = np.max(np.abs((jp - jm) @ S.T), axis=1) / np.max(jp + jm, axis=1)
    return xx, res


def bound_quantities(x, k, G):
    """sigma_ps, B_osc, B_osc_max, max|Im zeta|^2 and the leading complex pair."""
    jp, jm = fluxes(x, k, G)
    j, t = jp - jm, jp + jm
    sigma_ps = np.sum(2 * j ** 2 / t, axis=1)
    H = metric(x, G)
    J = jacobian(x, k, G)
    w, V = np.linalg.eig(J)
    max_im2 = np.max(w.imag ** 2, axis=1)
    SHS = np.einsum("ir,nij,jr->nr", S, H, S)
    NHN = np.einsum("ir,nij,jr->nr", NBAR, H, NBAR)
    SHN = np.einsum("ir,nij,jr->nr", S, H, NBAR)
    B_max = 0.125 * np.sum(t * (SHS * NHN - SHN ** 2), axis=1)
    # leading eigenvalue with positive imaginary part
    score = np.where(w.imag > 1e-12 * (1 + np.abs(w)), w.real, -np.inf)
    m = np.argmax(score, axis=1)
    n = np.arange(len(x))
    lam = w[n, m]
    u = V[n, :, m]
    Hu = np.einsum("nij,nj->ni", H, u)
    nrm = np.real(np.sum(np.conj(u) * Hu, axis=1))
    uHS = np.conj(Hu) @ S
    pu = np.einsum("ni,nir->nr", u, np.einsum("nij,jr->nir", H, NBAR))
    drift = -(np.abs(u) ** 2 / x ** 2) @ S
    g = (uHS * pu + 0.5 * drift) / nrm[:, None]
    B_osc = 0.5 * np.sum(t * g.imag ** 2, axis=1)
    has_pair = np.isfinite(score[n, m])
    return dict(sigma_ps=sigma_ps, B_osc=B_osc, B_max=B_max, max_im2=max_im2,
                lam=lam.real, beta=lam.imag, has_pair=has_pair)


# ------------------------------------------------------------------ read CSV
header = open(args.csv).readline().strip().split(",")
data = np.loadtxt(args.csv, delimiter=",", skiprows=1, ndmin=2)
c = {h: i for i, h in enumerate(header)}
if args.no_trails:
    data = data[data[:, c["source"]] == 0]
model = data[:, c["model"]]
k = data[:, [c[f"k{d}{r}"] for r in range(1, 6) for d in ("p", "m")]]
g6 = data[:, [c[n] for n in ("G11", "G12", "G13", "G22", "G23", "G33")]]
G = np.zeros((len(data), 3, 3))
for (i, j), col in zip([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)], g6.T):
    G[:, i, j] = G[:, j, i] = col
x0 = data[:, [c["x"], c["y"], c["z"]]]
print(f"{len(data)} fixed points read (ideal {np.sum(model == 0)}, "
      f"nonideal {np.sum(model == 1)}, trail {np.sum(data[:, c['source']] == 1)})")

# ------------------------------------------------------------------ recompute
x, res = newton(x0, k, G)
q = bound_quantities(x, k, G)
xb = q["sigma_ps"] * q["B_max"]
yb = q["max_im2"]
print(f"max fixed-point residual {res.max():.1e}, "
      f"max relative shift of the fixed point {np.max(np.abs(x / x0 - 1)):.1e}")
dx = np.abs(xb / data[:, c["xb"]] - 1)
dy = np.abs(yb / data[:, c["yb"]] - 1)
print(f"agreement with the exported values: max rel. diff {dx.max():.1e} (x), {dy.max():.1e} (y)")
print(f"complex pair present: {q['has_pair'].sum()} of {len(x)}")
print(f"mode bound  beta^2 <= sigma_ps B_osc      violations: "
      f"{np.sum(q['beta'][q['has_pair']]**2 > (q['sigma_ps']*q['B_osc'])[q['has_pair']]*(1+1e-9))}")
print(f"spectrum    max|Im|^2 <= sigma_ps B_osc_max violations: {np.sum(yb > xb*(1+1e-9))}")

# ------------------------------------------------------------------ panel (a)
def drive_k(k0, d):
    kk = k0.copy()
    a1 = np.log(kk[0] * kk[2] * kk[4] / (kk[1] * kk[3] * kk[5]))
    a2 = np.log(kk[0] * kk[6] * kk[8] / (kk[1] * kk[7] * kk[9]))
    kk[[3, 5]] *= np.exp(.5 * a1 * (1 - d))
    kk[[7, 9]] *= np.exp(.5 * a2 * (1 - d))
    return kk


G0 = np.zeros((1, 3, 3))
ka = drive_k(K_REF, 0.0)
xs = np.exp(np.linalg.lstsq(S.T, np.log(ka[0::2] / ka[1::2]), rcond=None)[0])[None]
for d in np.linspace(0.0, 0.80, 400):
    xs, _ = newton(xs, drive_k(K_REF, d)[None], G0, iters=50)
ka = drive_k(K_REF, 0.80)[None]
qa = bound_quantities(xs, ka, G0)
pa_x, pa_y = (qa["sigma_ps"] * qa["B_max"])[0], qa["max_im2"][0]
xs = xs[0]
Hh = np.real(sqrtm(metric(xs[None], G0)[0]))
Hmh = np.linalg.inv(Hh)
ev, V = np.linalg.eig(jacobian(xs[None], ka, G0)[0])
m = int(np.argmax(np.where(ev.imag > 0, ev.real, -np.inf)))
lam1, beta1, u = ev[m].real, ev[m].imag, V[:, m]
gamma1 = -lam1
n1 = beta1 / (2 * np.pi * gamma1)
print(f"panel (a): lambda_1 = {lam1:.4f}, beta_1 = {beta1:.4f}, n_1 = {n1:.2f}, "
      f"point ({pa_x:.4f}, {pa_y:.4f})")
a_vec, b_vec = np.real(Hh @ u), np.imag(Hh @ u)
e1 = a_vec / np.linalg.norm(a_vec)
e2 = b_vec - (b_vec @ e1) * e1
e2 /= np.linalg.norm(e2)
R3 = np.vstack([e1, e2, np.cross(e1, e2)])


def rhs(t, xx):
    jp, jm = fluxes(xx[None], ka, G0)
    return S @ (jp - jm)[0]


EPS, T_end = 1e-3, 3.0 / gamma1
ts = np.linspace(0, T_end, 4000)
xa0 = xs + EPS * np.linalg.norm(xs) * np.real(u) / np.linalg.norm(np.real(u))
sa = solve_ivp(rhs, (0, T_end), xa0, method="LSODA", rtol=1e-11, atol=1e-14, dense_output=True)
dA = Hh @ (sa.sol(ts) - xs[:, None])
scale = np.linalg.norm(dA[:, 0])
u3 = np.real(V[:, int(np.argmin(np.abs(ev.imag)))])
w_plane, w_fast = Hh @ np.real(u), Hh @ u3
w_fast *= np.linalg.norm(w_plane) / np.linalg.norm(w_fast)
xb0 = xs + EPS * np.linalg.norm(xs) / np.linalg.norm(np.real(u)) * (Hmh @ (w_plane + w_fast))
sb = solve_ivp(rhs, (0, T_end), xb0, method="LSODA", rtol=1e-11, atol=1e-14, dense_output=True)
Pa = R3 @ dA / scale
Pb = R3 @ (Hh @ (sb.sol(ts) - xs[:, None])) / scale
fast_dir = R3 @ (Hh @ u3)
fast_dir /= np.linalg.norm(fast_dir)

# ------------------------------------------------------------------ figure
FT, FL, FG, FP, FN = (STYLE[k] for k in ("fs_tick", "fs_label", "fs_legend",
                                        "fs_panel", "fs_note"))
fig = plt.figure(figsize=STYLE["figsize"])

# ---------- panel (a): 3D
ax = fig.add_axes(STYLE["pos_a"], projection="3d")
rr, pp = np.meshgrid(np.linspace(0, 1.15, 12), np.linspace(0, 2 * np.pi, 48))
ax.plot_surface(rr * np.cos(pp), rr * np.sin(pp), 0 * rr, color="0.70", alpha=0.45,
                linewidth=0, shade=False)
for axis in (ax.xaxis, ax.yaxis, ax.zaxis):      # white background panes
    axis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    axis._axinfo["grid"].update(color="0.88", linewidth=0.6)
th = np.linspace(0, 2 * np.pi, 400)
ell = np.real(np.exp(1j * th)[None, :] * (Hh @ u)[:, None])
ell_p = np.vstack([e1 @ ell, e2 @ ell]) / np.linalg.norm(a_vec)
for mm in range(4):
    ax.plot(np.exp(-mm) * ell_p[0], np.exp(-mm) * ell_p[1], 0 * ell_p[0], "k--", lw=0.7)
ax.plot(*Pa, lw=1.2, color=STYLE["col_ideal"], label="start in plane")
ax.plot(*Pb, lw=1.2, color=STYLE["col_star"], label="start off plane")
ax.plot(*Pa[:, :1], "o", ms=5, color=STYLE["col_ideal"])
ax.plot(*Pb[:, :1], "o", ms=5, color=STYLE["col_star"])
ax.quiver(0, 0, 0, *fast_dir, color="k", lw=1.2, arrow_length_ratio=0.12)
ax.text(*(1.1 * fast_dir), r"$H^{1/2}u_3$", fontsize=FN)
ax.set_xticks([-1, 0, 1]); ax.set_yticks([-1, 0, 1])
ax.set_zticks([np.round(Pb[2].min(), 1), 0.0])
ax.set_xlabel(r"$e_1\cdot\delta_H$", fontsize=FL, labelpad=6)
ax.set_ylabel(r"$e_2\cdot\delta_H$", fontsize=FL, labelpad=6)
# z label drawn as figure text: with this view the 3D label falls outside the figure
fig.text(0.0, 0.62, r"$e_3\cdot\delta_H$", fontsize=FL, rotation=90, va="center")
ax.tick_params(labelsize=FT - 4, pad=0)
ax.view_init(*STYLE["view"])
ax.set_box_aspect((1, 1, 0.6))
ax.legend(fontsize=FG - 2, frameon=False, loc="upper left", bbox_to_anchor=(0.10, 0.99),
          handlelength=1.4, borderaxespad=0)
ax.text2D(0.99, 0.84, rf"$n_1={n1:.2f}$", transform=ax.transAxes, ha="right", va="top",
          fontsize=FN)
fig.text(0.005, 0.965, "(a)", fontsize=FP, fontweight="bold", va="top")

# ---------- panel (b): bound test
axb = fig.add_axes(STYLE["pos_b"])
rng = np.random.default_rng(args.seed)
ok = (xb > 0) & (yb > 0)
idx = rng.permutation(np.flatnonzero(ok))
cols = np.where(model[idx] == 0, STYLE["col_ideal"], STYLE["col_nonideal"])
axb.scatter(xb[idx], yb[idx], s=STYLE["marker_size"], alpha=STYLE["alpha"], c=cols,
            edgecolors="none", rasterized=True)
(x0l, x1l), (y0l, y1l) = STYLE["xlim"], STYLE["ylim"]
lo, hi = min(x0l, y0l), max(x1l, y1l)
axb.plot([lo, hi], [lo, hi], "--", color="0.15", lw=2.0)
axb.scatter([pa_x], [pa_y], marker="*", s=260, color=STYLE["col_star"], edgecolors="k",
            linewidths=0.8, zorder=5)
axb.set_xscale("log"); axb.set_yscale("log")
axb.set_xlim(x0l, x1l); axb.set_ylim(y0l, y1l)
axb.set_aspect("equal", anchor="W")      # square box when both ranges span the same decades
if np.log10(x1l / x0l) != np.log10(y1l / y0l):
    print("note: x and y ranges differ, so panel (b) is not square")
axb.set_xticks([1e-1, 1e0, 1e1, 1e2, 1e3]); axb.set_yticks([1e-1, 1e0, 1e1, 1e2, 1e3])
handles = [Line2D([], [], ls="", marker="o", ms=8, color=STYLE["col_nonideal"], label="nonideal"),
           Line2D([], [], ls="", marker="o", ms=8, color=STYLE["col_ideal"], label="ideal"),
           Line2D([], [], ls="--", color="0.15", lw=2.0, label="bound"),
           Line2D([], [], ls="", marker="*", ms=15, color=STYLE["col_star"], mec="k", mew=0.8,
                  label="panel (a)")]
axb.legend(handles=handles, fontsize=FG, frameon=False, loc="upper left",
           handlelength=1.3, handletextpad=0.4, labelspacing=0.25, borderaxespad=0.2)
axb.set_xlabel(r"$\sigma_{\rm ps}B_{\rm osc}^{\max}$", fontsize=FL, labelpad=2)
axb.set_ylabel(r"$|{\rm Im}\,\zeta|^2_{\max}$", fontsize=FL, labelpad=2)
axb.tick_params(labelsize=FT, direction="out", length=5)
axb.tick_params(which="minor", length=3)
fig.text(STYLE["pos_b"][0] - 0.10, 0.965, "(b)", fontsize=FP, fontweight="bold", va="top")
fig.savefig(f"{args.out}.pdf", dpi=300)
fig.savefig(f"{args.out}.png", dpi=200)
print(f"saved {args.out}.pdf / .png")