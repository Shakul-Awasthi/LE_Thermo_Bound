#!/usr/bin/env python3
"""fig1_window.py -- Fig. 1 from fig1_data.npz: (a) chaotic attractor coloured by lambda_H,
(b) leading exponent along k+5, (c) window sqrt(sigma_ps B_1) and departure |lambda_1 + Delta_1|."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import TwoSlopeNorm
from matplotlib.collections import LineCollection
from matplotlib.ticker import NullFormatter

plt.rcParams.update({
    "text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"],
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "text.latex.preamble": r"\usepackage{amsmath,bm}",
})
C_FP, C_CY, C_CH = "#0072B2", "#009E73", "#D55E00"   # Okabe-Ito
C_BAND = "#9ecae1"

D = np.load("fig1_data.npz")
k5, lamK, DelK, BK, spsK, amp, spread = (D[q] for q in ("k5", "lam", "Delta", "B", "sps", "amp", "spread"))
# classification from the converged block exponents: chaos = positive exponent resolved against its
# block spread, cycle = exponent zero within 5e-4 with finite amplitude, fixed = negative, negligible spread
clsK = np.where((lamK > 1e-3) & (spread < 0.5 * lamK), "chaos",
        np.where((np.abs(lamK) < 5e-4) & (amp > 1e-3), "cycle",
         np.where((lamK < 0) & (spread < 1e-3), "fixed", "unresolved")))
ok = (clsK != "unresolved") & (spsK > 1e-3)   # drop unresolved points and the low-concentration state beyond the crisis
k5, clsK, lamK, DelK, BK, spsK = (v[ok] for v in (k5, clsK, lamK, DelK, BK, spsK))
wK = np.sqrt(spsK * BK)
print("regimes:", {c: int((clsK == c).sum()) for c in ("fixed", "cycle", "chaos")},
      " max departure/window: %.2f" % np.max(np.abs(lamK + DelK) / wK))

# Layout in inches. Panel (c) is 40% shorter (0.75 x 0.80) than in the original 3.4 x 4.25 in figure; the margins,
# panels (a)-(b), and the gap between the rows keep their original sizes, so the figure is shorter.
C_SCALE = 0.60                               # height of panel (c) relative to the original
H0 = 4.25                                    # original figure height (in)
_u = (0.90 - 0.095) * H0 / (0.73 + 1.16 + 0.45 * 0.5 * (0.73 + 1.16))   # original row unit (in)
H_TOP, H_C, H_GAP = 0.73 * _u, 1.16 * _u * C_SCALE, 0.45 * 0.5 * (0.73 + 1.16) * _u
M_TOP, M_BOT = (1 - 0.90) * H0, 0.095 * H0
FIG_H = M_TOP + H_TOP + H_GAP + H_C + M_BOT
def fy(inch_from_top):
    """figure-fraction y coordinate of a point given in inches below the top edge"""
    return 1 - inch_from_top / FIG_H
fig = plt.figure(figsize=(3.4, FIG_H))
gs = gridspec.GridSpec(2, 1, height_ratios=[H_TOP, H_C], hspace=H_GAP / (0.5 * (H_TOP + H_C)),
                       left=0.15, right=0.90, bottom=M_BOT / FIG_H, top=1 - M_TOP / FIG_H)
gtop = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], width_ratios=[0.95, 1.0], wspace=0.6)

# ---- (a) attractor in the coordinates u = 2 sqrt(x), coloured by lambda_H ----------------
ax = fig.add_subplot(gtop[0])
U = 2 * np.sqrt(D["trace_x"]); lamH = D["trace_lamH"]
i0 = int(0.3 * len(U))
pts = U[i0:, :2]
vmax = np.percentile(np.abs(lamH[i0:]), 97)
lc = LineCollection(np.stack([pts[:-1], pts[1:]], axis=1), cmap="RdBu_r",
                    norm=TwoSlopeNorm(vmin=-vmax, vcenter=0., vmax=vmax), linewidth=0.4, rasterized=True)
lc.set_array(lamH[i0:-1])
ax.add_collection(lc)
ax.set_xlim(pts[:, 0].min() - 0.3, pts[:, 0].max() + 0.3); ax.set_ylim(pts[:, 1].min() - 0.3, pts[:, 1].max() + 0.3)
ax.set_xlabel(r"$2\sqrt{x}$", labelpad=1); ax.set_ylabel(r"$2\sqrt{y}$", labelpad=1)
ax.set_aspect("equal"); ax.set_xticks([5, 10, 15]); ax.set_yticks([5, 10, 15])
cax = ax.inset_axes([0.0, 1.07, 1.0, 0.07])
cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
cb.set_ticks([-0.5, 0, 0.5]); cb.ax.tick_params(labelsize=6.5, length=2, labeltop=True, labelbottom=False, top=True, bottom=False)
cax.text(1.04, 0.5, r"$\lambda_H$", transform=cax.transAxes, fontsize=8, ha="left", va="center")
fig.text(0.02, fy(0.025 * H0), r"(a)", fontsize=9, ha="left", va="top")
ax.plot([0.92], [0.92], transform=ax.transAxes, marker="*", ms=8, mfc="gold", mec="k", mew=0.6, ls="none", clip_on=False)
ax.tick_params(length=2.5, labelsize=7)

# ---- three k+5 segments of equal width: before the Hopf point, before chaos, the chaotic window
def tint(col, a=0.12):
    """opaque colour equal to col drawn at opacity a over white (no transparency in the PDF)"""
    rgb = np.array(matplotlib.colors.to_rgb(col))
    return tuple(1 - a * (1 - rgb))


def shade(axis, xx, cc):
    """one opaque span per run of equal class, so no seams between adjacent rectangles"""
    edges = np.concatenate([[xx[0]], 0.5 * (xx[1:] + xx[:-1]), [xx[-1]]])
    i = 0
    while i < len(cc):
        j = i
        while j + 1 < len(cc) and cc[j + 1] == cc[i]:
            j += 1
        axis.axvspan(edges[i], edges[j + 1], color=tint({"fixed": C_FP, "cycle": C_CY, "chaos": C_CH}[cc[i]]),
                     lw=0, zorder=0)
        i = j + 1

i_hopf = int(np.argmax(clsK == "cycle")); k_hopf = 0.5 * (k5[i_hopf] + k5[i_hopf - 1])
i_ch = int(np.argmax(clsK == "chaos")); k_ch = 0.5 * (k5[i_ch] + k5[i_ch - 1])
W = 16.6 - k_ch
SEG = [(k_hopf - W, k_hopf, "fixed point", C_FP, [13.95, 14.05]),
       (k_ch - W, k_ch, "limit cycle", C_CY, [16.2, 16.3]),
       (k_ch, 16.6, "chaos", C_CH, [16.45, 16.55])]
print("Hopf %.4f  chaos onset %.4f  width %.4f" % (k_hopf, k_ch, W))
K_STAR = 16.5
i_star = int(np.argmin(np.abs(k5 - K_STAR)))
star = dict(marker="*", ms=8, mfc="gold", mec="k", mew=0.6, ls="none", zorder=10, clip_on=False)
kw_mark = dict(marker=[(-0.5, -1), (0.5, 1)], markersize=6, linestyle="none", color="k", mec="k", mew=0.7, clip_on=False)

def breakmarks(axl, axr):
    axl.plot([1, 1], [0, 1], transform=axl.transAxes, **kw_mark)
    axr.plot([0, 0], [0, 1], transform=axr.transAxes, **kw_mark)
    axl.spines["right"].set_visible(False); axr.spines["left"].set_visible(False)

# ---- (b) leading exponent -----------------------------------------------------------------
gsb = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gtop[1], width_ratios=[1, 0.16, 1, 1], wspace=0.0)
B = [fig.add_subplot(gsb[i]) for i in (0, 2, 3)]
for i, (lo, hi, name, col, ticks) in enumerate(SEG):
    m = (k5 >= lo - 1e-9) & (k5 <= hi + 1e-9); b = B[i]
    shade(b, k5[m], clsK[m])
    b.plot(k5[m], lamK[m], "-", color="k", lw=1.1)
    b.axhline(0, color="0.6", lw=0.5)
    b.set_xlim(lo, hi); b.set_ylim(-0.008, 0.024)
    b.set_xticks([0.5 * (lo + hi)]); b.set_xticklabels(["%.1f" % (0.5 * (lo + hi))], fontsize=6.5)
    b.tick_params(length=2.5, labelleft=(i == 0), left=(i == 0), labelsize=7)
B[0].set_ylabel(r"$\lambda_1$", labelpad=1); B[0].set_yticks([0, 0.01, 0.02]); B[0].set_yticklabels(["0", "0.01", "0.02"])
fig.text(0.53, fy(0.025 * H0), r"(b)", fontsize=9, ha="left", va="top")
bb0 = B[0].get_position(); bb2 = B[2].get_position()
fig.text(0.5 * (bb0.x0 + bb2.x1), bb0.y0 - 0.052 * H0 / FIG_H, r"$k_{+5}$", ha="center", va="top", fontsize=9)
B[2].plot([K_STAR], [lamK[i_star]], **star)
breakmarks(B[0], B[1])
B[1].spines["right"].set_visible(False); B[2].spines["left"].set_visible(False)
B[1].axvline(k_ch, color="0.4", lw=0.6, ls="--")

# ---- (c) window and departure -------------------------------------------------------------
gsc = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], width_ratios=[1, 0.14, 1, 1], wspace=0.0)
C = [fig.add_subplot(gsc[i]) for i in (0, 2, 3)]
for i, (lo, hi, name, col, ticks) in enumerate(SEG):
    m = (k5 >= lo - 1e-9) & (k5 <= hi + 1e-9); c = C[i]
    shade(c, k5[m], clsK[m])
    c.plot(k5[m], wK[m], "-", color="#3182bd", lw=1.2)
    c.fill_between(k5[m], 1e-9, wK[m], color=C_BAND, alpha=0.45, lw=0)
    c.plot(k5[m], np.abs(lamK[m] + DelK[m]), "-", color="k", lw=1.2)
    c.set_yscale("log"); c.set_ylim(0.2, 3); c.set_xlim(lo, hi); c.set_xticks(ticks)
    c.yaxis.set_minor_formatter(NullFormatter()); c.tick_params(which="minor", left=False)
    c.tick_params(length=2.5, labelleft=(i == 0), left=(i == 0), labelsize=7)
    c.set_xticklabels([("%g" % t) for t in ticks], fontsize=7)
    c.text(0.5, 1.03, name, transform=c.transAxes, color=col, fontsize=7.5, ha="center", va="bottom")
C[0].set_ylabel(r"rate", labelpad=1); C[0].set_yticks([0.3, 1, 3]); C[0].set_yticklabels(["0.3", "1", "3"])
C[0].text(-0.3, 1.03, r"(c)", transform=C[0].transAxes, fontsize=9, ha="left", va="bottom")
C[1].text(0.5, 0.80, r"$\sqrt{\overline\sigma_{\rm ps}\overline B_1}$", transform=C[1].transAxes, fontsize=8, color="#3182bd", ha="center")
C[1].text(0.5, 0.22, r"$|\lambda_1+\overline\Delta_1|$", transform=C[1].transAxes, fontsize=8, ha="center")
cc0 = C[0].get_position(); cc2 = C[2].get_position()
fig.text(0.5 * (cc0.x0 + cc2.x1), cc0.y0 - 0.062 * H0 / FIG_H, r"$k_{+5}$", ha="center", va="top", fontsize=9)
C[2].plot([K_STAR], [wK[i_star]], **star); C[2].plot([K_STAR], [abs(lamK[i_star] + DelK[i_star])], **star)
breakmarks(C[0], C[1])
C[1].spines["right"].set_visible(False); C[2].spines["left"].set_visible(False)
C[1].axvline(k_ch, color="0.4", lw=0.6, ls="--")

fig.savefig("fig1_window.pdf", dpi=300)
fig.savefig("fig1_window.png", dpi=200)
print("saved fig1_window.pdf, fig1_window.png")
