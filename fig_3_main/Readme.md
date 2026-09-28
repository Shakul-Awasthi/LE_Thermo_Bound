# Fixed-point frequency bound: data and figure

## What `fixed_points.csv` contains

Each row is one interior fixed point of the reversible Willamowski–Rössler (WR)
network that has a complex eigenvalue pair and lies inside the plotted window
(0.1 ≤ σ_ps B_osc^max ≤ 10³, 10⁻² ≤ |Im ζ|²_max ≤ 10³).
There are 142,239 rows: 60,298 ideal and 81,941 nonideal, of which 111,496 come
from random sampling of the rate constants and 30,743 from continuation toward
the boundary where the complex pair is born.

| Column | Meaning |
|---|---|
| `source` | 0 = random ensemble, 1 = boundary continuation |
| `model` | 0 = ideal mixture, 1 = nonideal mixture |
| `kp1, km1, …, kp5, km5` | forward and backward rate constants of reactions 1–5, in units of k₊₁ = 1 |
| `G11 G12 G13 G22 G23 G33` | symmetric interaction matrix Γ (all zero for ideal mixtures) |
| `x, y, z` | fixed-point concentrations of X, Y, Z (identify the fixed point when a rate vector has several) |
| `xb, yb` | exported values of σ_ps B_osc^max and \|Im ζ\|²_max, kept only for checking |

Nonideal activities are a_i = x_i exp[(Γx)_i]. All numbers are stored at full
double precision.

## What `fp_bosc.py` does

1. Reads the CSV.
2. For every row, refines the fixed point by Newton iteration and recomputes the
   Jacobian, its spectrum, the pseudo entropy production σ_ps, and the rotation
   coefficients B_osc and B_osc^max.
3. Prints a check: fixed-point residual, agreement with the exported values, and
   the number of violations of β² ≤ σ_ps B_osc and |Im ζ|²_max ≤ σ_ps B_osc^max
   (both should be 0).
4. Draws the figure:
   - (a) a stable focus of the ideal WR network in 3D, with trajectories starting
     in and off the invariant plane of the oscillating mode;
   - (b) |Im ζ|²_max against σ_ps B_osc^max for all rows, ideal and nonideal in
     random order, with the bound as the 45° dashed line and a star at the fixed
     point of panel (a).

It needs only Python 3 with `numpy`, `scipy` and `matplotlib`, and takes a few
seconds.

## How to run

Put `fixed_points.csv` and `fp_bosc_desktop.py` in the same folder, then

```bash
pip install numpy scipy matplotlib        # once, if not installed
python3 fp_bosc_desktop.py --csv fixed_points.csv --out fp_bosc_final
```

This writes `fp_bosc_final.pdf` and `fp_bosc_final.png`.

Options:

- `--no-trails` uses only the random ensemble (`source` = 0).
- `--seed N` changes the random drawing order of the points.

Colours, marker size, transparency, axis limits, and the 3D viewing angle are set
in the `STYLE` block at the top of `fp_bosc_desktop.py`.