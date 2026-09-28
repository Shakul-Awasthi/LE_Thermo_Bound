# Figure 1: instability window of the Willamowski–Rössler network

Minimal code that regenerates Fig. 1 of *Thermodynamic cost of dynamical
instability in deterministic chemical reaction networks* (S. Awasthi, J. S. Lee).

| file              | does                                                                 |
|-------------------|----------------------------------------------------------------------|
| `fig1_data.py`    | network, integrator and observables; k₊₅ scan and chaotic trace → `fig1_data.npz` |
| `fig1_window.py`  | reads `fig1_data.npz`, writes `fig1_window.pdf` and `fig1_window.png` |

## Requirements

Python 3.10+ with numpy, numba and matplotlib:

    pip install -r requirements.txt

The labels are rendered with LaTeX (`text.usetex`), so LaTeX with `amsmath`,
`bm`, cm-super fonts and `dvipng` is also needed
(Ubuntu: `sudo apt install texlive-latex-extra cm-super dvipng`; macOS: MacTeX).
Without LaTeX, set `"text.usetex": False` at the top of `fig1_window.py`.

## Run

    python3 fig1_data.py 4      # data; the argument is the number of CPU cores (default 1)
    python3 fig1_window.py      # figure

`fig1_data.py` takes about 5 minutes on one core (about 2.5 minutes on two).
It prints the number of scan points and trace samples, and `fig1_window.py`
prints the Hopf point (≈14.13), the onset of chaos (≈16.375) and the largest
ratio of departure to window.

## What is computed

- **Network** (chemostats absorbed): A₁+X ⇌ 2X, X+Y ⇌ 2Y, A₅+Y ⇌ A₂,
  X+Z ⇌ A₃, A₄+Z ⇌ 2Z, ideal mixing, H = diag(1/x). No reaction combination
  leaves all species, chemostats included, unchanged, so the rate constants
  are consistent with local detailed balance.
- **Rate constants** (k₊₁, k₋₁, …, k₊₅, k₋₅) = (30, 0.25, 1, 10⁻⁴, 10, 10⁻³,
  1, 10⁻⁴, 16.5, 0.5). Rates are reported in units of k₊₁ (divided by 30).
- **Scan (panels b, c).** k₊₅ from 13 to 16.6 (steps 0.05, then 0.01 above
  16.3), from (10, 10, 10). State and tangent are integrated with adaptive
  Dormand–Prince 5(4) (tolerances 10⁻⁸, 10⁻¹¹), the tangent renormalised in
  the thermodynamic norm after every step; transient 900 and window 3000
  reference time units in six blocks. Stored: λ₁, Δ̄₁, B̄₁, σ̄_ps, amplitude and
  block spread of λ₁.
- **Trace (panel a).** Attractor at k₊₅ = 16.5, RK4 with step 4×10⁻⁴,
  transient 400 and 400 recorded time units, with the instantaneous Hessian
  stretching rate λ_H along the aligned tangent.
- **Classification** (in `fig1_window.py`): chaos if λ₁ > 10⁻³ with block
  spread below λ₁/2, limit cycle if |λ₁| < 5×10⁻⁴ with finite amplitude, fixed
  point if λ₁ < 0 with negligible spread. Unresolved points and the
  low-concentration fixed point beyond the crisis at 16.6 are not plotted.
