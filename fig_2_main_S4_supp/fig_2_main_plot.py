"""Figure 2 (main text): cost of sustained motion, built only from the CSV of fig_2_input_selected.py.

sigma_ps against Delta_1^2/B_1, both coordinates rebuilt from the separately computed terms.
  * Stable fixed points of all four networks: light gray density in the background, drawn first
    (darker = more fixed points). Both axes run from 1e-2 to 1e3 (--lim; --auto-lim fits them to the
    cycles and chaos); points outside that window are not drawn and are counted in the audit.
  * Limit cycles (blue open squares) and chaos candidates (vermillion filled triangles) of all
    four networks, coloured by class only (the per-network panels are in fig_2_supp_plt.py).
    They are selected at random on square cells of fixed width (0.2 decade) so that every cell that contains
    eligible cycles or chaos shows at least one point; exactly --per-class (1000) cycles and chaos are drawn, spread as evenly as possible; each network shows
    exactly as many ideal as nonideal points, and there are exactly as many limit cycles as chaos
    points (--no-equal-classes switches this off). Selected points are drawn in one random order.
  * Dashed diagonal: the motion cost sigma_ps = Delta_1^2/B_1. Inset: saturation
    Delta_1^2/(B_1 sigma_ps) of all cycles and chaos candidates in the CSV.

    python3 fig_2_main_plot.py --input data_fig2/fig_2_inputs.csv --output Figures
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix

MODELS = ('wr', 'brusselator', 'oregonator', 'cs14')
FIXED_SHADES = ['#cacaca', '#b0b0b0', '#8d8d8d']                  # gray density of the stable fixed points
CLASS_COLORS = {'cycle': '#0072B2', 'chaos_candidate': '#D55E00'}   # marker colour by class, not by network
MOVING = ('cycle', 'chaos_candidate')


# ---------------------------------------------------------------- CSV ------
COLS = ('point_id', 'model', 'nonideal', 'cls', 'status', 'selected', 'test_dataset', 'target_per_case',
        'lambda1', 'Delta1', 'B1', 'sigma_ps')


def load_columns(path):
    """Only the columns the plot needs (keeps memory at ~1-2 GB for 800,000 rows)."""
    import sys
    rows = []
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.reader(f); head = next(rd)
        ix = [(k, head.index(k)) for k in COLS if k in head]
        for line in rd:
            rows.append({k: (sys.intern(line[i]) if k in ('model', 'nonideal', 'cls', 'status', 'selected', 'test_dataset',
                                                         'target_per_case') else line[i]) for k, i in ix})
    return rows


def target_of(rows):
    t = {r.get('target_per_case') for r in rows}
    if len(t) != 1 or not str(next(iter(t))).isdigit():
        raise ValueError('the CSV does not state one target per case: %s' % t)
    return int(next(iter(t)))


def nonideal(r):
    v = str(r['nonideal']).lower()
    if v not in ('true', 'false', '1', '0'):
        raise ValueError('invalid nonideal flag: ' + v)
    return v in ('true', '1')


def read_rows(path, test_input=False):
    """Load and check the CSV: accepted rows only, finite positive terms, exact quota."""
    rows = load_columns(path)
    counts = {(m, n): 0 for m in MODELS for n in (False, True)}
    for r in rows:
        if r.get('model') not in MODELS or r.get('cls') not in ('fixed',) + MOVING:
            raise ValueError('unknown model or class in row ' + r.get('point_id', '?'))
        if str(r.get('status')).lower() != 'ok' or str(r.get('selected')).lower() != 'true':
            raise ValueError('unaccepted row ' + r['point_id'])
        if not test_input and str(r.get('test_dataset', 'False')).lower() == 'true':
            raise ValueError('a test CSV cannot be plotted as production data (use --test-input)')
        for t in ('lambda1', 'Delta1', 'B1', 'sigma_ps'):
            if not math.isfinite(float(r[t])):
                raise ValueError('nonfinite ' + t + ' in row ' + r['point_id'])
        if min(float(r['Delta1']), float(r['B1']), float(r['sigma_ps'])) <= 0:
            raise ValueError('nonpositive plotting term in row ' + r['point_id'])
        counts[r['model'], nonideal(r)] += 1
    target = target_of(rows)
    if not test_input and any(n != target for n in counts.values()):
        raise ValueError('the production CSV must hold exactly %d rows per model and mixing case: %s' % (target, counts))
    return rows, counts


def xy(r):
    return float(r['Delta1']) ** 2 / float(r['B1']), float(r['sigma_ps'])


# ---------------------------------------------------------- selection ------
def select_coloured(rows, lim, cell, cap, seed, time_limit, equal_classes=True, floor=1, per_class=0, level=0):
    """Random selection of cycles and chaos: floor..cap points in every occupied cell,
    equal ideal and nonideal counts for every network, (by default) as many limit cycles as
    chaos points, and as many points as possible within these rules. Cells are squares of fixed
    width `cell` decades anchored at integer decades, so the density control does not depend on
    the axis range."""
    lo, hi = lim
    buckets, outside = {}, 0
    for r in rows:
        if r['cls'] not in MOVING:
            continue
        x, y = xy(r)
        if not (lo <= x <= hi and lo <= y <= hi):
            outside += 1
            continue
        ix = int(math.floor(math.log10(x) / cell + 1e-9))
        iy = int(math.floor(math.log10(y) / cell + 1e-9))
        buckets.setdefault((ix, iy, r['model'], nonideal(r), r['cls']), []).append(r)
    missing = [m for m in MODELS if not any(k[2] == m and k[3] for k in buckets) or not any(k[2] == m and not k[3] for k in buckets)]
    if missing:
        raise ValueError('no cycles or chaos in both mixing cases for: ' + ', '.join(missing))
    if per_class:
        # exactly per_class cycles and per_class chaos points: quick necessary conditions first
        equal_classes = False
        tot, cyc, cha = {}, {}, {}
        for k, v in buckets.items():
            c = k[:2]; tot[c] = tot.get(c, 0) + len(v)
            (cyc if k[4] == 'cycle' else cha)[c] = (cyc if k[4] == 'cycle' else cha).get(c, 0) + len(v)
        if sum(cyc.values()) < per_class or sum(cha.values()) < per_class:
            raise ValueError('only %d limit cycles and %d chaos points lie inside the axes; use --per-class %d or less'
                             % (sum(cyc.values()), sum(cha.values()), min(sum(cyc.values()), sum(cha.values()))))
        if floor * len(tot) > 2 * per_class:
            raise ValueError('%d occupied cells cannot each show a point with only %d + %d points; use a larger '
                             '--cell-decades or --min-per-cell 0' % (len(tot), per_class, per_class))
        if (sum(min(cap, v) for v in cyc.values()) < per_class or sum(min(cap, v) for v in cha.values()) < per_class
                or sum(min(cap, v) for v in tot.values()) < 2 * per_class):
            raise ValueError('impossible: %d per cell cannot hold %d cycles and %d chaos points' % (cap, per_class, per_class))
    if equal_classes:
        # necessary condition: every cell holding only cycles needs >= floor cycles, and the chaos
        # points available (at most cap per cell) must match that many cycles
        cyc_cells = {k[:2] for k in buckets if k[4] == 'cycle'}; ch = {}
        for k, v in buckets.items():
            if k[4] == 'chaos_candidate':
                ch[k[:2]] = ch.get(k[:2], 0) + len(v)
        need = floor * len(cyc_cells - set(ch))
        can = lambda c: sum(min(c, v) for v in ch.values())
        if not ch or can(cap) < need:
            better = next((c for c in range(cap, 21) if ch and can(c) >= need), None)
            raise ValueError(
                'equal numbers of cycles and chaos are impossible with these rules: %d cells hold only cycles and '
                'need at least %d cycles, but at most %d chaos points fit (%d cells hold chaos, cap %d per cell). %s'
                % (len(cyc_cells - set(ch)), need, can(cap), len(ch), cap,
                   ('Use --max-per-cell %d or more, or --min-per-cell 0, or --no-equal-classes.' % better) if better
                   else 'Use --min-per-cell 0 (cells may then stay empty) or --no-equal-classes.'))
    rng = np.random.default_rng(seed)
    keys = list(buckets); keys = [keys[i] for i in rng.permutation(len(keys))]
    cells = sorted({k[:2] for k in keys}); ci = {c: i for i, c in enumerate(cells)}
    nc, nv, nm = len(cells), len(keys), len(MODELS)
    # rows: one per occupied cell (floor..cap), one balance row per network (nonideal - ideal = 0),
    # and one class row (cycles - chaos = 0)
    ri, cj, val = [], [], []
    for j, (ix, iy, m, non, cls) in enumerate(keys):
        ri += [ci[ix, iy], nc + MODELS.index(m)]; cj += [j, j]; val += [1, 1 if non else -1]
        if equal_classes:
            ri.append(nc + nm); cj.append(j); val.append(1 if cls == 'cycle' else -1)
        if per_class:                                   # row nc+nm: cycles, row nc+nm+1: chaos
            ri.append(nc + nm + (0 if cls == 'cycle' else 1)); cj.append(j); val.append(1)
    nr = nc + nm + (1 if equal_classes else 0) + (2 if per_class else 0)
    A = coo_matrix((val, (ri, cj)), shape=(nr, nv)).tocsc()
    cell_floor = np.full(nc, float(floor))
    if level:                     # even spreading: every cell shows at least `level` points, or all it has
        avail = np.zeros(nc)
        for k in keys:
            avail[ci[k[:2]]] += min(cap, len(buckets[k]))
        cell_floor = np.maximum(cell_floor, np.minimum(level, avail))
        if cell_floor.sum() > 2 * per_class:
            raise ValueError('infeasible: level %d needs %d points' % (level, cell_floor.sum()))
    lower = np.r_[cell_floor, np.zeros(nm), np.full(nr - nc - nm, per_class)]
    upper = np.r_[np.full(nc, cap), np.zeros(nm), np.full(nr - nc - nm, per_class)]
    ub = np.array([min(cap, len(buckets[k])) for k in keys], float)
    if per_class:                                   # fixed totals: a random objective makes the choice random
        cost = rng.random(nv)
    else:                                           # maximise the number of points; random tie-break below one point
        cost = -np.ones(nv) + (rng.random(nv) - 0.5) * 0.25 / max(1, nv * cap)
    res = milp(cost, integrality=np.ones(nv), bounds=Bounds(np.zeros(nv), ub),
               constraints=LinearConstraint(A, lower, upper), options={'time_limit': time_limit})
    if res.x is None:
        raise ValueError('no selection satisfies %d..%d points per occupied cell, ideal/nonideal balance%s: %s'
                         % (floor, cap, ' and %d cycles + %d chaos' % (per_class, per_class) if per_class else
                            ' and equal numbers of cycles and chaos' if equal_classes else '', res.message))
    n = np.rint(res.x).astype(int); tot = A @ n
    if np.any(tot < lower - 1e-9) or np.any(tot > upper + 1e-9) or np.any(n < 0) or np.any(n > ub):
        raise ValueError('integer allocation failed verification')
    chosen, per_cell = [], {}
    for k, c in zip(keys, n):
        for i in rng.choice(len(buckets[k]), int(c), replace=False):
            chosen.append(buckets[k][i])
        per_cell[k[:2]] = per_cell.get(k[:2], 0) + int(c)
    chosen = [chosen[i] for i in rng.permutation(len(chosen))]          # one random drawing order
    balance = {m: dict(ideal=sum(r['model'] == m and not nonideal(r) for r in chosen),
                       nonideal=sum(r['model'] == m and nonideal(r) for r in chosen)) for m in MODELS}
    classes = {c: sum(r['cls'] == c for r in chosen) for c in MOVING}
    assert all(b['ideal'] == b['nonideal'] for b in balance.values())
    assert not equal_classes or classes['cycle'] == classes['chaos_candidate']
    assert not per_class or classes['cycle'] == classes['chaos_candidate'] == per_class
    shown = [v for v in per_cell.values() if v > 0]
    assert min(per_cell.values()) >= floor and max(per_cell.values()) <= cap
    report = dict(seed=seed, cell_decades=cell, limits=list(lim), cap=cap, min_per_cell=floor, even_level=level,
                  occupied_cells=nc,
                  per_class_requested=per_class,
                  cells_with_points=len(shown), coloured_points=len(chosen), per_class=classes,
                  equal_classes=equal_classes, per_cell_min=min(per_cell.values()),
                  per_cell_max=max(per_cell.values()), per_model=balance,
                  moving_outside_window=outside, solver=res.message)
    cells_csv = [dict(cell_x=c[0], cell_y=c[1], log10_x_from=round(c[0] * cell, 6), log10_y_from=round(c[1] * cell, 6),
                      plotted=v) for c, v in sorted(per_cell.items())]
    return chosen, report, cells_csv


# --------------------------------------------------------------- main ------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data_fig2/fig_2_inputs.csv')
    p.add_argument('--output', default='Figures')
    p.add_argument('--audit-dir', default='data_fig2/main_selection')
    p.add_argument('--lim', type=float, nargs=2, default=(1e-2, 1e3),
                   help='axis range of both axes (default 1e-2 1e3); points outside are counted in the audit')
    p.add_argument('--auto-lim', action='store_true',
                   help='instead: the decades holding every cycle and chaos point, plus --margin-decades')
    p.add_argument('--margin-decades', type=float, default=1., help='margin around the cycles and chaos (default 1 decade)')
    p.add_argument('--fixed-bins', type=int, default=200, help='bins per axis of the gray fixed-point density')
    p.add_argument('--cell-decades', type=float, default=0.2,
                   help='width of the square density-control cells in decades (fixed, independent of the axes)')
    p.add_argument('--max-per-cell', type=int, default=10, help='largest number of coloured points per cell (1..20)')
    p.add_argument('--min-per-cell', type=int, default=1, choices=(0, 1), help='smallest number per occupied cell (1, or 0 to allow empty cells)')
    p.add_argument('--no-equal-classes', action='store_true', help='do not force equal numbers of limit cycles and chaos points')
    p.add_argument('--per-class', type=int, default=1000,
                   help='plot exactly this many limit cycles and this many chaos points (default 1000). The per-cell '
                        'cap is then the smallest one that allows it, so the points are spread as evenly as possible. '
                        '0 = old mode: as many points as --max-per-cell allows')
    p.add_argument('--seed', type=int, default=20260927)
    p.add_argument('--time-limit', type=float, default=120., help='seconds for the integer allocation')
    p.add_argument('--test-input', action='store_true', help='allow a small test CSV (output is watermarked)')
    p.add_argument('--no-tex', action='store_true', help='matplotlib mathtext instead of LaTeX')
    a = p.parse_args()
    if not 1 <= a.max_per_cell <= 20:
        p.error('--max-per-cell must be between 1 and 20')

    rows, counts = read_rows(a.input, a.test_input)
    if not a.auto_lim:
        lo, hi = a.lim
    else:                          # every cycle and chaos point inside, plus a margin; far fixed points are left out
        mv = np.array([xy(r) for r in rows if r['cls'] in MOVING])
        lo = 10 ** (np.floor(np.log10(mv.min())) - a.margin_decades)
        hi = 10 ** (np.ceil(np.log10(mv.max())) + a.margin_decades)
    # equal numbers of cycles and chaos need enough room for chaos points: if the requested cap is too
    # small for that, the smallest sufficient cap up to 20 is used and reported
    chosen = None
    caps = range(1, 101) if a.per_class else range(a.max_per_cell, 21)
    for cap in caps:
        try:
            chosen, report, cells_csv = select_coloured(rows, (lo, hi), a.cell_decades, cap, a.seed, a.time_limit,
                                                        equal_classes=not a.no_equal_classes, floor=a.min_per_cell,
                                                        per_class=a.per_class)
            break
        except ValueError as err:
            last = err
            if (a.no_equal_classes and not a.per_class) or not any(w in str(err).lower() for w in ('impossible', 'infeasible')):
                break
    if chosen is None:
        raise SystemExit('selection failed: %s' % last)
    report['cap_requested'] = None if a.per_class else a.max_per_cell
    if a.per_class:
        # spread the points as evenly as possible: raise the per-cell minimum (cells with fewer
        # available points show all of them) as far as the fixed totals allow (binary search)
        cap, lo_l, hi_l = report['cap'], 1, report['cap']
        while lo_l < hi_l:
            mid = (lo_l + hi_l + 1) // 2
            try:
                trial = select_coloured(rows, (lo, hi), a.cell_decades, cap, a.seed, a.time_limit, equal_classes=False,
                                        floor=a.min_per_cell, per_class=a.per_class, level=mid)
                chosen, report, cells_csv = trial; lo_l = mid
            except ValueError:
                hi_l = mid - 1
        print('%d limit cycles + %d chaos points: at most %d per cell (smallest cap that allows it), at least %d per '
              'cell where that many are available' % (a.per_class, a.per_class, report['cap'], max(1, report['even_level'])))
    elif report['cap'] != a.max_per_cell:
        print('note: equal numbers of cycles and chaos need up to %d points per cell (requested %d)'
              % (report['cap'], a.max_per_cell))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba, LogNorm, LinearSegmentedColormap
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    plt.rcParams.update({
        'text.usetex': not a.no_tex, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'] if not a.no_tex else ['cmr10', 'DejaVu Serif'],
        'axes.formatter.use_mathtext': a.no_tex,   # cmr10 has no minus glyph; mathtext supplies it
        'mathtext.fontset': 'cm', 'font.size': 9, 'axes.labelsize': 9, 'legend.fontsize': 7,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.6,
        'text.latex.preamble': r'\usepackage{amsmath,bm}'})
    fig, ax = plt.subplots(figsize=(3.4, 3.4))
    plt.subplots_adjust(left=0.16, right=0.98, bottom=0.14, top=0.98)

    # drawn first: light gray density of the stable fixed points of all four networks (darker = more);
    # empty bins stay white. Fixed points outside the axes are counted in the audit, not drawn.
    fp = np.array([xy(r) for r in rows if r['cls'] == 'fixed']).reshape(-1, 2)
    bins = np.logspace(np.log10(lo), np.log10(hi), a.fixed_bins + 1)
    h, xe, ye = np.histogram2d(fp[:, 0], fp[:, 1], bins=[bins, bins])
    if h.max() > 0:
        cmap = LinearSegmentedColormap.from_list('fixed', FIXED_SHADES)
        ax.pcolormesh(xe, ye, np.ma.masked_equal(h.T, 0), cmap=cmap, norm=LogNorm(vmin=1, vmax=max(h.max(), 2)),
                      rasterized=True, zorder=0, linewidth=0)
    inside = ((fp >= lo) & (fp <= hi)).all(axis=1) if len(fp) else np.zeros(0, bool)

    # selected cycles and chaos, one collection in the random order of the selection
    sq = MarkerStyle('s'); tr = MarkerStyle('^')
    path = {'cycle': sq.get_path().transformed(sq.get_transform()),
            'chaos_candidate': tr.get_path().transformed(tr.get_transform())}
    pts = np.array([xy(r) for r in chosen])
    sc = ax.scatter(pts[:, 0], pts[:, 1], s=[6 if r['cls'] == 'cycle' else 9 for r in chosen],
                    linewidths=[0.45 if r['cls'] == 'cycle' else 0.25 for r in chosen],
                    rasterized=True, zorder=2)
    sc.set_paths([path[r['cls']] for r in chosen])
    sc.set_facecolors([(0, 0, 0, 0) if r['cls'] == 'cycle' else to_rgba(CLASS_COLORS[r['cls']]) for r in chosen])
    sc.set_edgecolors([to_rgba(CLASS_COLORS['cycle']) if r['cls'] == 'cycle' else to_rgba('k') for r in chosen])

    ax.plot([lo, hi], [lo, hi], '--', color='k', lw=0.8, zorder=3)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
    ax.set_xlabel(r'$\overline{\Delta}_1^{\,2}/\overline{B}_1$', labelpad=1)
    ax.set_ylabel(r'$\overline{\sigma}_{\rm ps}$', labelpad=1)
    ax.tick_params(length=2.5)
    handles = [Line2D([], [], marker='s', ls='', mfc='none', mec=CLASS_COLORS['cycle'], mew=0.8, ms=4.5, label='limit cycle'),
               Line2D([], [], marker='^', ls='', mfc=CLASS_COLORS['chaos_candidate'], mec='k', mew=0.3, ms=5, label='chaos'),
               Patch(facecolor=FIXED_SHADES[1], edgecolor='none', label='stable fixed points')]
    leg = ax.legend(handles=handles, loc='upper left', frameon=True, fancybox=False, framealpha=0.9, handlelength=1.0,
                    borderpad=0.3, labelspacing=0.15, edgecolor='none')          # white box: readable over the gray density
    leg.set_zorder(5)

    # inset: saturation of every cycle and chaos candidate in the CSV
    mov = np.array([xy(r) for r in rows if r['cls'] in MOVING]).reshape(-1, 2)
    sat = mov[:, 0] / mov[:, 1]
    # place the inset where it hides no drawn cycle/chaos point and as little fixed-point density as
    # possible; the region above the diagonal holding the legend is excluded. The box includes the
    # inset title and tick labels.
    def frac(p):
        return ((np.log10(p[:, 0]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)),
                (np.log10(p[:, 1]) - np.log10(lo)) / (np.log10(hi) - np.log10(lo)))
    fxm, fym = frac(pts); fxf, fyf = frac(fp) if len(fp) else (np.zeros(0), np.zeros(0))
    W, Hh = 0.33, 0.2
    def box(fx, fy, x0, y0):
        return int(np.sum((fx > x0 - 0.08) & (fx < x0 + W + 0.02) & (fy > y0 - 0.07) & (fy < y0 + Hh + 0.07)))
    def covered(x0, y0):
        return box(fxm, fym, x0, y0) * 10 ** 9 + box(fxf, fyf, x0, y0)
    cands = [(x0, y0) for y0 in (0.09, 0.2, 0.3, 0.4) for x0 in (0.64, 0.55, 0.45)] + \
            [(0.12, y0) for y0 in (0.45, 0.55)]                      # left side, below the legend
    best = min(cands, key=lambda c: (covered(*c), cands.index(c)))
    under_inset = box(fxm, fym, *best); fixed_under_inset = box(fxf, fyf, *best)
    axi = ax.inset_axes([best[0], best[1], W, Hh])
    axi.set_facecolor('white'); axi.set_zorder(4)
    axi.hist(np.clip(sat, 0, 1), bins=np.linspace(0, 1, 26), color='0.45', lw=0)
    axi.axvline(1, color='k', lw=0.6, ls='--')
    axi.set_xlim(0, 1.02); axi.set_xticks([0, 0.5, 1]); axi.set_yticks([])
    axi.tick_params(labelsize=5.5, length=2, pad=1)
    axi.set_title(r'$\overline{\Delta}_1^{\,2}/(\overline{B}_1\overline{\sigma}_{\rm ps})$', fontsize=6.5, pad=2)
    for s_ in axi.spines.values():
        s_.set_linewidth(0.4)
    if a.test_input:
        fig.text(0.5, 0.005, 'TEST DATA', ha='center', va='bottom', fontsize=7, color='crimson')

    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / 'fig2_motion.pdf', dpi=300)
    fig.savefig(out / 'fig2_motion.png', dpi=200)

    # audit: what was plotted and why
    aud = Path(a.audit_dir); aud.mkdir(parents=True, exist_ok=True)
    allxy = np.array([xy(r) for r in rows])
    report.update(input=a.input, rows=len(rows), per_case={f'{m}/{"nonideal" if n else "ideal"}': v for (m, n), v in counts.items()},
                  fixed_points=len(fp), fixed_drawn=int(inside.sum()), points_under_inset=under_inset, fixed_under_inset=fixed_under_inset, inset_position=list(best), fixed_outside_window=int(np.sum(((fp < lo) | (fp > hi)).any(axis=1))),
                  moving_total=len(mov), saturation_max=float(sat.max()),
                  motion_cost_violations=int(np.sum(sat > 1 + 1e-9)),
                  all_outside_window=int(np.sum(((allxy < lo) | (allxy > hi)).any(axis=1))), test_input=a.test_input)
    (aud / 'fig_2_main_selection.json').write_text(json.dumps(report, indent=2))
    with open(aud / 'fig_2_main_selected.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['plot_order', 'point_id', 'model', 'nonideal', 'cls', 'x', 'y'])
        w.writeheader()
        for i, r in enumerate(chosen):
            x, y = xy(r)
            w.writerow(dict(plot_order=i, point_id=r['point_id'], model=r['model'], nonideal=r['nonideal'], cls=r['cls'], x=x, y=y))
    with open(aud / 'fig_2_main_cells.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['cell_x', 'cell_y', 'log10_x_from', 'log10_y_from', 'plotted']); w.writeheader(); w.writerows(cells_csv)
    print(json.dumps({k: report[k] for k in ('coloured_points', 'per_class', 'per_cell_min', 'per_cell_max', 'per_model',
                                              'fixed_points', 'motion_cost_violations', 'saturation_max')}, indent=1))


if __name__ == '__main__':
    main()
