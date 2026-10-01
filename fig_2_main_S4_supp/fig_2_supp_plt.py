"""Figure 2 supplement: one figure, a 2 x 4 grid. Columns are the networks, labelled (a) WR,
(b) Brusselator, (c) Oregonator, (d) CS14; rows are ideal (top) and nonideal (bottom) mixing.

Built only from the CSV of fig_2_input_selected.py. Every input of a
panel is drawn: no selection, thinning or density control. Stable fixed points in gray,
limit cycles as open squares and chaos candidates as filled triangles in the colour of the
network. Drawing order: fixed points, then all limit cycles, then all chaos points on top
(each class in random order). Both panels
of a column share axis limits that contain every point. Output: Figures/fig2_supp.png (300 dpi) and Figures/fig2_supp.pdf, in which the points
of each panel are one embedded image and the axes and text are vector (--pdf-raster: the
whole figure as one 600-dpi image).

    python3 fig_2_supp_plt.py --input data_fig2/fig_2_inputs.csv --output Figures
"""
import argparse, csv, json, math
from pathlib import Path
import numpy as np

MODELS = ('wr', 'brusselator', 'oregonator', 'cs14')
LABELS = {'wr': 'WR', 'brusselator': 'Brusselator', 'oregonator': 'Oregonator', 'cs14': 'CS14'}
COLORS = {'wr': '#0072B2', 'brusselator': '#009E73', 'oregonator': '#CC79A7', 'cs14': '#D55E00'}
MOVING = ('cycle', 'chaos_candidate')
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
    rows = load_columns(path)
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
    return rows


def xy(r):
    return float(r['Delta1']) ** 2 / float(r['B1']), float(r['sigma_ps'])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input', default='data_fig2/fig_2_inputs.csv')
    p.add_argument('--output', default='Figures')
    p.add_argument('--seed', type=int, default=20260927, help='seed of the random drawing order')
    p.add_argument('--test-input', action='store_true', help='allow a small test CSV (output is watermarked)')
    p.add_argument('--no-tex', action='store_true', help='matplotlib mathtext instead of LaTeX')
    p.add_argument('--pdf-raster', action='store_true',
                   help='write the PDF as one 600-dpi image of the whole figure (always opens, text not selectable)')
    a = p.parse_args()
    rows = read_rows(a.input, a.test_input); TARGET = target_of(rows)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.ticker
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    plt.rcParams.update({
        'text.usetex': not a.no_tex, 'font.family': 'serif',
        'font.serif': ['Computer Modern Roman'] if not a.no_tex else ['cmr10', 'DejaVu Serif'],
        'axes.formatter.use_mathtext': a.no_tex,   # cmr10 has no minus glyph; mathtext supplies it
        'mathtext.fontset': 'cm', 'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7.5,
        'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.linewidth': 0.6,
        'text.latex.preamble': r'\usepackage{amsmath,bm}'})
    out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    # one figure: rows = networks, columns = ideal | nonideal, panels labelled (a)..(h) row by row
    # columns = networks (a) WR .. (d) CS14, rows = ideal (top) and nonideal (bottom)
    fig, axes = plt.subplots(2, len(MODELS), figsize=(7.0, 3.9))
    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.1, top=0.86, wspace=0.42, hspace=0.22)
    summary = []
    for mi, m in enumerate(MODELS):
        groups = [[r for r in rows if r['model'] == m and nonideal(r) == non] for non in (False, True)]
        for non, g in zip((False, True), groups):
            if not g:
                raise SystemExit('no inputs for %s %s' % (m, 'nonideal' if non else 'ideal'))
            if not a.test_input and len(g) != TARGET:
                raise SystemExit('%s %s has %d inputs, not %d' % (m, 'nonideal' if non else 'ideal', len(g), TARGET))
        # shared limits of a column, containing every point of both panels (nothing clipped)
        allxy = np.array([xy(r) for r in groups[0] + groups[1]])
        lo = 10 ** np.floor(np.log10(allxy.min())); hi = 10 ** np.ceil(np.log10(allxy.max()))
        for ci, (non, g) in enumerate(zip((False, True), groups)):
            ax = axes[ci, mi]
            fp = np.array([xy(r) for r in g if r['cls'] == 'fixed']).reshape(-1, 2)
            # order: fixed points, then every limit cycle (random order), then every chaos point on top
            cy_ = [r for r in g if r['cls'] == 'cycle']; ch_ = [r for r in g if r['cls'] == 'chaos_candidate']
            mv = [cy_[i] for i in rng.permutation(len(cy_))] + [ch_[i] for i in rng.permutation(len(ch_))]
            if len(fp):
                ax.scatter(fp[:, 0], fp[:, 1], s=1.5, c='0.75', edgecolors='none', zorder=0.5)   # light gray, drawn first
            if mv:
                pts = np.array([xy(r) for r in mv]); cyc = np.array([r['cls'] == 'cycle' for r in mv])
                sq = MarkerStyle('s'); tr = MarkerStyle('^')
                paths = [(sq if c else tr).get_path().transformed((sq if c else tr).get_transform()) for c in cyc]
                sc = ax.scatter(pts[:, 0], pts[:, 1], s=np.where(cyc, 5, 8), linewidths=np.where(cyc, 0.4, 0.25),
                                zorder=1.1)
                sc.set_paths(paths)
                col = matplotlib.colors.to_rgba(COLORS[m])
                sc.set_facecolors([(0, 0, 0, 0) if c else col for c in cyc])
                sc.set_edgecolors([col if c else (0, 0, 0, 1) for c in cyc])
            ax.set_rasterization_zorder(1.2)   # points (zorder < 1.2) -> one image in the PDF; axes (1.5), text, diagonal stay vector
            ax.plot([lo, hi], [lo, hi], '--', color='k', lw=0.7, zorder=3)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
            if ci == 1:
                ax.set_xlabel(r'$\overline{\Delta}_1^{\,2}/\overline{B}_1$', labelpad=1)
            ax.set_ylabel(r'$\overline{\sigma}_{\rm ps}$', labelpad=0)
            # at most five decade labels per axis so wide ranges (WR) stay readable
            d0, d1 = int(round(np.log10(lo))), int(round(np.log10(hi))); step = max(1, math.ceil((d1 - d0) / 4))
            ticks = [10.0 ** e for e in range(d0, d1 + 1, step)]
            ax.set_xticks(ticks); ax.set_yticks(ticks)
            subs = 'auto' if step == 1 else (1.0,)               # wide range: minor ticks at every decade only
            ax.xaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=subs, numticks=100))
            ax.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(base=10, subs=subs, numticks=100))
            ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter()); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            if mi == 0:                                           # row label
                ax.text(-0.42, 0.5, ('nonideal' if non else 'ideal'), transform=ax.transAxes, rotation=90,
                        ha='center', va='center', fontsize=9)
            ax.tick_params(length=2.2, pad=1.5)
            letter = 'abcd'[mi]                                   # one letter per network (column)
            if ci == 0:                                           # column label (a) WR, (b) Brusselator, ...
                ax.set_title(((r'\textbf{(%s)} ' if not a.no_tex else '(%s) ') % letter) + LABELS[m],
                             loc='left', fontsize=9, pad=4)
            summary.append(dict(panel=letter, model=m, nonideal=non, plotted=len(g), fixed=len(fp),
                                cycle=sum(r['cls'] == 'cycle' for r in mv),
                                chaos=sum(r['cls'] == 'chaos_candidate' for r in mv), limits=[float(lo), float(hi)]))
    # one legend for the whole figure: marker shape = class, colour = network (row)
    handles = [Line2D([], [], marker='s', ls='', mfc='none', mec='k', mew=0.7, ms=4.5, label='limit cycle'),
               Line2D([], [], marker='^', ls='', mfc='0.35', mec='k', mew=0.3, ms=5, label='chaos'),
               Line2D([], [], marker='o', ls='', mfc='0.75', mec='none', ms=3, label='stable fixed point'),
               Line2D([], [], ls='--', color='k', lw=0.7, label=r'$\overline{\sigma}_{\rm ps}=\overline{\Delta}_1^{\,2}/\overline{B}_1$')]
    fig.legend(handles=handles, loc='upper center', ncol=4, frameon=False, handlelength=1.4, columnspacing=1.5,
               bbox_to_anchor=(0.53, 1.0))
    if a.test_input:
        fig.text(0.5, 0.003, 'TEST DATA', ha='center', va='bottom', fontsize=7, color='crimson')
    fig.savefig(out / 'fig2_supp.png', dpi=300)
    # PDF: the points of each panel are one embedded image, so the file stays small and every viewer opens it.
    # Written to a temporary name first, so an interrupted run never leaves a broken fig2_supp.pdf behind.
    pdf, tmp = out / 'fig2_supp.pdf', out / 'fig2_supp.pdf.part'
    if a.pdf_raster:
        from PIL import Image
        fig.savefig(out / 'fig2_supp_600dpi.png', dpi=600)
        Image.open(out / 'fig2_supp_600dpi.png').convert('RGB').save(tmp, 'PDF', resolution=600)
        (out / 'fig2_supp_600dpi.png').unlink()
    else:
        fig.savefig(tmp, dpi=300, format='pdf')
    with open(tmp, 'rb') as f:
        f.seek(max(0, tmp.stat().st_size - 1024)); ok = b'%%EOF' in f.read()
    if not ok:
        raise SystemExit('the PDF was not written completely: %s' % tmp)
    tmp.replace(pdf)
    plt.close(fig)
    (out / 'fig2_supp_summary.json').write_text(json.dumps(summary, indent=2))
    for s_ in summary:
        print(s_)

if __name__ == '__main__':
    main()
