#!/usr/bin/env python3
"""Side-by-side figure of two poses from two `octreg qc` outputs of the same run (same planes, different T).

    python bench/compose_pair.py PREFIX_A PREFIX_B OUT.png [--titles "pose A" "pose B"] [--rows qc|montage:i,j,k]

Columns: OCT | MRI and checkerboard of pose A | MRI and checkerboard of pose B, one row per plane: the three centroid planes of
qc.png, or montage rows i, j, k of qc_montage.png. Plane images and row labels are cut pixel for pixel from the qc figures (layout
constants from octreg.register); the OCT planes of the two inputs must be identical. bench/figures/fig_handedness_xiangrui.png:

    python -m octreg qc --run RUN --oct OCT --mri MRI --T ABL/variants/A8/T_oct2mri.txt -o OTHER
    python bench/compose_pair.py RUN/qc OTHER OUT.png --titles "octreg result (handedness of the file headers)" \
        "best pose of the other handedness"
"""
import argparse

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from PIL import Image

from octreg.register import QC_CELL_IN, QC_DPI, QC_HEAD_IN

GAP, HEAD, SEP, LABEL_W = 8, 64, 14, 34          # pixels


def longest_run(flags):
    """(start, stop) of the longest run of True."""
    best, start = (0, 0), None
    for i, f in enumerate(list(flags) + [False]):
        if f and start is None:
            start = i
        elif not f and start is not None:
            best, start = max(best, (i - start, start)), None
    return best[1], best[1] + best[0]


def extent(img, r, j, n, bg):
    """Pixel box of the plane image in qc row r, column j: the largest block of non-background pixels in its axes box."""
    t, l = round((QC_HEAD_IN + r * QC_CELL_IN) * QC_DPI), round((0.34 + j * QC_CELL_IN) * QC_DPI) - 2
    box = img[t:t + n + 8, l:l + n + 4] != bg
    y0, y1 = longest_run(box.any(axis=1))
    x0, x1 = longest_run(box[y0:y1].any(axis=0))
    return t + y0, t + y1, l + x0, l + x1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("out")
    ap.add_argument("--titles", nargs=2, default=("pose A", "pose B"))
    ap.add_argument("--rows", default="qc", help="'qc' (centroid planes) or 'montage:i,j,k' (montage rows)")
    a = ap.parse_args()
    suffix, rows = ("", [0, 1, 2]) if a.rows == "qc" else ("_montage", [int(x) for x in a.rows.split(":")[1].split(",")])
    A, B = (np.asarray(Image.open(f"{p}{suffix}.png").convert("L")) for p in (a.a, a.b))
    if A.shape != B.shape:
        ap.error(f"the two qc figures differ in size: {A.shape} {B.shape}")
    bg, n = int(A[2, 2]), round((QC_CELL_IN - 0.08) * QC_DPI)
    cols = [(A, 0), (A, 1), (A, 2), (B, 1), (B, 2)]
    xs = [LABEL_W + j * (n + GAP) + SEP * (j >= 3) for j in range(5)]
    W, H = xs[-1] + n + 6, HEAD + len(rows) * (n + GAP)
    canvas = np.full((H, W), 255, np.uint8)
    for i, r in enumerate(rows):
        ys = HEAD + i * (n + GAP)
        boxes = [extent(img, r, c, n, bg) for img, c in cols]
        y0, y1, x0, x1 = boxes[0]
        if not all(b[1] - b[0] == y1 - y0 and b[3] - b[2] == x1 - x0 for b in boxes) or \
                not np.array_equal(A[y0:y1, x0:x1], B[y0:y1, x0:x1]):
            ap.error("the OCT planes differ between the two qc figures (not the same run?)")
        oy, ox = ys + (n - (y1 - y0)) // 2, (n - (x1 - x0)) // 2
        top = round((QC_HEAD_IN + r * QC_CELL_IN + 0.04) * QC_DPI)
        label = A[top:top + n, :LABEL_W].copy()
        label[label == bg] = 255
        canvas[ys:ys + label.shape[0], :LABEL_W] = label
        for j, ((img, _), (b0, b1, c0, c1)) in enumerate(zip(cols, boxes)):
            canvas[oy:oy + b1 - b0, xs[j] + ox:xs[j] + ox + c1 - c0] = img[b0:b1, c0:c1]
    fig = Figure(figsize=(W / QC_DPI, H / QC_DPI), dpi=QC_DPI, facecolor="white")
    fig.figimage(canvas, 0, 0, cmap="gray", vmin=0, vmax=255, origin="upper")
    for j, t in enumerate(("OCT", "MRI through T, inverted", "checkerboard", "MRI through T, inverted", "checkerboard")):
        fig.text((xs[j] + n / 2) / W, 1 - (HEAD - 8) / H, t, ha="center", va="bottom", fontsize=9)
    for (j0, j1), t in zip(((1, 2), (3, 4)), a.titles):
        xa, xb = xs[j0] / W, (xs[j1] + n) / W
        fig.text((xa + xb) / 2, 1 - 18 / H, t, ha="center", va="center", fontsize=10, weight="bold")
        fig.add_artist(Line2D([xa + 0.01, xb - 0.01], [1 - 30 / H] * 2, color="0.3", lw=0.8))
    xm = (xs[3] - SEP / 2 - GAP / 2) / W
    fig.add_artist(Line2D([xm, xm], [0.005, 1 - 6 / H], color="0.5", lw=0.8))
    agg = FigureCanvasAgg(fig)
    agg.draw()
    Image.fromarray(np.asarray(agg.buffer_rgba())[..., :3]).convert("L").save(a.out, optimize=True)
    print(a.out, W, H)


if __name__ == "__main__":
    main()
