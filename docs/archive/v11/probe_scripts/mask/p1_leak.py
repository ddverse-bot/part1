import sys; sys.path.insert(0, "/tmp")
from p1_common import *
from skimage.filters import apply_hysteresis_threshold
from skimage.graph import route_through_array
o = np.load(f"{W}/oct150.npy"); zero = o <= 0; x = o.astype(np.float32)
t, mu = local_std(x, 3); ts = gaussian_filter(t, 0.7); hi = otsu(ts[~zero])
vmax = np.percentile(o[o > 0], 99.5)
for lo_f, dil in ((0.5, 1), (0.35, 1)):
    rim = apply_hysteresis_threshold(ts, lo_f * hi, hi)
    if dil: rim = ndimage.binary_dilation(rim, structure=ball(1))
    passable = ~rim & ~zero
    cost = np.where(passable, 1.0, 1e4).astype(np.float32)
    # exterior target: passable voxels on the y and z faces (agarose faces); route from the box centre
    c = tuple(s // 2 for s in o.shape); say("centre passable?", passable[c])
    tgt = np.zeros(o.shape, bool); tgt[0], tgt[-1], tgt[:, 0], tgt[:, -1] = True, True, True, True; tgt &= passable
    # geodesic BFS distance from the exterior faces through passable; the leak = the min-cost path centre->faces
    lab, n = label(passable); say("passable comps", n, "centre comp size cm3", (lab == lab[c]).sum() * VOX150**3 / 1000, "touches y/z faces:", bool((tgt & (lab == lab[c])).any()))
    # find nearest target voxel by Dijkstra: route to each face is expensive -> approximate with the face voxel of the centre comp that is closest in euclid, then route
    idx = np.argwhere(tgt & (lab == lab[c]))
    if len(idx) == 0: say("no leak to y/z faces at lo", lo_f); continue
    d2 = ((idx - np.array(c)) ** 2).sum(1); e = tuple(idx[np.argmin(d2)]); say("closest face voxel (euclid)", e)
    path, tc = route_through_array(cost, c, e, fully_connected=False, geometric=False); path = np.array(path); say("path len", len(path), "cost", tc)
    # along the path, sample the texture map and the intensity; print where ts is highest (the crossing)
    tp = ts[path[:, 0], path[:, 1], path[:, 2]]; ip = o[path[:, 0], path[:, 1], path[:, 2]]
    say("ts along path (every 10):", np.round(tp[::10]).astype(int).tolist()); say("path start/end", path[0].tolist(), path[-1].tolist())
    np.save(f"{OUT}/leakpath_lo{lo_f}.npy", path)
    # figure: slices through the path at 5 positions, showing the rim and the path
    fig, ax = plt.subplots(2, 5, figsize=(25, 10))
    for k, frac in enumerate((0.1, 0.3, 0.5, 0.7, 0.9)):
        p = path[int(len(path) * frac)]
        for r, axn in enumerate((0, 1)):
            i = p[axn]; im = np.take(o, i, axis=axn); rm = np.take(rim, i, axis=axn)
            a = ax[r, k]; a.imshow(im, cmap="gray", vmax=vmax); a.contour(rm.astype(float), levels=[0.5], colors="r", linewidths=0.5)
            sel = np.abs(path[:, axn] - i) <= 2; oth = [j for j in range(3) if j != axn]; a.plot(path[sel, oth[1]], path[sel, oth[0]], "c.", ms=3); a.plot(p[oth[1]], p[oth[0]], "yo", ms=8, mfc="none")
            a.set_title(f"path {frac:.0%} axis{axn} idx{i}", fontsize=9); a.axis("off")
    plt.tight_layout(); plt.savefig(f"{OUT}/leak_lo{lo_f}.png", dpi=55); plt.close(fig)
say("done leak")
