#!/usr/bin/env python3
"""Cache a fine OCT level (default 24 um = 2x pooling of the slab-normalised 12 um stack) for the label-free vascular
channel: work/oct24.npy (float16), oct24_mask.npy (tissue), oct24_affine.npy.  For another dataset: point --oct at any
(Z,Y,X) OCT stack readable by tifffile / numpy and give --spacing-um and the same layout as prep_i46."""
import argparse, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from octreg.common import oct_slab_normalize, oct_tissue_mask, pool_mean_np
ap = argparse.ArgumentParser()
ap.add_argument("--dandi", type=Path, default=None); ap.add_argument("--oct", type=Path, default=None, help="OCT stack (.ome.tiff/.tif/.npy) if not DANDI sub-I46")
ap.add_argument("--work", type=Path, required=True); ap.add_argument("--pool", type=int, default=2)
ap.add_argument("--slab-window", type=int, default=100); ap.add_argument("--tissue-thresh", type=float, default=30.0)
a = ap.parse_args(); t0 = time.time()
src = a.oct or (a.dandi / "sub-I46/ses-OCT/micr/sub-I46_ses-OCT_sample-BrocaAreaS01_OCT.ome.tiff")
if str(src).endswith(".npy"):
    oct12 = np.load(src, mmap_mode="r")
else:
    import tifffile; oct12 = tifffile.imread(str(src))
print("OCT", oct12.shape, oct12.dtype, f"({time.time()-t0:.0f}s)", flush=True)
octn, info = oct_slab_normalize(np.asarray(oct12), tissue_thresh=a.tissue_thresh, window=a.slab_window); del oct12
p = pool_mean_np(octn, a.pool); del octn
m, thr = oct_tissue_mask(p, thresh=None, closing_iter=2)
A12 = np.load(a.work / "oct12_affine.npy"); A = A12.copy(); A[:3, 3] = A12[:3, 3] + A12[:3, :3] @ (np.ones(3) * (a.pool - 1) / 2.0); A[:3, :3] *= a.pool
name = f"oct{int(round(12 * a.pool))}"
np.save(a.work / f"{name}.npy", p.astype(np.float16)); np.save(a.work / f"{name}_mask.npy", m); np.save(a.work / f"{name}_affine.npy", A)
print(f"saved {name}: {p.shape}, tissue {m.mean():.2f}, thr {thr:.1f} ({time.time()-t0:.0f}s)")
