import sys, numpy as np, torch
sys.path.insert(0,"octreg")
from octreg.parser import load_parser, parse_volume
pd=sys.argv[1]
net=load_parser(f"work/{pd}/parser.pt")
oct=np.load("work/i46/oct150.npy"); mask=np.load("work/i46/oct150_mask.npy")
prob=parse_volume(net, oct, patch=64, stride=32).float().numpy()
np.save(f"work/{pd}/oct150_prob.npy", prob.astype(np.float16))
pred=prob.argmax(0); print(pd, "OCT parsed class fractions inside tissue: bg %.3f WM %.3f infra %.3f supra %.3f"%tuple((pred[mask]==c).mean() for c in range(4)), "| predicted-tissue vs mask agreement %.3f"%(((pred>0)==mask).mean()))
