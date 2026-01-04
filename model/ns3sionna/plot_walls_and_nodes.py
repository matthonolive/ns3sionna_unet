#!/usr/bin/env python3
import argparse, json
from dataclasses import replace
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sionna.rt import load_scene

from mlink.scene import Scene as MlinkScene
from mlink.antenna import AntennaGrid, AntennaDatabase
from mlink.feature import build_feature_tensor

def load_placements(path):
    tx=None; stas=[]
    for line in Path(path).read_text().splitlines():
        line=line.strip()
        if not line or line.startswith("#"): continue
        tag,xs,ys,zs = [x.strip() for x in line.split(",")[:4]]
        x,y,z = float(xs),float(ys),float(zs)
        if tag in ("tx","ap"): tx=np.array([x,y,z],np.float32)
        if tag in ("sta","rx"): stas.append([x,y,z])
    if tx is None or not stas: raise RuntimeError("placements missing tx or sta")
    return tx, np.asarray(stas,np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--placementsCsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fcGHz", type=float, default=5.21)

    # match your UNet grid defaults (override if needed)
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--K", type=int, default=8)
    ap.add_argument("--scale", type=float, default=0.625)
    ap.add_argument("--zStep", type=float, default=0.625)
    ap.add_argument("--zMargin", type=float, default=0.3125)
    ap.add_argument("--originXY", choices=["bbox_min","zero"], default="bbox_min")
    args = ap.parse_args()

    tx, stas = load_placements(args.placementsCsv)
    scene = load_scene(args.xml)
    bbox = scene.mi_scene.bbox()

    if args.originXY == "zero":
        x0,y0 = 0.0,0.0
    else:
        x0,y0 = float(bbox.min.x), float(bbox.min.y)

    zmin,zmax = float(bbox.min.z), float(bbox.max.z)
    span = (args.K-1)*args.zStep
    z0 = zmin + args.zMargin
    if z0 + span > (zmax - args.zMargin):
        z0 = max(zmin, (zmax - args.zMargin) - span)

    rx_grid = AntennaGrid(
        origin=np.array([x0,y0,z0],np.float32),
        deltas=np.array([[args.scale,0,0],[0,args.scale,0],[0,0,args.zStep]],np.float32),
        shape=(args.K,args.H,args.W),
    )

    xs = x0 + args.scale*np.arange(args.W,dtype=np.float32)
    ys = y0 + args.scale*np.arange(args.H,dtype=np.float32)
    zs = z0 + args.zStep*np.arange(args.K,dtype=np.float32)
    Z,Y,X = np.meshgrid(zs,ys,xs,indexing="ij")
    rx_coords = np.stack([X,Y,Z],axis=-1).reshape(-1,3).astype(np.float32)

    base = MlinkScene.from_sionna(scene)
    adb = AntennaDatabase(tx.reshape(1,3), rx_coords, None, rx_grid)
    mscene = replace(base, antenna_database=adb)

    x = build_feature_tensor(mscene, args.fcGHz*1e9, requested=["binary_walls"]).astype(np.float32)
    walls = x[0,0]          # (K,H,W)
    walls_any = walls.max(0) # (H,W)

    img = walls_any.T  # imshow expects [y,x]
    extent = [x0, x0+args.scale*args.W, y0, y0+args.scale*args.H]

    fig, ax = plt.subplots(figsize=(8,6), dpi=160)
    ax.imshow(img, origin="lower", extent=extent, interpolation="nearest")
    ax.scatter([tx[0]],[tx[1]], marker="*", s=180, label="AP/TX")
    for i,p in enumerate(stas):
        ax.scatter([p[0]],[p[1]], s=40)
        ax.text(p[0],p[1], f"STA{i}", fontsize=8, ha="left", va="bottom")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(args.out)
    print("Wrote", args.out)

if __name__ == "__main__":
    main()
