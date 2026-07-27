#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mlink.geometry import walls_to_mesh
from xml_io import export_walls_floor_ceiling_xml


def build_blocked_room(
    H: int = 32,
    W: int = 32,
    wall_thickness: int = 1,
    add_outer_walls: bool = True,
    gap: bool = False,
) -> np.ndarray:
    """
    Build a simple 2-D wall raster.

    The default geometry is a rectangular room with one interior vertical wall
    between TX and RX, so the direct path is blocked.

    If gap=True, a doorway is cut in the blocker wall.
    """
    walls = np.zeros((H, W), dtype=np.uint8)

    if add_outer_walls:
        walls[0, :] = 1
        walls[-1, :] = 1
        walls[:, 0] = 1
        walls[:, -1] = 1

    # Interior blocker wall, slightly right of center.
    c0 = W // 2
    r0 = 4
    r1 = H - 4
    walls[r0:r1, c0:c0 + wall_thickness] = 1

    if gap:
        g0 = H // 2 - 2
        g1 = H // 2 + 2
        walls[g0:g1, c0:c0 + wall_thickness] = 0

    return walls


def write_placements_csv(out_dir: Path, tx_xyz: tuple[float, float, float], rx_xyz: tuple[float, float, float]) -> None:
    text = (
        f"tx,{tx_xyz[0]:.3f},{tx_xyz[1]:.3f},{tx_xyz[2]:.3f}\n"
        f"sta,{rx_xyz[0]:.3f},{rx_xyz[1]:.3f},{rx_xyz[2]:.3f}\n"
    )
    (out_dir / "placements.csv").write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a hard-coded two-node ns3sionna scene suite")
    ap.add_argument("--out", type=Path, default=Path("seed_blocked_two_node"), help="Output directory")
    ap.add_argument("--H", type=int, default=32, help="Grid height (cells)")
    ap.add_argument("--W", type=int, default=32, help="Grid width (cells)")
    ap.add_argument("--scale", type=float, default=0.625, help="Cell size in meters")
    ap.add_argument("--floor-h", type=float, default=0.0, help="Floor height in meters")
    ap.add_argument("--ceil-h", type=float, default=3.0, help="Ceiling height in meters")
    ap.add_argument("--tx-z", type=float, default=2.0, help="TX height in meters")
    ap.add_argument("--rx-z", type=float, default=1.0, help="RX height in meters")
    ap.add_argument("--tx-x", type=float, default=4.0, help="TX x position in meters")
    ap.add_argument("--tx-y", type=float, default=10.0, help="TX y position in meters")
    ap.add_argument("--rx-x", type=float, default=14.0, help="RX x position in meters")
    ap.add_argument("--rx-y", type=float, default=10.0, help="RX y position in meters")
    ap.add_argument("--gap", action="store_true", help="Cut a doorway in the blocker wall")
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    walls = build_blocked_room(H=args.H, W=args.W, gap=args.gap)
    mesh = walls_to_mesh(
        walls,
        floor_height=args.floor_h,
        ceiling_height=args.ceil_h,
    ).apply_scale(args.scale)

    export_walls_floor_ceiling_xml(
        mesh=mesh,
        out_dir=out_dir,
        xml_name="scene.xml",
        wall_bsdf="mat-itu_brick",
        floor_bsdf="mat-itu_concrete",
        ceil_bsdf="mat-itu_concrete",
    )

    tx_xyz = (args.tx_x, args.tx_y, args.tx_z)
    rx_xyz = (args.rx_x, args.rx_y, args.rx_z)
    write_placements_csv(out_dir, tx_xyz, rx_xyz)

    extent_x = (args.W - 1) * args.scale
    extent_y = (args.H - 1) * args.scale
    print(f"Wrote suite to: {out_dir}")
    print(f"Scene extent ≈ {extent_x:.2f} m x {extent_y:.2f} m")
    print(f"TX: {tx_xyz}")
    print(f"RX: {rx_xyz}")
    print(f"Mode: {'doorway / partial visibility' if args.gap else 'blocked LoS'}")


if __name__ == "__main__":
    main()
