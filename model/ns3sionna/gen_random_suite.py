#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from mlink.geometry import generate_wall_map, walls_to_mesh
from mesh.xml_io import ItuRadioMaterialSpec, export_walls_floor_ceiling_xml


def make_double_sided(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Duplicate faces with reversed winding so the surface is hittable from both sides.
    This is the "bidirectional" alternative to wrapping materials in <bsdf type="twosided">,
    which can break Sionna radio-material handling.
    """
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    if f.size == 0:
        return mesh
    f2 = np.vstack([f, f[:, ::-1]])
    return trimesh.Trimesh(vertices=v, faces=f2, process=False)


def sample_free_cells(walls_2d: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    """
    Sample n free cells from the occupancy grid (0 = free, 1 = wall).
    Returns indices array shape (n, 2) in (row, col).
    """
    free = np.argwhere(walls_2d == 0)
    if free.shape[0] == 0:
        raise RuntimeError("No free cells found (unexpected).")
    replace = free.shape[0] < n
    idx = rng.choice(free.shape[0], size=n, replace=replace)
    return free[idx]


def rc_to_xy_m(rc: np.ndarray, cell_size_m: float) -> np.ndarray:
    """
    Convert (row, col) grid indices into (x,y) meters at the cell center.
    """
    x = (rc[:, 0].astype(np.float32) + 0.5) * cell_size_m
    y = (rc[:, 1].astype(np.float32) + 0.5) * cell_size_m
    return np.stack([x, y], axis=1)


def clamp(z: float, lo: float, hi: float) -> float:
    return float(min(max(z, lo), hi))

def count_wall_cells_on_segment(
    walls_2d: np.ndarray,
    a_rc: np.ndarray,
    b_rc: np.ndarray,
    oversample: int = 6,
) -> int:
    """
    Approximate how many wall cells (value==1) a straight line from a_rc to b_rc passes through.
    Uses oversampled line sampling + unique visited cells.
    """
    H, W = walls_2d.shape
    a_r, a_c = int(a_rc[0]), int(a_rc[1])
    b_r, b_c = int(b_rc[0]), int(b_rc[1])

    dr = b_r - a_r
    dc = b_c - a_c
    n = int(max(abs(dr), abs(dc)) * oversample) + 1
    if n <= 2:
        return 0

    rr = np.linspace(a_r, b_r, n)
    cc = np.linspace(a_c, b_c, n)
    r = np.clip(np.rint(rr).astype(np.int32), 0, H - 1)
    c = np.clip(np.rint(cc).astype(np.int32), 0, W - 1)

    # ignore endpoints
    r = r[1:-1]
    c = c[1:-1]
    if r.size == 0:
        return 0

    # unique visited cells
    idx = np.unique(r * W + c)
    r_u = idx // W
    c_u = idx % W
    return int(walls_2d[r_u, c_u].sum())

def rc_dist_m(a_rc: np.ndarray, b_rc: np.ndarray, cell_size_m: float) -> float:
    dr = float(a_rc[0] - b_rc[0])
    dc = float(a_rc[1] - b_rc[1])
    return (dr * dr + dc * dc) ** 0.5 * cell_size_m

def select_friis_adversarial_placements(
    walls_2d: np.ndarray,
    rng: np.random.Generator,
    n_sta: int,
    cell_size_m: float,
    *,
    d_min_m: float,
    d_max_m: float,
    dist_tol_frac: float,
    min_wall_cells: int,
    max_ap_tries: int = 200,
    max_sta_tries: int = 20000,
    oversample: int = 6,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Returns:
      tx_rc: (2,)
      sta_rc: (n_sta, 2)
      meta: dict with wall_hits/dist_m/is_los per STA, and the chosen control distance.
    Strategy:
      - pick AP in free cell
      - pick 1 LoS control STA at distance in [d_min_m, d_max_m]
      - pick remaining STAs blocked with wall_hits>=min_wall_cells and distance within dist_tol_frac of control
    """
    free = np.argwhere(walls_2d == 0)
    if free.shape[0] < (1 + n_sta):
        raise RuntimeError("Not enough free cells for requested nodes")

    def wall_hits(a, b) -> int:
        return count_wall_cells_on_segment(walls_2d, a, b, oversample=oversample)

    for _ in range(max_ap_tries):
        tx_rc = free[rng.integers(0, free.shape[0])]

        # 1) pick LoS control STA
        control_rc = None
        control_d = None
        control_hits = None

        for _j in range(max_sta_tries // 10):
            cand = free[rng.integers(0, free.shape[0])]
            if (cand == tx_rc).all():
                continue
            d = rc_dist_m(tx_rc, cand, cell_size_m)
            if not (d_min_m <= d <= d_max_m):
                continue
            hits = wall_hits(tx_rc, cand)
            if hits == 0:
                control_rc = cand
                control_d = d
                control_hits = hits
                break

        if control_rc is None:
            continue

        # 2) pick blocked STAs with similar distance
        sta_rc_list = [control_rc]
        sta_meta = [{
            "is_los": True,
            "wall_cells": int(control_hits),
            "dist_m": float(control_d),
        }]

        # distance matching window
        d0 = float(control_d)
        d_lo = d0 * (1.0 - dist_tol_frac)
        d_hi = d0 * (1.0 + dist_tol_frac)

        tries = 0
        while len(sta_rc_list) < n_sta and tries < max_sta_tries:
            tries += 1
            cand = free[rng.integers(0, free.shape[0])]
            if (cand == tx_rc).all():
                continue
            if any((cand == x).all() for x in sta_rc_list):
                continue

            d = rc_dist_m(tx_rc, cand, cell_size_m)
            if not (d_lo <= d <= d_hi):
                continue

            hits = wall_hits(tx_rc, cand)
            if hits >= min_wall_cells:
                sta_rc_list.append(cand)
                sta_meta.append({
                    "is_los": False,
                    "wall_cells": int(hits),
                    "dist_m": float(d),
                })

        if len(sta_rc_list) < n_sta:
            # couldn't fill blocked STAs for this AP; try another AP
            continue

        sta_rc = np.stack(sta_rc_list, axis=0)
        meta = {
            "control_dist_m": float(d0),
            "d_window_m": [float(d_lo), float(d_hi)],
            "min_wall_cells": int(min_wall_cells),
            "sta": sta_meta,
        }
        return tx_rc, sta_rc, meta

    raise RuntimeError(
        "Failed to find Friis-adversarial placements. "
        "Try lowering min_wall_cells, increasing dist_tol_frac, or increasing max_partitions."
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True, help="Output root directory for suite")
    ap.add_argument("--n_scenes", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=int(np.random.default_rng().integers(0, 10000)))

    # Training-matching defaults
    ap.add_argument("--H", type=int, default=64)
    ap.add_argument("--W", type=int, default=64)
    ap.add_argument("--cell_size_m", type=float, default=0.625)

    ap.add_argument("--min_wall_length", type=int, default=8)
    ap.add_argument("--min_door_length", type=int, default=4)
    ap.add_argument("--max_partitions", type=int, default=24)

    # Heights are sampled in *grid units* then scaled by cell_size_m
    ap.add_argument("--floor_h_units", type=float, default=0.0)
    ap.add_argument("--ceil_min_units", type=float, default=8.0)
    ap.add_argument("--ceil_max_units", type=float, default=20.0)

    # Placements
    ap.add_argument("--n_sta", type=int, default=8)
    ap.add_argument("--tx_h_m", type=float, default=1.5)
    ap.add_argument("--rx_h_m", type=float, default=1.5)
    ap.add_argument("--z_margin_m", type=float, default=0.25)

    # Bookkeeping (doesn't affect geometry)
    ap.add_argument("--frequency_hz", type=float, default=2e9)

    # Materials (explicit, named, consistent)
    ap.add_argument("--wall_itu", type=str, default="brick")
    ap.add_argument("--floor_itu", type=str, default="concrete")
    ap.add_argument("--ceiling_itu", type=str, default="concrete")
    ap.add_argument("--wall_thickness_m", type=float, default=0.10)
    ap.add_argument("--floor_thickness_m", type=float, default=0.15)
    ap.add_argument("--ceiling_thickness_m", type=float, default=0.10)

    #Adversarial Friis placements
    ap.add_argument("--friis_adversarial", action="store_true",
                    help="Choose placements to make Friis fail (LoS control + blocked STAs at same distance)")
    ap.add_argument("--d_min_m", type=float, default=8.0)
    ap.add_argument("--d_max_m", type=float, default=30.0)
    ap.add_argument("--dist_tol_frac", type=float, default=0.10,
                    help="Blocked STA distance must be within +/- this fraction of control distance")
    ap.add_argument("--min_wall_cells", type=int, default=2,
                    help="Require >= this many wall grid cells intersected by the TX->STA line for blocked STAs")



    ap.add_argument("--double_sided", action="store_false", help="Duplicate faces to make surfaces bidirectional")
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    suite_manifest = {
        "n_scenes": args.n_scenes,
        "seed0": args.seed0,
        "grid_hw": [args.H, args.W],
        "cell_size_m": args.cell_size_m,
        "partition_params": {
            "min_wall_length": args.min_wall_length,
            "min_door_length": args.min_door_length,
            "max_partitions": args.max_partitions,
        },
        "height_sampling_units": {
            "floor_h_units": args.floor_h_units,
            "ceil_min_units": args.ceil_min_units,
            "ceil_max_units": args.ceil_max_units,
        },
        "frequency_hz": args.frequency_hz,
        "materials": {
            "walls": {"itu": args.wall_itu, "thickness_m": args.wall_thickness_m},
            "floor": {"itu": args.floor_itu, "thickness_m": args.floor_thickness_m},
            "ceiling": {"itu": args.ceiling_itu, "thickness_m": args.ceiling_thickness_m},
        },
        "double_sided": bool(args.double_sided),
    }
    (out_root / "suite_manifest.json").write_text(json.dumps(suite_manifest, indent=2))

    for k in range(args.n_scenes):
        seed = args.seed0 + k
        rng = np.random.default_rng(seed)

        walls_2d = generate_wall_map(
            (args.H, args.W),
            min_wall_length=args.min_wall_length,
            min_door_length=args.min_door_length,
            max_partitions=args.max_partitions,
            rng=rng,
        )

        # Sample ceiling height in GRID UNITS, then create mesh in those units
        ceil_h_units = float(rng.uniform(args.ceil_min_units, args.ceil_max_units))
        floor_h_units = float(args.floor_h_units)

        mesh_units = walls_to_mesh(
            walls_2d,
            floor_height=floor_h_units,
            ceiling_height=ceil_h_units,
        )

        # Scale to meters (matches your training convention)
        mesh_m = mesh_units.copy()
        mesh_m.apply_scale(float(args.cell_size_m))

        print("double sided?: ", args.double_sided)
        if args.double_sided:
            mesh_m = make_double_sided(mesh_m)

        # Scene folder
        scene_dir = out_root / f"seed{seed:04d}"
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Save the occupancy grid
        np.save(scene_dir / "walls_2d.npy", walls_2d)

        # Materials: explicit IDs + explicit ITU type
        wall_mat = ItuRadioMaterialSpec(
            bsdf_id=f"mat-itu_{args.wall_itu}",
            itu_type=args.wall_itu,
            thickness=float(args.wall_thickness_m),
        )
        floor_mat = ItuRadioMaterialSpec(
            bsdf_id=f"mat-itu_{args.floor_itu}",
            itu_type=args.floor_itu,
            thickness=float(args.floor_thickness_m),
        )
        ceil_mat = ItuRadioMaterialSpec(
            bsdf_id=f"mat-itu_{args.ceiling_itu}",
            itu_type=args.ceiling_itu,
            thickness=float(args.ceiling_thickness_m),
        )

        # Export XML+OBJs (splits by min-z/max-z planes internally)
        xml_path = export_walls_floor_ceiling_xml(
            mesh_m,
            out_dir=scene_dir,
            xml_name="scene.xml",
            wall_material=wall_mat,
            floor_material=floor_mat,
            ceiling_material=ceil_mat,
        )

        # Placements: sample one AP + N STAs in free cells
        if args.friis_adversarial:
            tx_rc, sta_rc, place_meta = select_friis_adversarial_placements(
                walls_2d,
                rng,
                n_sta=args.n_sta,
                cell_size_m=float(args.cell_size_m),
                d_min_m=float(args.d_min_m),
                d_max_m=float(args.d_max_m),
                dist_tol_frac=float(args.dist_tol_frac),
                min_wall_cells=int(args.min_wall_cells),
            )
            tx_xy = rc_to_xy_m(tx_rc[None, :], float(args.cell_size_m))[0]
            sta_xy = rc_to_xy_m(sta_rc, float(args.cell_size_m))
        else:
            rc = sample_free_cells(walls_2d, rng, n=(1 + args.n_sta))
            xy = rc_to_xy_m(rc, float(args.cell_size_m))
            tx_xy = xy[0]
            sta_xy = xy[1:]
            place_meta = None

        # Clamp heights inside floor/ceiling (meters)
        floor_h_m = floor_h_units * float(args.cell_size_m)
        ceil_h_m = ceil_h_units * float(args.cell_size_m)
        z_lo = floor_h_m + float(args.z_margin_m)
        z_hi = ceil_h_m - float(args.z_margin_m)

        tx_z = clamp(float(args.tx_h_m), z_lo, z_hi)
        rx_z = clamp(float(args.rx_h_m), z_lo, z_hi)

        placements = {
            "seed": seed,
            "frequency_hz": float(args.frequency_hz),
            "cell_size_m": float(args.cell_size_m),
            "floor_h_m": float(floor_h_m),
            "ceil_h_m": float(ceil_h_m),
            "tx_xyz": [float(tx_xy[0]), float(tx_xy[1]), float(tx_z)],
            "sta_xyz": [[float(x), float(y), float(rx_z)] for x, y in sta_xy],
        }

        if place_meta is not None:
            placements["friis_adversarial_meta"] = place_meta

        (scene_dir / "placements.json").write_text(json.dumps(placements, indent=2))

        scene_info = {
            "seed": seed,
            "xml": str(xml_path.name),
            "materials": {
                "walls": wall_mat.__dict__,
                "floor": floor_mat.__dict__,
                "ceiling": ceil_mat.__dict__,
            },
            "height_units": {"floor": floor_h_units, "ceiling": ceil_h_units},
            "height_m": {"floor": floor_h_m, "ceiling": ceil_h_m},
        }
        (scene_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

        print(f"[ok] {scene_dir.name}: wrote scene.xml + meshes + placements.json")

        csv_lines = []
        tx = placements["tx_xyz"]
        csv_lines.append(f"tx,{tx[0]},{tx[1]},{tx[2]}")
        for s in placements["sta_xyz"]:
            csv_lines.append(f"sta,{s[0]},{s[1]},{s[2]}")
        (scene_dir / "placements.csv").write_text("\n".join(csv_lines) + "\n")

    print(f"\n[done] suite written to: {out_root}")


if __name__ == "__main__":
    main()
