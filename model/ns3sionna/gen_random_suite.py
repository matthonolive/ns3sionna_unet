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
        rc = sample_free_cells(walls_2d, rng, n=(1 + args.n_sta))
        xy = rc_to_xy_m(rc, float(args.cell_size_m))
        tx_xy = xy[0]
        sta_xy = xy[1:]

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

    print(f"\n[done] suite written to: {out_root}")


if __name__ == "__main__":
    main()
