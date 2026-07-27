#!/usr/bin/env python3
"""
gen_validation_suite.py
==================================================================
Deterministic validation scenes for the RESIDUAL-OVER-COST U-Net
(train_delta_tau.py checkpoints served by ns3unet_spectrum.py).

Unlike gen_random_suite.py (random wall maps), every scene here is a
DESIGNED layout that isolates one property of the residual model, so a
pass/fail is interpretable:

  v01_open_los        LOS floor property. No interior walls; every link
                      is LOS, so the gate forces r=0 and the UNet must
                      reproduce the cost model (=FSPL) EXACTLY. Any
                      throughput gap vs RT here is NOT the net's fault.

  v02_single_wall     Residual upside. One partition with a 4-cell door;
                      cost charges the through-wall loss while RT finds
                      the door path, so wb_cost overestimates loss on the
                      far side. The residual head should recover the gap.
                      STAs are placed at increasing detour depth.

  v03_deep_nlos       Stratified depth. Three staggered partitions with
                      offset doors (zigzag detour); STA pairs at 0/1/2/3
                      wall depth. Per-stratum comparison vs cost is the
                      network-level analog of the training diagnostic.

  v04_sealed_pocket   Coverage head. A fully sealed closet holds one STA;
                      RT sees (near) no-path, cost still reports a finite
                      loss. Validates sentinel emission and gives a knob
                      to sweep --unet_cov_thresh.

  v05_door_crossing   Link-activity detection (mobility). Same geometry
                      as v02 plus a DESIGNED waypoint trace: one STA
                      walks LOS -> through the door -> deep NLOS -> back,
                      with crossing times recorded in expected.json.
                      Windowed activity/throughput transitions vs RT
                      measure whether the surrogate flips links at the
                      right time, not just the right long-run average.

  v06_bigscene_tiling Tiling / off-patch OOD. 112x112-cell scene (70 m);
                      TX centered, STA ring out to 32.5 m so distant
                      patches do NOT contain the TX (the off-patch
                      regime). Plus a walker trace crossing patch seams
                      (stride = 32 cells = 20 m) to expose cache-handoff
                      discontinuities.

Output format is gen_random_suite-compatible per scene dir:
  walls_2d.npy, scene.xml + meshes/, placements.json, placements.csv,
  scene_info.json, expected.json, and (mobility themes) mobility_trace.csv
  with header t_s,node,x,y,z (node = placement index, 0 = TX; matches
  ns3unet_spectrum.py --mobility_trace_in).

Grid conventions (must match training + server):
  walls_2d[i, j] == 1 is a wall cell; x = (i + 0.5)*cell, y = (j + 0.5)*cell.
  cell = 0.625 m; 64x64 cells = one 40 m training patch.

Usage:
  python gen_validation_suite.py --out_root worldbuilding/valsuite
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

from mlink.geometry import walls_to_mesh
from mesh.xml_io import export_walls_floor_ceiling_xml, RadioMaterialSpec


# ------------------------------------------------------------------
# small helpers
# ------------------------------------------------------------------

def make_double_sided(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Duplicate the mesh with inverted winding so RT hits faces from
    both sides (same intent as gen_random_suite --double_sided)."""
    inv = mesh.copy()
    inv.invert()
    return trimesh.util.concatenate([mesh, inv])


def cell_to_m(i: float, j: float, cell: float) -> tuple[float, float]:
    """Grid cell (i=row -> x, j=col -> y) to meters at the cell center."""
    return (float(i) + 0.5) * cell, (float(j) + 0.5) * cell


def empty_grid(H: int, W: int) -> np.ndarray:
    """Perimeter walls only."""
    g = np.zeros((H, W), dtype=np.uint8)
    g[0, :] = 1
    g[-1, :] = 1
    g[:, 0] = 1
    g[:, -1] = 1
    return g


def partition_col(g: np.ndarray, j: int, door_i0: int | None, door_len: int):
    """Full-height wall at column j, optional door of door_len cells
    starting at row door_i0."""
    g[:, j] = 1
    if door_i0 is not None:
        g[door_i0:door_i0 + door_len, j] = 0


def partition_row(g: np.ndarray, i: int, door_j0: int | None, door_len: int):
    g[i, :] = 1
    if door_j0 is not None:
        g[i, door_j0:door_j0 + door_len] = 0


def wall_crossings(g: np.ndarray, i0, j0, i1, j1) -> int:
    """Count wall cells intersected by the straight segment between two
    cell centers (same rasterization style as the mlink wall feature).
    Used only to ANNOTATE expected LOS depth in expected.json."""
    n = int(max(abs(i1 - i0), abs(j1 - j0), 1)) + 1
    ii = np.linspace(i0, i1, n)
    jj = np.linspace(j0, j1, n)
    cells = set(zip(np.round(ii).astype(int), np.round(jj).astype(int)))
    return int(sum(g[i, j] for (i, j) in cells
                   if 0 <= i < g.shape[0] and 0 <= j < g.shape[1]))


def write_trace(path: Path, rows: list[tuple[float, int, float, float, float]]):
    """rows: (t_s, node, x, y, z). Node ids follow placement order, 0=TX.
    Header matches ns3unet_spectrum.py::_read_mobility_trace_csv."""
    lines = ["t_s,node,x,y,z"]
    for t, n, x, y, z in rows:
        lines.append(f"{t:.3f},{n},{x:.4f},{y:.4f},{z:.4f}")
    path.write_text("\n".join(lines) + "\n")


def waypoint_walk(node: int, z: float, waypoints_m: list[tuple[float, float]],
                  speed_mps: float, dt_s: float, t0_s: float = 0.0,
                  dwell_s: float = 0.0) -> list[tuple[float, int, float, float, float]]:
    """Piecewise-linear walk through waypoints at constant speed, sampled
    every dt_s. A dwell is inserted at every waypoint after the first."""
    rows = []
    t = t0_s
    x, y = waypoints_m[0]
    rows.append((t, node, x, y, z))
    for (nx, ny) in waypoints_m[1:]:
        d = float(np.hypot(nx - x, ny - y))
        steps = max(int(np.ceil(d / (speed_mps * dt_s))), 1)
        for s in range(1, steps + 1):
            a = s / steps
            t += dt_s
            rows.append((t, node, x + a * (nx - x), y + a * (ny - y), z))
        x, y = nx, ny
        if dwell_s > 0:
            t += dwell_s
            rows.append((t, node, x, y, z))
    return rows


def static_rows(node: int, x: float, y: float, z: float,
                t_end_s: float) -> list[tuple[float, int, float, float, float]]:
    return [(0.0, node, x, y, z), (t_end_s, node, x, y, z)]


# ------------------------------------------------------------------
# scene writer (gen_random_suite-compatible layout)
# ------------------------------------------------------------------

def write_scene(scene_dir: Path, name: str, walls_2d: np.ndarray, args,
                tx_cell: tuple[float, float], sta_cells: list[tuple[float, float]],
                expected: dict, trace_rows=None, sta_notes=None):
    scene_dir.mkdir(parents=True, exist_ok=True)
    cell = args.cell_size_m

    np.save(scene_dir / "walls_2d.npy", walls_2d.astype(np.uint8))

    # mesh + XML (heights in grid units, scaled afterwards -- same as
    # training and gen_random_suite)
    floor_h_units = float(args.floor_h_units)
    ceil_h_units = float(args.ceil_h_units)
    mesh = walls_to_mesh(walls_2d.astype(np.float32),
                         floor_height=floor_h_units,
                         ceiling_height=ceil_h_units).apply_scale(cell)
    if args.double_sided:
        mesh = make_double_sided(mesh)

    mat = RadioMaterialSpec(
        bsdf_id="mat-radio_uniform",
        relative_permittivity=float(args.eps_r),
        conductivity=float(args.sigma_s_m),
        thickness=float(args.thickness_m),
    )
    xml_path = export_walls_floor_ceiling_xml(
        mesh,
        out_dir=scene_dir,
        xml_name="scene.xml",
        wall_material=mat,
        floor_material=mat,
        ceiling_material=mat,
    )

    floor_h_m = floor_h_units * cell
    ceil_h_m = ceil_h_units * cell

    tx_x, tx_y = cell_to_m(*tx_cell, cell)
    tx_xyz = [tx_x, tx_y, float(args.tx_h_m)]
    sta_xyz = []
    for (si, sj) in sta_cells:
        sx, sy = cell_to_m(si, sj, cell)
        sta_xyz.append([sx, sy, float(args.rx_h_m)])

    placements = {
        "frequency_hz": float(args.frequency_hz),
        "cell_size_m": float(cell),
        "floor_h_m": floor_h_m,
        "ceil_h_m": ceil_h_m,
        "tx_xyz": tx_xyz,
        "sta_xyz": sta_xyz,
    }
    (scene_dir / "placements.json").write_text(json.dumps(placements, indent=2))

    csv_lines = [f"tx,{tx_xyz[0]},{tx_xyz[1]},{tx_xyz[2]}"]
    for s in sta_xyz:
        csv_lines.append(f"sta,{s[0]},{s[1]},{s[2]}")
    (scene_dir / "placements.csv").write_text("\n".join(csv_lines) + "\n")

    scene_info = {
        "name": name,
        "suite": "validation",
        "xml": xml_path.name,
        "material_mode": "radio",
        "materials": {"walls": mat.__dict__, "floor": mat.__dict__, "ceiling": mat.__dict__},
        "height_units": {"floor": floor_h_units, "ceiling": ceil_h_units},
        "height_m": {"floor": floor_h_m, "ceiling": ceil_h_m},
    }
    (scene_dir / "scene_info.json").write_text(json.dumps(scene_info, indent=2))

    # annotate per-STA wall-crossing depth (straight-line; a detour path
    # may exist, that's the point of several scenes)
    depth = []
    for k, (si, sj) in enumerate(sta_cells):
        d = wall_crossings(walls_2d, tx_cell[0], tx_cell[1], si, sj)
        note = (sta_notes[k] if sta_notes else "")
        depth.append({"sta_index": k, "node_id": k + 1,
                      "straightline_wall_cells": d, "note": note})
    expected = dict(expected)
    expected["per_sta"] = depth
    (scene_dir / "expected.json").write_text(json.dumps(expected, indent=2))

    if trace_rows is not None:
        write_trace(scene_dir / "mobility_trace.csv", sorted(trace_rows))

    n_mob = len({r[1] for r in trace_rows}) if trace_rows else 0
    print(f"[ok] {scene_dir.name}: {walls_2d.shape[0]}x{walls_2d.shape[1]} cells, "
          f"{len(sta_cells)} STAs, mobile nodes in trace: {n_mob}")


# ------------------------------------------------------------------
# themes
# ------------------------------------------------------------------

def build_v01_open_los(args, out_root: Path):
    H = W = 64
    g = empty_grid(H, W)
    tx = (16.0, 32.0)
    stas = [(8, 8), (8, 56), (32, 8), (32, 56), (48, 16), (48, 48), (56, 32), (24, 44)]
    expected = {
        "theme": "open_los",
        "claim": "LOS gate forces r=0, so UNet wb == cost wb == FSPL exactly "
                 "for every link; network stats must match the cost231 run to "
                 "within CFR-synthesis noise.",
        "pass_criterion": "per-pair |thr_unet - thr_cost231| ~ 0 and both track RT.",
    }
    write_scene(out_root / "v01_open_los", "v01_open_los", g, args, tx, stas, expected)


def build_v02_single_wall(args, out_root: Path):
    H = W = 64
    g = empty_grid(H, W)
    door_i0, door_len = 8, 4                # door near the i=8 edge
    partition_col(g, 32, door_i0, door_len)
    tx = (32.0, 16.0)                       # left half, centered in i
    stas = [
        (24, 10), (44, 20),                 # LOS side
        (12, 40),                           # far side, close to door (short detour)
        (32, 40),                           # far side, mid detour
        (52, 44),                           # far side, long detour
        (56, 58),                           # far corner, longest detour
    ]
    notes = ["LOS", "LOS", "NLOS short detour via door", "NLOS mid detour",
             "NLOS long detour", "NLOS longest detour"]
    expected = {
        "theme": "single_wall_door",
        "claim": "cost charges the through-wall loss on the far side while RT "
                 "leaks through the door; residual head should close the gap. "
                 "Expect unet closer to RT than cost231 on far-side pairs, and "
                 "identical to cost231 on LOS pairs.",
        "door_cells": {"col_j": 32, "rows_i": [door_i0, door_i0 + door_len - 1]},
    }
    write_scene(out_root / "v02_single_wall", "v02_single_wall", g, args, tx, stas,
                expected, sta_notes=notes)


def build_v03_deep_nlos(args, out_root: Path):
    H = W = 64
    g = empty_grid(H, W)
    partition_col(g, 16, 52, 4)             # door far from...
    partition_col(g, 32, 8, 4)              # ...this door (zigzag)
    partition_col(g, 48, 52, 4)
    tx = (32.0, 8.0)
    stas = [
        (20, 12), (44, 12),                 # depth 0 (LOS)
        (20, 24), (44, 24),                 # depth 1
        (20, 40), (44, 40),                 # depth 2
        (20, 56), (44, 56),                 # depth 3
    ]
    notes = ["depth0", "depth0", "depth1", "depth1", "depth2", "depth2",
             "depth3", "depth3"]
    expected = {
        "theme": "deep_nlos_strata",
        "claim": "network-level analog of the per-epoch LOS/NLOS diagnostic: "
                 "unet should beat cost231 (closer to RT) increasingly with "
                 "depth; must never be WORSE than cost231 in any stratum "
                 "(the residual floor).",
    }
    write_scene(out_root / "v03_deep_nlos", "v03_deep_nlos", g, args, tx, stas,
                expected, sta_notes=notes)


def build_v04_sealed_pocket(args, out_root: Path):
    H = W = 64
    g = empty_grid(H, W)
    # 10x10-cell fully sealed closet in the far corner (NO door)
    partition_row_i, partition_col_j = 52, 52
    g[partition_row_i, partition_col_j:] = 1
    g[partition_row_i:, partition_col_j] = 1
    tx = (16.0, 16.0)
    stas = [
        (16, 40), (40, 16), (40, 40),       # normal links
        (58, 58),                           # inside the sealed closet
    ]
    notes = ["LOS", "LOS", "LOS", "inside sealed closet (expect ~no-path in RT)"]
    expected = {
        "theme": "sealed_pocket_coverage",
        "claim": "RT should show no-path (or sentinel-level loss) into the "
                 "closet; cost231 reports finite multi-wall loss. Coverage "
                 "head must emit the sentinel so the unet flow to STA idx 3 "
                 "carries ~zero throughput like RT. This is the scene to "
                 "sweep --unet_cov_thresh on (0.3 / 0.5 / 0.7).",
        "sealed_sta_index": 3,
    }
    write_scene(out_root / "v04_sealed_pocket", "v04_sealed_pocket", g, args,
                tx, stas, expected, sta_notes=notes)


def build_v05_door_crossing(args, out_root: Path):
    """v02 geometry + a designed walk: STA 1 (node id 1) starts LOS,
    crosses the door into deep NLOS, dwells, and returns."""
    H = W = 64
    g = empty_grid(H, W)
    door_i0, door_len = 8, 4
    partition_col(g, 32, door_i0, door_len)
    cell = args.cell_size_m
    tx = (32.0, 16.0)
    stas = [
        (24, 24),                           # node 1: the walker (initial pos)
        (44, 20),                           # node 2: static LOS anchor
        (52, 44),                           # node 3: static NLOS anchor
    ]
    notes = ["walker (see mobility_trace.csv)", "static LOS anchor",
             "static NLOS anchor"]

    z = float(args.rx_h_m)
    door_mid_i = door_i0 + door_len / 2.0   # 10.0
    wp = [cell_to_m(24, 24, cell),          # LOS start
          cell_to_m(door_mid_i, 28, cell),  # approach door (still LOS side)
          cell_to_m(door_mid_i, 36, cell),  # THROUGH the door
          cell_to_m(44, 48, cell)]          # deep NLOS
    speed, dt = 0.8, 0.25                   # within trained 0.2-1.0 m/s range
    rows = waypoint_walk(1, z, wp, speed, dt, dwell_s=3.0)
    t_far = rows[-1][0]
    rows += [(r[0] + t_far + 3.0, 1, r[2], r[3], r[4])
             for r in waypoint_walk(1, z, list(reversed(wp)), speed, dt)]
    t_end = rows[-1][0] + 3.0

    # static nodes: TX (0) and anchors (2, 3)
    tx_m = cell_to_m(*tx, cell)
    rows += static_rows(0, tx_m[0], tx_m[1], float(args.tx_h_m), t_end)
    for nid, (si, sj) in ((2, stas[1]), (3, stas[2])):
        sx, sy = cell_to_m(si, sj, cell)
        rows += static_rows(nid, sx, sy, z, t_end)

    # door-crossing times (walker passes column j=32) for transition scoring.
    # Signs are propagated through samples that land exactly ON the door
    # plane so those crossings are not missed.
    door_y_m = (32 + 0.5) * cell
    walker = sorted([r for r in rows if r[1] == 1])
    ts_w = np.array([r[0] for r in walker])
    ys_w = np.array([r[3] for r in walker])
    sgn = np.sign(ys_w - door_y_m)
    for k in range(1, len(sgn)):
        if sgn[k] == 0:
            sgn[k] = sgn[k - 1]
    cross = []
    for k in np.where(sgn[1:] != sgn[:-1])[0]:
        y0, y1 = ys_w[k], ys_w[k + 1]
        frac = 0.5 if y1 == y0 else (door_y_m - y0) / (y1 - y0)
        cross.append(round(float(ts_w[k] + frac * (ts_w[k + 1] - ts_w[k])), 2))

    expected = {
        "theme": "door_crossing_activity",
        "claim": "windowed throughput/active flag for node 1 must transition "
                 "at the same times as the RT run (link-activity detection). "
                 "cost231 will transition too early/hard (wall loss applied "
                 "the moment the straight line clips the partition); the "
                 "residual+coverage model should match RT's softer door "
                 "transition.",
        "walker_node_id": 1,
        "door_crossing_times_s": cross,
        "sim_time_hint_s": float(np.ceil(t_end)),
        "run_note": "run ALL models with --mobilityTraceIn=mobility_trace.csv, "
                    "enableMobility=0, pokeSionnaPositions=1 so positions are "
                    "identical across RT/unet/cost231/friis.",
    }
    write_scene(out_root / "v05_door_crossing", "v05_door_crossing", g, args,
                tx, stas, expected, trace_rows=rows, sta_notes=notes)


def build_v06_bigscene_tiling(args, out_root: Path):
    """112x112 cells = 70 m: bigger than one 64-cell patch in both axes,
    so the server enables tiling. Distant STAs sit in patches that do NOT
    contain the TX (off-patch inference)."""
    H = W = 112
    g = empty_grid(H, W)
    partition_col(g, 40, 20, 4)
    partition_col(g, 72, 88, 4)
    partition_row(g, 56, 56, 6)
    cell = args.cell_size_m
    tx = (56.0, 56.0)                       # scene center (35 m, 35 m)

    # STA 0 (node 1) is the seam walker: its placement equals the trace
    # start so placements.csv and mobility_trace.csv agree at t=0.
    stas = [(56.0, 56.0 + 16.0)]
    notes = ["seam walker (see mobility_trace.csv)"]
    # ring placements: radii in cells (8=5m in-patch ... 52=32.5m off-patch)
    for r_cells, tag in ((8, "in-patch"), (24, "in-patch edge"),
                         (40, "off-patch"), (52, "off-patch far")):
        for ang in (0, 90, 180, 270):
            a = np.deg2rad(ang)
            si = float(np.clip(tx[0] + r_cells * np.cos(a), 2, H - 3))
            sj = float(np.clip(tx[1] + r_cells * np.sin(a), 2, W - 3))
            if g[int(round(si)), int(round(sj))]:
                si += 2.0                   # nudge off a wall cell
            stas.append((si, sj))
            notes.append(f"r={r_cells}c ({tag}), az={ang}")

    # walker: radial escape from 10 m to 32.5 m crossing patch seams
    # (patch stride = 32 cells = 20 m from bbox_min)
    z = float(args.rx_h_m)
    wp = [cell_to_m(56, 56 + 16, cell), cell_to_m(56, 56 + 52, cell)]
    rows = waypoint_walk(1, z, wp, speed_mps=0.8, dt_s=0.25)
    t_end = rows[-1][0] + 2.0
    tx_m = cell_to_m(*tx, cell)
    rows += static_rows(0, tx_m[0], tx_m[1], float(args.tx_h_m), t_end)
    for nid, (si, sj) in enumerate(stas[1:], start=2):
        sx, sy = cell_to_m(si, sj, cell)
        rows += static_rows(nid, sx, sy, z, t_end)

    expected = {
        "theme": "bigscene_tiling_offpatch",
        "claim": "distant STAs exercise the tiling path with the TX outside "
                 "the 64x64 patch (the off-patch regime the finetune store "
                 "targets). Watch for (a) wb discontinuities as the walker "
                 "crosses patch seams every 20 m, (b) systematic bias on "
                 "off-patch vs in-patch rings. cost231 has no tiling issue, "
                 "so unet-minus-cost231 disagreement localized at seams "
                 "isolates tiling artifacts from model error.",
        "patch_stride_m": 32 * cell,
        "walker_node_id": 1,
        "sim_time_hint_s": float(np.ceil(t_end)),
    }
    write_scene(out_root / "v06_bigscene_tiling", "v06_bigscene_tiling", g, args,
                tx, stas, expected, trace_rows=rows, sta_notes=notes)


# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--themes", type=str, default="all",
                    help="comma list from: v01,v02,v03,v04,v05,v06 or 'all'")

    # Training-matching defaults (same values as gen_random_suite.py)
    ap.add_argument("--cell_size_m", type=float, default=0.625)
    ap.add_argument("--floor_h_units", type=float, default=0.0)
    ap.add_argument("--ceil_h_units", type=float, default=8.0,
                    help="ceiling in grid units (8*0.625 = 5 m; training "
                         "sampled 8..20)")
    ap.add_argument("--tx_h_m", type=float, default=1.5)
    ap.add_argument("--rx_h_m", type=float, default=1.5)
    ap.add_argument("--frequency_hz", type=float, default=5.21e9)
    ap.add_argument("--eps_r", type=float, default=4.0)
    ap.add_argument("--sigma_s_m", type=float, default=0.01)
    ap.add_argument("--thickness_m", type=float, default=0.10)
    ap.add_argument("--double_sided", action="store_true", default=True)
    ap.add_argument("--no_double_sided", dest="double_sided", action="store_false")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    builders = {
        "v01": build_v01_open_los,
        "v02": build_v02_single_wall,
        "v03": build_v03_deep_nlos,
        "v04": build_v04_sealed_pocket,
        "v05": build_v05_door_crossing,
        "v06": build_v06_bigscene_tiling,
    }
    themes = list(builders) if args.themes == "all" else args.themes.split(",")
    for t in themes:
        builders[t.strip()](args, out_root)

    print(f"\n[done] validation suite written to: {out_root}")


if __name__ == "__main__":
    main()