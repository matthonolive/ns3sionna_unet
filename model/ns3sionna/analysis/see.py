#!/usr/bin/env python3
"""
verify_xml_load.py

Sanity-check / summarize / optionally visualize a Mitsuba XML scene used by Sionna RT.

Features:
- Lists <shape type="obj"> entries, verifies referenced OBJ files exist
- Summarizes BSDF/material definitions (especially itu-radio-material parameters)
- Loads referenced OBJs directly via trimesh and prints bounds/components/floor-ceil face counts
- Optionally loads the scene through sionna.rt.load_scene and optionally converts to mlink.Scene

Works with both XML styles:
- <ref name="bsdf" id="..."/>  (older)
- <ref id="..."/>             (new from-scratch exporter)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh


# ----------------------------
# Mesh summarization helpers
# ----------------------------

def summarize_mesh(mesh: trimesh.Trimesh, name: str = "mesh") -> None:
    if mesh.vertices is None or len(mesh.vertices) == 0:
        print(f"\n[{name}] EMPTY")
        return

    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    print(f"\n[{name}]")
    print("  vertices:", int(len(mesh.vertices)))
    print("  faces:", int(len(mesh.faces)))
    print("  bounds:")
    print(f"    x: {xmin:.3f} .. {xmax:.3f}  (span {xmax-xmin:.3f})")
    print(f"    y: {ymin:.3f} .. {ymax:.3f}  (span {ymax-ymin:.3f})")
    print(f"    z: {zmin:.3f} .. {zmax:.3f}  (span {zmax-zmin:.3f})")

    # connected components
    try:
        parts = mesh.split(only_watertight=False)
        sizes = sorted([int(p.faces.shape[0]) for p in parts], reverse=True)
        print("  connected components:", int(len(parts)))
        if sizes:
            print("  largest components (faces):", sizes[:10])
    except Exception as e:
        print("  connected components: (failed)", repr(e))

    # floor/ceiling face counts (rough)
    z = np.asarray(mesh.vertices)[:, 2]
    zmin = float(z.min())
    zmax = float(z.max())
    fz = z[np.asarray(mesh.faces)]
    eps = 1e-5
    is_floor = np.all(np.isclose(fz, zmin, atol=eps), axis=1)
    is_ceil = np.all(np.isclose(fz, zmax, atol=eps), axis=1)

    print("  approx floor faces:", int(is_floor.sum()))
    print("  approx ceiling faces:", int(is_ceil.sum()))
    print("  approx wall/other faces:", int((~(is_floor | is_ceil)).sum()))


# ----------------------------
# XML parsing helpers
# ----------------------------

def _get_shape_filename(shape: ET.Element) -> str | None:
    fn = shape.find("./string[@name='filename']")
    return fn.get("value") if fn is not None else None


def _get_shape_bsdf_id(shape: ET.Element) -> str | None:
    """
    Supports both:
      <ref name="bsdf" id="mat-..."/>
      <ref id="mat-..."/>
    """
    ref = shape.find("./ref[@name='bsdf']")
    if ref is not None:
        return ref.get("id")

    # New exporter: <ref id="..."/>
    ref2 = shape.find("./ref")
    if ref2 is not None:
        return ref2.get("id")

    return None


def list_xml_shapes(xml_path: Path) -> list[tuple[str, str | None, str | None]]:
    """
    Returns list of (shape_id_or_index, rel_obj_path, bsdf_id)
    and prints a summary table.
    """
    xml_path = Path(xml_path)
    base = xml_path.parent

    print("\n[XML shapes]")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    shapes = root.findall(".//shape")
    if not shapes:
        print("  (no <shape> tags found)")
        return []

    entries: list[tuple[str, str | None, str | None]] = []
    for i, shape in enumerate(shapes):
        stype = shape.get("type", "")
        if stype != "obj":
            continue

        sid = shape.get("id", f"shape[{i:02d}]")
        rel = _get_shape_filename(shape)
        bsdf_id = _get_shape_bsdf_id(shape)

        mesh_path = (base / rel).resolve() if rel else None
        ok = bool(mesh_path and mesh_path.exists())
        print(f"  {sid:>10s}: file={rel}  exists={ok}  bsdf={bsdf_id}")

        entries.append((sid, rel, bsdf_id))

    return entries


def list_xml_materials(xml_path: Path) -> None:
    """
    Print material/BSDF declarations, focusing on radio materials.
    """
    print("\n[XML materials]")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    bsdfs = root.findall(".//bsdf")
    if not bsdfs:
        print("  (no <bsdf> tags found)")
        return

    # Only show top-level bsdfs (direct children of <scene>) if present; otherwise show all
    scene = root if root.tag == "scene" else root.find(".//scene")
    top_bsdfs = []
    if scene is not None:
        top_bsdfs = [b for b in list(scene) if b.tag == "bsdf"]
    bsdfs_to_show = top_bsdfs if top_bsdfs else bsdfs

    for b in bsdfs_to_show:
        btype = b.get("type", "")
        bid = b.get("id", "(no-id)")

        if btype == "itu-radio-material":
            itu_type = None
            thickness = None
            sc = None
            xpd = None
            for child in list(b):
                if child.tag == "string" and child.get("name") == "type":
                    itu_type = child.get("value")
                if child.tag == "float" and child.get("name") == "thickness":
                    thickness = child.get("value")
                if child.tag == "float" and child.get("name") == "scattering_coefficient":
                    sc = child.get("value")
                if child.tag == "float" and child.get("name") == "xpd_coefficient":
                    xpd = child.get("value")

            print(f"  {bid}: itu-radio-material(type={itu_type}, thickness={thickness}, sc={sc}, xpd={xpd})")

        elif btype == "radio-material":
            # in case you ever use explicit radio-material
            epsr = sigma = thickness = None
            for child in list(b):
                if child.tag == "float" and child.get("name") == "relative_permittivity":
                    epsr = child.get("value")
                if child.tag == "float" and child.get("name") == "conductivity":
                    sigma = child.get("value")
                if child.tag == "float" and child.get("name") == "thickness":
                    thickness = child.get("value")
            print(f"  {bid}: radio-material(eps_r={epsr}, sigma={sigma}, thickness={thickness})")

        else:
            print(f"  {bid}: bsdf(type={btype})")


def load_xml_meshes_direct(xml_path: Path) -> list[tuple[str, trimesh.Trimesh, str | None]]:
    """
    Load referenced OBJ meshes directly (no Sionna).
    Returns list of (shape_id, mesh, bsdf_id).
    """
    xml_path = Path(xml_path)
    base = xml_path.parent

    entries = list_xml_shapes(xml_path)
    out: list[tuple[str, trimesh.Trimesh, str | None]] = []

    for sid, rel, bsdf_id in entries:
        if rel is None:
            continue
        mesh_path = (base / rel).resolve()
        if not mesh_path.exists():
            continue
        mesh = trimesh.load_mesh(mesh_path, process=False)
        out.append((sid, mesh, bsdf_id))

    return out


# ----------------------------
# Sionna / mlink optional load
# ----------------------------

def try_load_sionna(xml_path: Path, frequency_hz: float | None) -> object | None:
    try:
        from sionna.rt import load_scene  # type: ignore
    except Exception as e:
        print("\n[Sionna load] (skipped: could not import sionna.rt)", repr(e))
        return None

    print("\n[Sionna load]")
    sc = load_scene(str(xml_path))
    if frequency_hz is not None:
        sc.frequency = float(frequency_hz)
        print(f"  set scene.frequency = {sc.frequency}")
    print("  ok: load_scene() succeeded")
    try:
        obj_keys = list(sc.objects.keys())  # type: ignore
        print("  objects:", obj_keys[:10], ("..." if len(obj_keys) > 10 else ""))
    except Exception:
        pass
    try:
        mat_keys = list(sc.radio_materials.keys())  # type: ignore
        print("  radio_materials:", mat_keys[:10], ("..." if len(mat_keys) > 10 else ""))
    except Exception:
        pass
    return sc


def try_load_mlink(xml_path: Path, frequency_hz: float) -> object | None:
    """
    Attempts:
      from mesh.xml_io import load_xml_as_mlink_scene
    Falls back gracefully if unavailable.
    """
    try:
        from mesh.xml_io import load_xml_as_mlink_scene  # type: ignore
    except Exception as e:
        print("\n[Sionna->mlink load] (skipped: mesh.xml_io.load_xml_as_mlink_scene unavailable)", repr(e))
        return None

    print("\n[Sionna->mlink load]")
    scene = load_xml_as_mlink_scene(xml_path, frequency_hz=frequency_hz)
    print("  ok: converted to mlink.Scene")
    # Try to summarize combined mesh if present
    m = getattr(scene, "mesh", None)
    if isinstance(m, trimesh.Trimesh):
        summarize_mesh(m, name="scene.mesh")
    return scene


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml", type=str, help="path/to/scene.xml")
    ap.add_argument("--frequency", type=float, default=None, help="set scene.frequency when loading through Sionna (Hz)")
    ap.add_argument("--no_sionna", action="store_true", help="skip sionna.rt.load_scene step")
    ap.add_argument("--mlink", action="store_true", help="also try to convert to mlink.Scene (requires load_xml_as_mlink_scene)")
    ap.add_argument("--show", action="store_true", help="open trimesh viewer for the direct-loaded meshes")
    args = ap.parse_args()

    xml_path = Path(args.xml).resolve()
    if not xml_path.exists():
        print(f"ERROR: XML not found: {xml_path}")
        raise SystemExit(2)

    print(f"[XML] {xml_path}")

    # 1) Materials summary
    list_xml_materials(xml_path)

    # 2) Shapes + direct OBJ loads
    meshes = load_xml_meshes_direct(xml_path)
    print("\n[Direct OBJ loads]")
    if not meshes:
        print("  (no OBJ meshes loaded)")
    for sid, m, bsdf in meshes:
        summarize_mesh(m, name=f"{sid} (bsdf={bsdf})")

    # 3) Sionna load
    sc = None
    if not args.no_sionna:
        sc = try_load_sionna(xml_path, args.frequency)

    # 4) Optional mlink conversion
    if args.mlink:
        if args.frequency is None:
            print("\n[mlink] ERROR: please provide --frequency for mlink conversion (e.g., --frequency 2e9)")
        else:
            try_load_mlink(xml_path, frequency_hz=float(args.frequency))

    # 5) Optional visualization
    if args.show:
        print("\n[Viewer]")
        for sid, m, bsdf in meshes:
            try:
                m.show()
            except Exception as e:
                print(f"  viewer failed for {sid}: {e!r}")
            break

    print("\n[done]")


if __name__ == "__main__":
    main()
