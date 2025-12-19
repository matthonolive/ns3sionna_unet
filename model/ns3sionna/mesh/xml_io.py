from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import polars as pl
import trimesh

from mlink.scene import Scene
from mlink.antenna import AntennaDatabase


# ----------------------------
# Helpers
# ----------------------------

def _triangulate_if_quads(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Mitsuba prefers triangles. Your walls_to_mesh() produces quads."""
    if mesh.faces.ndim == 2 and mesh.faces.shape[1] == 4:
        tri = trimesh.geometry.triangulate_quads(mesh.faces)
        return trimesh.Trimesh(vertices=mesh.vertices.copy(), faces=tri, process=False)
    return mesh


def split_floor_ceiling_walls(mesh: trimesh.Trimesh, eps: float = 1e-6):
    """
    Split a mesh into (floor, ceiling, walls) by looking at face vertex z.
    Works well for meshes created by walls_to_mesh() where floor/ceiling are flat.
    """
    m = mesh
    z = m.vertices[:, 2]
    zmin = float(z.min())
    zmax = float(z.max())

    fz = z[m.faces]  # (F,3) or (F,4)
    is_floor = np.all(np.isclose(fz, zmin, atol=eps), axis=1)
    is_ceil  = np.all(np.isclose(fz, zmax, atol=eps), axis=1)
    is_wall  = ~(is_floor | is_ceil)

    def submask(mask):
        idx = np.nonzero(mask)[0]
        if idx.size == 0:
            return None
        sm = m.submesh([idx], append=True, repair=False)
        return _triangulate_if_quads(sm)

    return submask(is_floor), submask(is_ceil), submask(is_wall)


def _write_minimal_mitsuba_xml(
    xml_path: Path,
    shape_entries: list[tuple[str, str, str]],
    # list of (shape_id, mesh_relpath, bsdf_id)
):
    """
    Writes a Mitsuba XML file similar to ns3sionna examples.
    """
    xml_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append('<scene version="2.1.0">')
    lines.append('  <integrator type="path"><integer name="max_depth" value="12"/></integrator>')
    lines.append('')

    # Common ITU-ish materials (same ids as many ns3sionna scenes use)
    # NOTE: These are "visual" diffuse colors; Sionna/ns3sionna often maps ids to radio materials.
    lines.append('  <bsdf type="twosided" id="mat-itu_brick"><bsdf type="diffuse"><rgb name="reflectance" value="0.9 0.63 0.225"/></bsdf></bsdf>')
    lines.append('  <bsdf type="twosided" id="mat-itu_concrete"><bsdf type="diffuse"><rgb name="reflectance" value="0.5 0.5 0.5"/></bsdf></bsdf>')
    lines.append('')
    lines.append('  <emitter type="constant" id="World"><rgb name="radiance" value="1 1 1"/></emitter>')
    lines.append('')

    for shape_id, rel_mesh, bsdf_id in shape_entries:
        lines.append(f'  <shape type="obj" id="{shape_id}">')
        lines.append(f'    <string name="filename" value="{rel_mesh}"/>')
        lines.append('    <boolean name="face_normals" value="true"/>')
        lines.append(f'    <ref name="bsdf" id="{bsdf_id}"/>')
        lines.append('  </shape>')
        lines.append('')

    lines.append('</scene>')
    xml_path.write_text("\n".join(lines), encoding="utf-8")


# ----------------------------
# Export: mesh -> (objs + xml)
# ----------------------------

def export_mesh_as_ns3sionna_xml(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    xml_name: str = "scene.xml",
    mesh_name: str = "scene.obj",
    bsdf_id: str = "mat-itu_concrete",
):
    """
    Export a single-mesh scene:
      out_dir/scene.xml
      out_dir/meshes/scene.obj
    """
    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    m = _triangulate_if_quads(mesh)
    m.export(meshes_dir / mesh_name)

    _write_minimal_mitsuba_xml(
        out_dir / xml_name,
        shape_entries=[("mesh-scene", f"meshes/{mesh_name}", bsdf_id)],
    )


def export_walls_floor_ceiling_xml(
    mesh: trimesh.Trimesh,
    out_dir: Path,
    xml_name: str = "scene.xml",
    wall_bsdf: str = "mat-itu_brick",
    floor_bsdf: str = "mat-itu_concrete",
    ceil_bsdf: str = "mat-itu_concrete",
):
    """
    Export a walls_to_mesh()-style mesh into 3 OBJ files + XML:
      meshes/walls.obj, meshes/floor.obj, meshes/ceiling.obj
    """
    out_dir = Path(out_dir)
    meshes_dir = out_dir / "meshes"
    meshes_dir.mkdir(parents=True, exist_ok=True)

    floor, ceil, walls = split_floor_ceiling_walls(mesh)

    shape_entries = []
    if ceil is not None:
        ceil.export(meshes_dir / "ceiling.obj")
        shape_entries.append(("mesh-ceiling", "meshes/ceiling.obj", ceil_bsdf))
    if floor is not None:
        floor.export(meshes_dir / "floor.obj")
        shape_entries.append(("mesh-floor", "meshes/floor.obj", floor_bsdf))
    if walls is not None:
        walls.export(meshes_dir / "walls.obj")
        shape_entries.append(("mesh-walls", "meshes/walls.obj", wall_bsdf))

    _write_minimal_mitsuba_xml(out_dir / xml_name, shape_entries)


# ----------------------------
# Import: xml -> Scene/mesh
# ----------------------------

def load_ns3sionna_xml_as_mlink_scene(xml_path: Path, frequency_hz: float) -> Scene:
    """
    Best import path for your pipeline:
    - Uses sionna.rt.load_scene(xml)
    - Keeps the loaded sionna scene attached to Scene.sionna_scene
      so cost.py can traverse bsdf radio parameters without losing them.
    """
    from sionna.rt import load_scene  # local import

    xml_path = Path(xml_path)
    si = load_scene(str(xml_path))
    si.frequency = float(frequency_hz)

    sc = Scene.from_sionna(si)
    sc.sionna_scene = si  # IMPORTANT: preserve original radio materials/bssdf params
    return sc


def load_ns3sionna_xml_meshes(xml_path: Path) -> list[tuple[trimesh.Trimesh, str]]:
    """
    Lightweight import (no Sionna):
    - Parses the XML
    - Loads referenced OBJ files
    - Returns [(mesh, bsdf_id), ...]
    """
    xml_path = Path(xml_path)
    base = xml_path.parent

    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    out = []
    for shape in root.findall(".//shape"):
        if shape.get("type") != "obj":
            continue
        fn = shape.find("./string[@name='filename']")
        if fn is None:
            continue
        rel = fn.get("value")
        if rel is None:
            continue

        ref = shape.find("./ref[@name='bsdf']")
        bsdf_id = ref.get("id") if ref is not None else "unknown"

        mesh_path = (base / rel).resolve()
        mesh = trimesh.load_mesh(mesh_path, process=False)
        out.append((mesh, bsdf_id))

    return out