"""
mesh/xml_io.py

Minimal, robust Mitsuba XML exporter for Sionna RT scenes that:
- Writes OBJ meshes to disk
- Writes a Mitsuba XML scene that Sionna can load (sionna.rt.load_scene)
- Uses explicit Sionna radio materials (itu-radio-material) so both:
    (a) ns3sionna raytracer
    (b) your UNet/cost-map pipeline that reads bsdf params
  can agree on material behavior.

Key design constraints:
- DO NOT store Trimesh objects in mesh.metadata (causes deepcopy recursion)
- DO NOT wrap radio materials in twosided (can break Sionna's radio_material detection)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import trimesh
import xml.etree.ElementTree as ET


# ----------------------------
# Material specs
# ----------------------------

@dataclass(frozen=True)
class ItuRadioMaterialSpec:
    """
    ITU radio material as understood by Sionna RT via Mitsuba XML.

    This matches Sionna's documented XML syntax:
      <bsdf type="itu-radio-material" id="...">
        <string name="type" value="concrete"/>
        <float name="thickness" value="0.1"/>
        <float name="scattering_coefficient" value="0.0"/>
        <float name="xpd_coefficient" value="0.0"/>
      </bsdf>

    Notes:
    - thickness is in meters
    - scattering_coefficient and xpd_coefficient are optional but useful for future extensions
    """
    bsdf_id: str                 # e.g., "mat-itu_concrete"
    itu_type: str                # e.g., "concrete", "brick", "glass", "metal", ...
    thickness: float = 0.10
    scattering_coefficient: float = 0.0
    xpd_coefficient: float = 0.0


# ----------------------------
# Utilities
# ----------------------------

def _indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print indentation for ElementTree."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent_xml(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def _safe_export_obj(mesh: trimesh.Trimesh, out_path: Path) -> None:
    """
    Export a Trimesh to OBJ robustly.
    We also clear metadata to avoid any accidental deepcopy recursion.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Avoid recursive metadata graphs
    try:
        mesh.metadata = {}  # type: ignore[attr-defined]
    except Exception:
        pass

    mesh.export(out_path)


def _faces_on_z_plane(mesh: trimesh.Trimesh, z: float, atol: float = 1e-6) -> np.ndarray:
    """
    Return a boolean mask over faces whose all vertices have z ~= given z.
    """
    v = np.asarray(mesh.vertices)
    f = np.asarray(mesh.faces)
    z_verts = v[f, 2]  # (F,3)
    return np.all(np.isclose(z_verts, z, atol=atol), axis=1)


def split_walls_floor_ceiling(mesh: trimesh.Trimesh, atol: float = 1e-6) -> Tuple[trimesh.Trimesh, trimesh.Trimesh, trimesh.Trimesh]:
    """
    Split a single mesh into (walls, floor, ceiling) by detecting the min-z and max-z planes.

    This is designed to work with meshes like mlink.geometry.walls_to_mesh(), which
    include floor/ceiling as planar faces.

    Returns:
        walls_mesh, floor_mesh, ceiling_mesh
    """
    v = np.asarray(mesh.vertices)
    zmin = float(np.min(v[:, 2]))
    zmax = float(np.max(v[:, 2]))

    floor_mask = _faces_on_z_plane(mesh, zmin, atol=atol)
    ceil_mask  = _faces_on_z_plane(mesh, zmax, atol=atol)
    wall_mask  = ~(floor_mask | ceil_mask)

    faces = np.asarray(mesh.faces)
    parts = []
    for mask in (wall_mask, floor_mask, ceil_mask):
        part_faces = faces[mask]
        if part_faces.shape[0] == 0:
            # Empty mesh fallback
            parts.append(trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64), process=False))
        else:
            # trimesh.submesh can copy metadata; keep it simple by rebuilding directly
            used_verts = np.unique(part_faces.reshape(-1))
            index_map = {old: new for new, old in enumerate(used_verts.tolist())}
            new_vertices = v[used_verts]
            new_faces = np.vectorize(index_map.get)(part_faces)
            parts.append(trimesh.Trimesh(vertices=new_vertices, faces=new_faces, process=False))

    return parts[0], parts[1], parts[2]


# ----------------------------
# XML writing
# ----------------------------

def _add_itu_bsdf(scene_el: ET.Element, mat: ItuRadioMaterialSpec) -> None:
    bsdf = ET.SubElement(scene_el, "bsdf", attrib={"type": "itu-radio-material", "id": mat.bsdf_id})
    ET.SubElement(bsdf, "string", attrib={"name": "type", "value": str(mat.itu_type)})
    ET.SubElement(bsdf, "float", attrib={"name": "thickness", "value": f"{float(mat.thickness):.6g}"})
    ET.SubElement(bsdf, "float", attrib={"name": "scattering_coefficient", "value": f"{float(mat.scattering_coefficient):.6g}"})
    ET.SubElement(bsdf, "float", attrib={"name": "xpd_coefficient", "value": f"{float(mat.xpd_coefficient):.6g}"})


def _add_obj_shape(scene_el: ET.Element, obj_relpath: str, bsdf_id: str, shape_id: Optional[str] = None) -> None:
    attrib = {"type": "obj"}
    if shape_id is not None:
        attrib["id"] = shape_id
    shape = ET.SubElement(scene_el, "shape", attrib=attrib)
    ET.SubElement(shape, "string", attrib={"name": "filename", "value": obj_relpath})
    ET.SubElement(shape, "ref", attrib={"id": bsdf_id})


def export_scene_xml_from_parts(
    *,
    out_dir: Path,
    xml_name: str = "scene.xml",
    meshes_subdir: str = "meshes",
    parts: Dict[str, trimesh.Trimesh],
    materials: Dict[str, ItuRadioMaterialSpec],
) -> Path:
    """
    Export a Sionna-loadable Mitsuba XML scene from explicit mesh parts.

    Args:
      out_dir: output directory
      xml_name: name of the xml file
      meshes_subdir: where to put OBJs relative to out_dir
      parts: dict like {"walls": mesh, "floor": mesh, "ceiling": mesh}
      materials: dict like {"walls": ItuRadioMaterialSpec(...), ...} (keys must match parts)

    Returns:
      Path to written XML file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    meshes_dir = out_dir / meshes_subdir
    meshes_dir.mkdir(parents=True, exist_ok=True)

    # --- Build XML ---
    scene_el = ET.Element("scene", attrib={"version": "3.0.0"})

    # Integrator is not important for RT, but Mitsuba XML likes having one.
    integrator = ET.SubElement(scene_el, "integrator", attrib={"type": "path"})
    ET.SubElement(integrator, "integer", attrib={"name": "max_depth", "value": "12"})

    # --- Materials ---
    # De-duplicate by bsdf_id (multiple parts can share a material)
    seen = set()
    for key, mat in materials.items():
        if mat.bsdf_id in seen:
            continue
        _add_itu_bsdf(scene_el, mat)
        seen.add(mat.bsdf_id)

    # --- Shapes (write OBJs and reference them) ---
    for part_name, mesh in parts.items():
        if part_name not in materials:
            raise KeyError(f"Missing material for part '{part_name}'. Have materials={list(materials.keys())}")

        obj_name = f"{part_name}.obj"
        obj_path = meshes_dir / obj_name
        _safe_export_obj(mesh, obj_path)

        obj_rel = f"{meshes_subdir}/{obj_name}"
        _add_obj_shape(scene_el, obj_rel, materials[part_name].bsdf_id, shape_id=part_name)

    _indent_xml(scene_el)
    xml_path = out_dir / xml_name
    ET.ElementTree(scene_el).write(xml_path, encoding="utf-8", xml_declaration=True)
    return xml_path


def export_walls_floor_ceiling_xml(
    mesh: trimesh.Trimesh,
    *,
    out_dir: Path,
    xml_name: str = "scene.xml",
    meshes_subdir: str = "meshes",
    wall_material: ItuRadioMaterialSpec,
    floor_material: Optional[ItuRadioMaterialSpec] = None,
    ceiling_material: Optional[ItuRadioMaterialSpec] = None,
    atol: float = 1e-6,
) -> Path:
    """
    Convenience wrapper: given ONE mesh (e.g., from walls_to_mesh),
    split it into walls/floor/ceiling by z-planes and export an XML scene.

    If floor_material/ceiling_material are None, they reuse wall_material.
    """
    walls_m, floor_m, ceil_m = split_walls_floor_ceiling(mesh, atol=atol)

    parts = {"walls": walls_m, "floor": floor_m, "ceiling": ceil_m}
    mats = {
        "walls": wall_material,
        "floor": floor_material or wall_material,
        "ceiling": ceiling_material or wall_material,
    }
    return export_scene_xml_from_parts(
        out_dir=out_dir,
        xml_name=xml_name,
        meshes_subdir=meshes_subdir,
        parts=parts,
        materials=mats,
    )


# ----------------------------
# Optional: round-trip helpers
# ----------------------------

def load_xml_as_sionna_scene(xml_path: Path, *, frequency_hz: Optional[float] = None):
    """
    Load XML via sionna.rt.load_scene (requires sionna[rt] installed).
    """
    import sionna.rt as rt  # noqa: F401
    sc = rt.load_scene(str(xml_path))
    if frequency_hz is not None:
        sc.frequency = float(frequency_hz)
    return sc


def load_xml_as_mlink_scene(xml_path: Path, *, frequency_hz: float):
    """
    Load XML into Sionna scene, then convert to your internal mlink.scene.Scene.

    Requires mlink.scene.Scene.from_sionna to exist in your repo.
    """
    sc = load_xml_as_sionna_scene(xml_path, frequency_hz=frequency_hz)
    from mlink.scene import Scene  # imported late to avoid hard dependency
    return Scene.from_sionna(sc)
