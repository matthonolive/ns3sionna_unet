import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import trimesh

from mesh.xml_io import load_ns3sionna_xml_meshes, load_ns3sionna_xml_as_mlink_scene

def summarize_mesh(mesh: trimesh.Trimesh, name="mesh"):
    (xmin, ymin, zmin), (xmax, ymax, zmax) = mesh.bounds
    print(f"\n[{name}]")
    print("  vertices:", len(mesh.vertices))
    print("  faces:", len(mesh.faces))
    print("  bounds:")
    print(f"    x: {xmin:.3f} .. {xmax:.3f}  (span {xmax-xmin:.3f})")
    print(f"    y: {ymin:.3f} .. {ymax:.3f}  (span {ymax-ymin:.3f})")
    print(f"    z: {zmin:.3f} .. {zmax:.3f}  (span {zmax-zmin:.3f})")

    # connected components
    parts = mesh.split(only_watertight=False)
    sizes = sorted([p.faces.shape[0] for p in parts], reverse=True)
    print("  connected components:", len(parts))
    if sizes:
        print("  largest components (faces):", sizes[:10])

    # floor/ceiling face counts (rough)
    z = mesh.vertices[:, 2]
    zmin = float(z.min()); zmax = float(z.max())
    fz = z[mesh.faces]
    eps = 1e-5
    is_floor = np.all(np.isclose(fz, zmin, atol=eps), axis=1)
    is_ceil  = np.all(np.isclose(fz, zmax, atol=eps), axis=1)
    print("  approx floor faces:", int(is_floor.sum()))
    print("  approx ceiling faces:", int(is_ceil.sum()))
    print("  approx wall/other faces:", int((~(is_floor | is_ceil)).sum()))

def list_xml_shapes(xml_path: Path):
    xml_path = Path(xml_path)
    base = xml_path.parent
    print("\n[XML shapes]")
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    shapes = root.findall(".//shape")
    if not shapes:
        print("  (no <shape> tags found)")
        return

    for i, shape in enumerate(shapes):
        stype = shape.get("type")
        fn = shape.find("./string[@name='filename']")
        ref = shape.find("./ref[@name='bsdf']")
        rel = fn.get("value") if fn is not None else None
        bsdf = ref.get("id") if ref is not None else None

        if stype != "obj":
            continue

        mesh_path = (base / rel).resolve() if rel else None
        ok = mesh_path.exists() if mesh_path else False
        print(f"  {i:02d}: type=obj  file={rel}  exists={ok}  bsdf={bsdf}")

def main():
    if len(sys.argv) < 2:
        print("usage: python verify_xml_load.py path/to/scene.xml")
        raise SystemExit(2)

    xml_path = Path(sys.argv[1])

    # 1) Check XML references
    list_xml_shapes(xml_path)

    # 2) Load referenced OBJs directly (no Sionna)
    meshes = load_ns3sionna_xml_meshes(xml_path)
    print("\n[Direct OBJ loads]")
    for i, (m, bsdf) in enumerate(meshes):
        summarize_mesh(m, name=f"obj[{i}] bsdf={bsdf}")

    # 3) Load through Sionna -> mlink Scene (your pipeline)
    print("\n[Sionna->mlink load]")
    scene = load_ns3sionna_xml_as_mlink_scene(xml_path, frequency_hz=2e9)
    summarize_mesh(scene.mesh, name="scene.mesh")

    # 4) Check face2material distribution if available
    if getattr(scene, "face2material", None):
        vals = list(scene.face2material.values())
        uniq = sorted(set(vals))
        print("\n[face2material]")
        print("  unique material ids:", uniq)
        for mid in uniq:
            cnt = sum(v == mid for v in vals)
            print(f"  material {mid}: {cnt} faces")

    scene.mesh.show()

if __name__ == "__main__":
    main()
