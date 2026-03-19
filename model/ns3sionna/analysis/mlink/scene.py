from collections import defaultdict
from dataclasses import dataclass
from functools import reduce

import mitsuba as mi
import numpy as np
import polars as pl
import sionna.rt
import trimesh
from trimesh import Trimesh

from mlink.antenna import AntennaDatabase


SIONNA_SCHEMA = {
    "eta_r": "permittivity",
    "sigma": "conductivity",
    "d": "thickness",
    "s": "scattering_coefficient",
}

def _coerce_frequency_hz(frequency) -> tuple[float, int]:
    """Return (float_hz, int_hz) from Sionna/Tensor/np scalars."""
    try:
        f_val = float(frequency)
    except Exception:
        f_val = float(np.asarray(frequency).ravel()[0])
    f_hz_i = int(np.round(f_val))
    return f_val, f_hz_i


@dataclass
class Scene:
    mesh: Trimesh
    material_database: pl.DataFrame
    face2material: dict[int, int]
    antenna_database: AntennaDatabase
    sionna_scene: sionna.rt.Scene | None = None
    sionna_scene_geometry: sionna.rt.Scene | None = None

    def to_sionna(self, frequency: float, tol_hz: float = 1.0):
        # cache check (robust to tensor-ish frequency types)
        if self.sionna_scene is not None:
            try:
                if float(self.sionna_scene.frequency) == float(frequency):
                    return self.sionna_scene
            except Exception:
                pass

        f_val, f_hz_i = _coerce_frequency_hz(frequency)

        # group faces by material id
        material2face = defaultdict(list)
        for face_id, material_id in self.face2material.items():
            material2face[int(material_id)].append(int(face_id))

        df = self.material_database

        # Optional but recommended: ensure frequency_hz exists for stable matching
        if "frequency_hz" not in df.columns:
            if "frequency" in df.columns:
                df = df.with_columns(pl.col("frequency").round(0).cast(pl.Int64).alias("frequency_hz"))
            else:
                df = df.with_columns(pl.lit(f_hz_i).cast(pl.Int64).alias("frequency_hz"))
            self.material_database = df

        def get_material_row(material_id: int) -> pl.DataFrame:
            df = self.material_database

            # 1) exact integer match
            hit = df.filter((pl.col("id") == material_id) & (pl.col("frequency_hz") == f_hz_i)).head(1)
            if hit.height > 0:
                return hit

            # 2) nearest frequency_hz for this id (copy row to requested freq)
            same_id = df.filter(pl.col("id") == material_id)
            if same_id.height > 0:
                best = (
                    same_id.with_columns((pl.col("frequency_hz") - f_hz_i).abs().alias("_df"))
                        .sort("_df")
                        .drop("_df")
                        .head(1)
                )
                row = best.to_dicts()[0]
                row["frequency"] = float(f_val)
                row["frequency_hz"] = int(f_hz_i)
                self.material_database = pl.concat([df, pl.DataFrame([row])], how="vertical_relaxed")
                return pl.DataFrame([row])

            # 3) last resort: create a default row
            row = {
                "id": int(material_id),
                "frequency": float(f_val),
                "frequency_hz": int(f_hz_i),
                "name": str(material_id),
                "thickness": 0.1,
                "permittivity": 4.0,
                "conductivity": 0.01,
            }
            self.material_database = pl.concat([df, pl.DataFrame([row])], how="vertical_relaxed")
            return pl.DataFrame([row])

        meshes = []
        for material_id, face_list in material2face.items():
            material_data = get_material_row(material_id)  # guaranteed 1 row

            sionna_material = sionna.rt.RadioMaterial(
                name=material_data[0, "name"],
                thickness=float(material_data[0, "thickness"]),
                relative_permittivity=float(material_data[0, "permittivity"]),
                conductivity=float(material_data[0, "conductivity"]),
            )

            sub = self.mesh.submesh([face_list], append=False)
            meshes.extend([trimesh2mitsuba(m, sionna_material) for m in sub])

        if len(meshes) == 0:
            raise ValueError("No meshes were produced (check face2material / mesh).")

        # build ONE mitsuba scene
        mi_scene = mi.load_dict(
            {"type": "scene", "integrator": {"type": "path"}}
            | {f"mesh_{i}": mesh for i, mesh in enumerate(meshes)}
        )
        assert isinstance(mi_scene, mi.Scene)

        si_scene = sionna.rt.Scene(mi_scene)
        si_scene.tx_array = sionna.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        si_scene.rx_array = si_scene.tx_array

        # add TX/RX nodes
        for i, tx_coord in enumerate(self.antenna_database.tx_coords):
            si_scene.add(sionna.rt.Transmitter(f"transmitter_{i:03d}", mi.Point3f(tx_coord)))
        for i, rx_coord in enumerate(self.antenna_database.rx_coords):
            si_scene.add(sionna.rt.Receiver(f"receiver_{i:03d}", mi.Point3f(rx_coord)))

        si_scene.frequency = float(f_val)
        self.sionna_scene = si_scene
        return si_scene


    @classmethod
    def from_sionna(cls, scene: sionna.rt.Scene):
        # --- get frequency as a float ---
        freq = scene.frequency
        try:
            freq_val = float(freq)
        except Exception:
            freq_val = float(np.asarray(freq).ravel()[0])

        # --- access underlying Mitsuba scene (Sionna RT stores it here) ---
        mi_scene = getattr(scene, "mi_scene", None)
        if mi_scene is None:
            # fallback for some builds
            mi_scene = getattr(scene, "_scene", None)
        if mi_scene is None:
            raise AttributeError("Could not access underlying Mitsuba scene (scene.mi_scene).")

        # --- extract geometry + per-shape material params from Mitsuba shapes ---
        material_rows = []
        face_to_material = {}
        face_count = 0
        submeshes = []

        shapes = list(mi_scene.shapes())
        for mat_id, shape in enumerate(shapes):
            # Only handle mesh-like shapes
            if not hasattr(shape, "face_count") or not hasattr(shape, "vertex_count"):
                continue

            tmsh = mitsuba2trimesh(shape)
            submeshes.append(tmsh)

            # map faces to this "material id"
            for _ in range(tmsh.faces.shape[0]):
                face_to_material[face_count] = mat_id
                face_count += 1

            # Pull Sionna radio-material parameters from the shape's bsdf
            params = mi.traverse(shape)

            def get_scalar(key: str, default: float) -> float:
                if key not in params:
                    return float(default)
                v = params[key]
                return float(np.asarray(v).ravel()[0])

            # Try to get a nice name (optional)
            name = None
            if hasattr(shape, "id"):
                try:
                    name = shape.id()
                except Exception:
                    pass
            if not isinstance(name, str) or len(name) == 0:
                name = f"shape_{mat_id}"

            material_rows.append(
                {
                    "id": mat_id,
                    "frequency": freq_val,
                    "name": name,
                    # These are the important ones used by Scene.to_sionna() and cost.py:
                    "thickness": get_scalar("bsdf.d", 0.1),
                    "permittivity": get_scalar("bsdf.eta_r", 4.0),
                    "conductivity": get_scalar("bsdf.sigma", 0.01),
                }
            )

        # Concatenate all extracted meshes
        if len(submeshes) == 0:
            mesh = trimesh.Trimesh(vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=int))
        else:
            mesh = reduce(trimesh.util.concatenate, submeshes)

        material_database = pl.DataFrame(material_rows)

        # --- extract TX/RX coordinates if present in scene ---
        if len(scene.receivers.values()) > 0:
            rx_coords = np.concatenate([rx.position for rx in scene.receivers.values()]).reshape(-1, 3)
        else:
            rx_coords = np.empty((0, 3), dtype=np.float32)

        if len(scene.transmitters.values()) > 0:
            tx_coords = np.concatenate([tx.position for tx in scene.transmitters.values()]).reshape(-1, 3)
        else:
            tx_coords = np.empty((0, 3), dtype=np.float32)

        antenna_database = AntennaDatabase.from_coords(tx_coords, rx_coords)

        return Scene(
            mesh=mesh,
            material_database=material_database,
            antenna_database=antenna_database,
            face2material=face_to_material,
        )
    
    
    def to_sionna_geometry(self, frequency: float, tol_hz: float = 1.0):
        """Build geometry + radio materials ONLY (no TX/RX nodes)."""
        if self.sionna_scene_geometry is not None:
            try:
                if float(self.sionna_scene_geometry.frequency) == float(frequency):
                    return self.sionna_scene_geometry
            except Exception:
                pass

        f_val, f_hz_i = _coerce_frequency_hz(frequency)

        material2face = defaultdict(list)
        for face_id, material_id in self.face2material.items():
            material2face[int(material_id)].append(int(face_id))

        df = self.material_database
        if "frequency_hz" not in df.columns:
            if "frequency" in df.columns:
                df = df.with_columns(pl.col("frequency").round(0).cast(pl.Int64).alias("frequency_hz"))
            else:
                df = df.with_columns(pl.lit(f_hz_i).cast(pl.Int64).alias("frequency_hz"))
            self.material_database = df

        def get_material_row(material_id: int) -> pl.DataFrame:
            df = self.material_database
            hit = df.filter((pl.col("id") == material_id) & (pl.col("frequency_hz") == f_hz_i)).head(1)
            if hit.height > 0:
                return hit

            same_id = df.filter(pl.col("id") == material_id)
            if same_id.height > 0:
                best = (
                    same_id.with_columns((pl.col("frequency_hz") - f_hz_i).abs().alias("_df"))
                        .sort("_df")
                        .drop("_df")
                        .head(1)
                )
                row = best.to_dicts()[0]
                row["frequency"] = float(f_val)
                row["frequency_hz"] = int(f_hz_i)
                self.material_database = pl.concat([df, pl.DataFrame([row])], how="vertical_relaxed")
                return pl.DataFrame([row])

            row = {
                "id": int(material_id),
                "frequency": float(f_val),
                "frequency_hz": int(f_hz_i),
                "name": str(material_id),
                "thickness": 0.1,
                "permittivity": 4.0,
                "conductivity": 0.01,
            }
            self.material_database = pl.concat([df, pl.DataFrame([row])], how="vertical_relaxed")
            return pl.DataFrame([row])

        meshes = []
        for material_id, face_list in material2face.items():
            material_data = get_material_row(material_id)

            sionna_material = sionna.rt.RadioMaterial(
                name=material_data[0, "name"],
                thickness=float(material_data[0, "thickness"]),
                relative_permittivity=float(material_data[0, "permittivity"]),
                conductivity=float(material_data[0, "conductivity"]),
            )

            sub = self.mesh.submesh([face_list], append=False)
            meshes.extend([trimesh2mitsuba(m, sionna_material) for m in sub])

        if len(meshes) == 0:
            raise ValueError("No meshes were produced (check face2material / mesh).")

        mi_scene = mi.load_dict(
            {"type": "scene", "integrator": {"type": "path"}}
            | {f"mesh_{i}": mesh for i, mesh in enumerate(meshes)}
        )
        assert isinstance(mi_scene, mi.Scene)

        si_scene = sionna.rt.Scene(mi_scene)
        si_scene.tx_array = sionna.rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
        si_scene.rx_array = si_scene.tx_array
        si_scene.frequency = float(f_val)

        self.sionna_scene_geometry = si_scene
        return si_scene




def mitsuba2trimesh(mesh: mi.Mesh) -> Trimesh:
    num_faces = mesh.face_count()
    faces = np.asarray(mesh.faces_buffer()).reshape(num_faces, 3)

    num_vertices = mesh.vertex_count()
    vertices = np.asarray(mesh.vertex_positions_buffer()).reshape(num_vertices, 3)
    vertex_normals = (
        np.asarray(mesh.vertex_normals_buffer()).reshape(num_vertices, 3)
        if mesh.has_vertex_normals()
        else None
    )

    return Trimesh(vertices, faces, vertex_normals=vertex_normals)


def trimesh2mitsuba(mesh: Trimesh, material: sionna.rt.RadioMaterialBase) -> mi.Mesh:
    num_vertices = mesh.vertices.shape[0]
    num_faces = mesh.faces.shape[0]

    has_vertex_normals = mesh.vertex_normals is not None

    # holder = sionna.rt.HolderMaterial(mi.Properties())
    # holder.radio_material = material

    props = mi.Properties()
    props["bsdf"] = material

    mi_mesh = mi.Mesh(
        "mesh", num_vertices, num_faces, props, has_vertex_normals=has_vertex_normals
    )
    mesh_params = mi.traverse(mi_mesh)
    for k in mesh_params.keys():
        if k == "faces":
            mesh_params[k] = np.asarray(mesh.faces, dtype=np.int32).flatten()
        elif k == "vertex_positions":
            mesh_params[k] = np.asarray(mesh.vertices, dtype=np.float32).flatten()
        elif has_vertex_normals and k == "vertex_normals":
            mesh_params[k] = np.asarray(mesh.vertex_normals, dtype=np.float32).flatten()

    mesh_params.update()
    mi_mesh.initialize()
    return mi_mesh
