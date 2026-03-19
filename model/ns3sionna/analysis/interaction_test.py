import json
import numpy as np
from pathlib import Path
import sionna
from sionna.rt import load_scene, Transmitter, Receiver, PlanarArray, PathSolver

scene_dir = Path("worldbuilding/seed1697")
scene = load_scene(str(scene_dir / "scene.xml"))

# use the suite placements
pl = json.loads((scene_dir / "placements.json").read_text())
tx = np.array(pl["tx_xyz"], dtype=np.float32)
rx = np.array(pl["sta_xyz"][5], dtype=np.float32)   # choose a blocked STA

scene.tx_array = PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
scene.rx_array = scene.tx_array
scene.frequency = float(pl["frequency_hz"])

scene.add(Transmitter("tx", tx))
scene.add(Receiver("rx", rx))

solver = PathSolver()

for d in range(0, 21):
    paths = solver(
        scene=scene,
        max_depth=d,
        samples_per_src=10**6,
        los=True,
        specular_reflection=False,
        diffuse_reflection=False,
        refraction=True,
        synthetic_array=False,
        diffraction=False,
        edge_diffraction=False,
        diffraction_lit_region=False,
    )
    a, tau = paths.cir(sampling_frequency=1e9, normalize_delays=False, out_type="numpy")
    valid = tau[np.isfinite(tau) & (tau >= 0)]
    print(d, "PATH" if valid.size > 0 else "NO_PATH", valid.min() if valid.size else None)