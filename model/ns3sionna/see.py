import argparse
import sys
from pathlib import Path

import sionna_vispy
from sionna_vispy import patch, get_canvas
from sionna.rt import load_scene


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("scene_xml")
    args = parser.parse_args()

    scene_path = Path(args.scene_xml).expanduser().resolve()
    if not scene_path.exists():
        parser.error(f"File not found: {scene_path}")

    scene = load_scene(str(scene_path))

    with patch():
        scene.preview()

    canvas = get_canvas(scene)
    canvas.show()
    canvas.app.run()  # blocks until you close the window

    return 0


if __name__ == "__main__":
    raise SystemExit(main())