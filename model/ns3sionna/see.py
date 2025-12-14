import sionna_vispy
from sionna_vispy import patch
from sionna.rt import load_scene

scene = load_scene("path/to/scene.xml")

with patch():
    scene.preview()