"""Roadmap v3 line 302: render two differently-coloured objects with the CPU TinyRenderer,
save the image, and check which world axis maps to image-right."""
import pathlib, numpy as np, pybullet as p, pybullet_data
from PIL import Image

OUT = pathlib.Path.home() / "dronevla/results/phase0"
OUT.mkdir(parents=True, exist_ok=True)

p.connect(p.DIRECT)                       # headless: no OpenGL window, no GPU
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")

# RED cube at world y = +0.5 ; BLUE sphere at world y = -0.5
red = p.createMultiBody(
    0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[.15]*3),
    p.createVisualShape(p.GEOM_BOX, halfExtents=[.15]*3, rgbaColor=[1, 0, 0, 1]),
    [0, 0.5, 0.15])
blue = p.createMultiBody(
    0, p.createCollisionShape(p.GEOM_SPHERE, radius=.15),
    p.createVisualShape(p.GEOM_SPHERE, radius=.15, rgbaColor=[0, 0, 1, 1]),
    [0, -0.5, 0.15])

# camera on +x looking back at the origin, up = +z
view = p.computeViewMatrix(cameraEyePosition=[2.0, 0, 0.7],
                           cameraTargetPosition=[0, 0, 0.15],
                           cameraUpVector=[0, 0, 1])

for name, (w, h) in {"policy_128x96": (128, 96), "inspect_640x480": (640, 480)}.items():
    proj = p.computeProjectionMatrixFOV(60, w / h, 0.1, 100)
    _, _, rgb, _, seg = p.getCameraImage(w, h, view, proj, renderer=p.ER_TINY_RENDERER)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(h, w, 4)
    Image.fromarray(rgb[:, :, :3]).save(OUT / f"two_objects_{name}.png")

    # which half of the frame does each body occupy?
    s = np.asarray(seg).reshape(h, w)
    cols = {"red(y=+0.5)": red, "blue(y=-0.5)": blue}
    where = {k: ("LEFT" if np.asarray(np.where(s == v)[1]).mean() < w / 2 else "RIGHT")
             for k, v in cols.items() if (s == v).any()}
    print(f"{name:16s} {rgb.shape} {rgb.dtype}  ->  {where}")

print("saved to", OUT)
p.disconnect()
