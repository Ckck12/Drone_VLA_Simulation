"""Front onboard camera for the observation contract in roadmap v3 §4.3.

This re-implements the camera instead of calling `BaseAviary._getDroneImages`, because
that method does not fit the contract:

* it hardcodes `IMG_RES = [64, 48]`, while the contract fixes H=96, W=128, C=3;
* it never passes `renderer=`, so which rasteriser ran is not recorded anywhere;
* it defaults to `segmentation=True` (`ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX`) and
  `shadow=1`, both of which cost time and neither of which is a policy input;
* it calls `computeProjectionMatrixFOV(..., aspect=1.0)` while rendering a 4:3 image,
  which stretches the pixels.

The view transform is kept identical to `_getDroneImages` on purpose, so frames here are
comparable with anything the library's own vision path produces. Deviations are listed in
`describe()` and written into the report, per §4.3 ("camera intrinsics/extrinsics와 FOV를
manifest에 기록").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

RENDERERS = {"tiny": p.ER_TINY_RENDERER}


@dataclass
class FrontCamera:
    """Forward-looking camera rigidly attached above the drone's centre."""

    width: int = 128
    height: int = 96
    fov_deg: float = 60.0          # vertical FOV, as PyBullet defines it
    near: float = 0.0397           # CF2X arm length L, matching BaseAviary
    far: float = 1000.0
    offset_world_z: float = 0.0397  # BaseAviary adds L along *world* +z, not body +z
    shadow: bool = False
    segmentation: bool = False
    renderer: str = "tiny"
    # Kept for the record: BaseAviary passes world +z as the up vector, so the image does
    # not roll with the airframe. Harmless near hover, degenerate near +-90 deg pitch.
    up_vector: tuple = (0.0, 0.0, 1.0)

    _projection: list = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.renderer not in RENDERERS:
            raise ValueError(
                f"renderer must be one of {sorted(RENDERERS)}, got {self.renderer!r}. "
                "In DIRECT mode without the EGL plugin a hardware-OpenGL request cannot be "
                "verified as honoured, so it is not offered here."
            )
        self._projection = p.computeProjectionMatrixFOV(
            fov=self.fov_deg,
            aspect=self.width / self.height,   # square pixels; BaseAviary uses 1.0
            nearVal=self.near,
            farVal=self.far,
        )

    @property
    def renderer_flag(self) -> int:
        return RENDERERS[self.renderer]

    def view_matrix(self, pos, quat):
        """Eye above `pos`, looking along the body +x axis."""
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        eye = np.asarray(pos, dtype=float) + np.array([0.0, 0.0, self.offset_world_z])
        target = eye + rot.dot(np.array([1.0, 0.0, 0.0]))
        return p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=target.tolist(),
            cameraUpVector=list(self.up_vector),
        )

    def render(self, pos, quat, client: int = 0) -> np.ndarray:
        """Return one frame as `(height, width, 3)` uint8, alpha dropped."""
        flags = (
            p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX
            if self.segmentation
            else p.ER_NO_SEGMENTATION_MASK
        )
        w, h, rgba, _, _ = p.getCameraImage(
            width=self.width,
            height=self.height,
            viewMatrix=self.view_matrix(pos, quat),
            projectionMatrix=self._projection,
            shadow=1 if self.shadow else 0,
            flags=flags,
            renderer=self.renderer_flag,
            physicsClientId=client,
        )
        # pybullet.isNumpyEnabled() == 1 in this build, so rgba is already an ndarray;
        # np.asarray keeps the non-numpy build working without copying when it is one.
        return np.ascontiguousarray(
            np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        )

    def describe(self) -> dict:
        """Intrinsics/extrinsics for the manifest."""
        f_px = (self.height / 2) / math.tan(math.radians(self.fov_deg) / 2)
        hfov = 2 * math.degrees(math.atan((self.width / 2) / f_px))
        return {
            "width": self.width,
            "height": self.height,
            "channels": 3,
            "dtype": "uint8",
            "renderer": self.renderer,
            "fov_vertical_deg": self.fov_deg,
            "fov_horizontal_deg": round(hfov, 3),
            "aspect": self.width / self.height,
            "near_m": self.near,
            "far_m": self.far,
            "intrinsics_px": {
                "fx": round(f_px, 4),
                "fy": round(f_px, 4),
                "cx": self.width / 2,
                "cy": self.height / 2,
                "note": "square pixels; derived from the vertical FOV and aspect=w/h",
            },
            "extrinsics": {
                "eye": "drone position + [0, 0, offset_world_z] in world frame",
                "offset_world_z_m": self.offset_world_z,
                "look_direction": "body +x (forward)",
                "up_vector": list(self.up_vector),
                "note": (
                    "up is world +z, copied from BaseAviary._getDroneImages, so the image "
                    "does not roll with the airframe"
                ),
            },
            "shadow": self.shadow,
            "segmentation": self.segmentation,
            "deviations_from_baseaviary": [
                "resolution 128x96 instead of the hardcoded 64x48",
                "aspect = w/h instead of 1.0 (BaseAviary stretches pixels)",
                "renderer passed explicitly instead of left to the PyBullet default",
                "segmentation off by default (not a policy input)",
                "shadow off by default",
            ],
        }
