"""Front onboard camera for the observation contract in roadmap v3 §4.3.

This re-implements the camera instead of calling `BaseAviary._getDroneImages`, because
that method does not fit the contract:

* it hardcodes `IMG_RES = [64, 48]`, while the contract fixes H=96, W=128, C=3;
* it never passes `renderer=`, so which rasteriser ran is not recorded anywhere;
* it defaults to `segmentation=True` (`ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX`) and
  `shadow=1`, both of which cost time and neither of which is a policy input;
* it calls `computeProjectionMatrixFOV(..., aspect=1.0)` while rendering a 4:3 image,
  which stretches the pixels.

Three ways of attaching the camera to the airframe (`mount`):

* ``"legacy"`` (default) -- exactly `_getDroneImages`'s view transform, kept so the
  Phase 0 profile stays reproducible. The eye sits L above the drone's centre along
  *world* +z, looks along body +x, and uses world +z as the up vector. Physically this is
  an odd hybrid: the image pitches with the airframe but never rolls with it.
* ``"rigid"`` -- a camera bolted to the airframe, the way a camera without a gimbal is.
  It pitches *and* rolls with the drone. `tilt_deg` is a fixed mounting angle.
* ``"gimbal"`` -- a stabilised camera, like the 3-axis gimbal on a DJI Mavic. Roll and
  pitch of the airframe are cancelled, the heading follows the drone's yaw, and the camera
  holds `tilt_deg` below (negative) or above the horizon. The stabilisation is ideal:
  real gimbals leave a small residual (DJI quotes +-0.003 deg in normal mode).

Deviations and settings are listed in `describe()` and written into reports, per §4.3
("camera intrinsics/extrinsics와 FOV를 manifest에 기록").
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pybullet as p

RENDERERS = {"tiny": p.ER_TINY_RENDERER}
MOUNTS = ("legacy", "rigid", "gimbal")


@dataclass
class FrontCamera:
    """Forward-looking camera attached to the drone."""

    width: int = 128
    height: int = 96
    fov_deg: float = 60.0          # vertical FOV, as PyBullet defines it
    near: float = 0.0397           # CF2X arm length L, matching BaseAviary
    far: float = 1000.0
    offset_world_z: float = 0.0397  # legacy only: BaseAviary adds L along *world* +z
    shadow: bool = False
    segmentation: bool = False
    renderer: str = "tiny"
    mount: str = "legacy"
    tilt_deg: float = 0.0          # rigid/gimbal: negative looks down
    # rigid/gimbal: lens position relative to the drone's centre, body frame (FLU):
    # (forward, left, up) in metres. Default: front edge of the CF2X body, slightly below.
    mount_offset_body: tuple = (0.035, 0.0, -0.015)
    # Kept for the record: BaseAviary passes world +z as the up vector, so the legacy
    # image does not roll with the airframe. Harmless near hover, degenerate near +-90 deg.
    up_vector: tuple = (0.0, 0.0, 1.0)

    _projection: list = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if self.renderer not in RENDERERS:
            raise ValueError(
                f"renderer must be one of {sorted(RENDERERS)}, got {self.renderer!r}. "
                "In DIRECT mode without the EGL plugin a hardware-OpenGL request cannot be "
                "verified as honoured, so it is not offered here."
            )
        if self.mount not in MOUNTS:
            raise ValueError(f"mount must be one of {MOUNTS}, got {self.mount!r}")
        if self.mount == "legacy" and self.tilt_deg != 0:
            raise ValueError("the legacy mount has no tilt; use mount='rigid' or 'gimbal'")
        if not -90.0 <= self.tilt_deg <= 70.0:
            # the Mavic 4 Pro gimbal's controllable tilt range
            raise ValueError(f"tilt_deg must be within [-90, 70], got {self.tilt_deg}")
        self._projection = p.computeProjectionMatrixFOV(
            fov=self.fov_deg,
            aspect=self.width / self.height,   # square pixels; BaseAviary uses 1.0
            nearVal=self.near,
            farVal=self.far,
        )

    @property
    def renderer_flag(self) -> int:
        return RENDERERS[self.renderer]

    def pose(self, pos, quat):
        """World-frame (eye, forward, up) unit vectors for the current drone pose."""
        rot = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        pos = np.asarray(pos, dtype=float)

        if self.mount == "legacy":
            eye = pos + np.array([0.0, 0.0, self.offset_world_z])
            return eye, rot.dot([1.0, 0.0, 0.0]), np.array(self.up_vector, dtype=float)

        eye = pos + rot.dot(np.asarray(self.mount_offset_body, dtype=float))
        th = math.radians(self.tilt_deg)
        if self.mount == "rigid":
            # tilt about the body's left (y) axis, then carried along by the airframe
            fwd = rot.dot([math.cos(th), 0.0, math.sin(th)])
            up = rot.dot([-math.sin(th), 0.0, math.cos(th)])
            return eye, fwd, up

        # gimbal: keep only the heading of the body x axis, then apply the fixed tilt
        yaw = math.atan2(rot[1, 0], rot[0, 0])
        fwd = np.array([math.cos(th) * math.cos(yaw), math.cos(th) * math.sin(yaw),
                        math.sin(th)])
        up = np.array([-math.sin(th) * math.cos(yaw), -math.sin(th) * math.sin(yaw),
                       math.cos(th)])
        return eye, fwd, up

    def view_matrix(self, pos, quat):
        eye, fwd, up = self.pose(pos, quat)
        return p.computeViewMatrix(
            cameraEyePosition=eye.tolist(),
            cameraTargetPosition=(eye + fwd).tolist(),
            cameraUpVector=up.tolist(),
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
        half_diag = math.hypot(self.width / 2, self.height / 2)
        dfov = 2 * math.degrees(math.atan(half_diag / f_px))
        if self.mount == "legacy":
            extrinsics = {
                "mount": "legacy",
                "eye": "drone position + [0, 0, offset_world_z] in world frame",
                "offset_world_z_m": self.offset_world_z,
                "look_direction": "body +x (forward)",
                "up_vector": list(self.up_vector),
                "note": ("up is world +z, copied from BaseAviary._getDroneImages, so the "
                         "image pitches with the airframe but does not roll with it"),
            }
        else:
            extrinsics = {
                "mount": self.mount,
                "mount_offset_body_flu_m": list(self.mount_offset_body),
                "tilt_deg": self.tilt_deg,
                "note": ("rigid: pitches and rolls with the airframe. gimbal: ideal "
                         "roll/pitch stabilisation, heading follows the drone's yaw"),
            }
        return {
            "width": self.width,
            "height": self.height,
            "channels": 3,
            "dtype": "uint8",
            "renderer": self.renderer,
            "fov_vertical_deg": self.fov_deg,
            "fov_horizontal_deg": round(hfov, 3),
            "fov_diagonal_deg": round(dfov, 3),
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
            "extrinsics": extrinsics,
            "shadow": self.shadow,
            "segmentation": self.segmentation,
            "deviations_from_baseaviary": [
                "resolution 128x96 instead of the hardcoded 64x48",
                "aspect = w/h instead of 1.0 (BaseAviary stretches pixels)",
                "renderer passed explicitly instead of left to the PyBullet default",
                "segmentation off by default (not a policy input)",
                "shadow off by default",
            ] + ([] if self.mount == "legacy" else
                 [f"mount '{self.mount}' instead of BaseAviary's view transform"]),
        }
