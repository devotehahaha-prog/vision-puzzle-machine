"""Fixed orthographic sensor; only RGB pixels are exposed to vision."""

import numpy as np
import pybullet as p

from config import CFG, MM


class VirtualCamera:
    def __init__(self, client_id: int, pixels_per_mm: float = 4.0):
        self.client_id = client_id
        self.ppm = pixels_per_mm
        self.margin_mm = 20.0
        self.width = round((CFG.a4_x_mm + 2 * self.margin_mm) * self.ppm)
        self.height = round((CFG.a4_y_mm + 2 * self.margin_mm) * self.ppm)
        half_x = self.width / self.ppm * MM / 2
        half_y = self.height / self.ppm * MM / 2
        near, far = 0.01, 0.5
        projection = np.diag([1 / half_x, 1 / half_y, -2 / (far - near), 1.0])
        projection[2, 3] = -(far + near) / (far - near)
        self.projection = projection.flatten(order="F").tolist()
        self.view = p.computeViewMatrix(
            [CFG.a4_x_mm * MM / 2, CFG.a4_y_mm * MM / 2, 0.24],
            [CFG.a4_x_mm * MM / 2, CFG.a4_y_mm * MM / 2, 0], [0, 1, 0])
        margin_px = self.margin_mm * self.ppm
        self.matrix = np.array([[1., 0., -margin_px], [0., 1., -margin_px], [0., 0., 1.]])

    def capture(self) -> np.ndarray:
        rgba = p.getCameraImage(
            self.width, self.height, self.view, self.projection,
            renderer=p.ER_TINY_RENDERER, flags=p.ER_NO_SEGMENTATION_MASK,
            shadow=0, lightDirection=[0, 0, 1], lightAmbientCoeff=0.8,
            lightDiffuseCoeff=0.2, lightSpecularCoeff=0,
            physicsClientId=self.client_id)[2]
        rgba = np.asarray(rgba, dtype=np.uint8).reshape(self.height, self.width, 4)
        # Mirror the optical X axis to match the device's left/up coordinate convention.
        return np.ascontiguousarray(rgba[:, ::-1, :3][:, :, ::-1])
