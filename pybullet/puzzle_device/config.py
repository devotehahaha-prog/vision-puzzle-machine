"""Physical dimensions aligned with the E-题 control firmware.

Control frame (same as STM32 app_motion):
  origin = A4 bottom-right corner
  +X = left,  0 .. 210 mm
  +Y = up,    0 .. 297 mm
  +Z = up from the paper
  theta = 0 deg along +Y, 180 deg along -Y
"""

from dataclasses import dataclass

MM = 0.001


@dataclass(frozen=True)
class DeviceConfig:
    a4_x_mm: float = 210.0
    a4_y_mm: float = 297.0
    table_margin_mm: float = 40.0

    z_high_mm: float = 42.0
    z_low_mm: float = 5.0
    z_travel_mm: float = 50.0

    piece_thickness_mm: float = 3.0
    magnet_radius_mm: float = 8.0
    magnet_height_mm: float = 12.0
    pick_clearance_mm: float = 6.0

    xy_speed_m_s: float = 0.18
    z_speed_m_s: float = 0.08
    theta_speed_rad_s: float = 4.0

    magnet_attach_gap_mm: float = 12.0
    magnet_xy_capture_mm: float = 18.0


CFG = DeviceConfig()
