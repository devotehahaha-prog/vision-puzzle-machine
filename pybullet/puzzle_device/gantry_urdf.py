"""Serial XYZ + theta gantry URDF, visually resembling a writing-machine frame."""

import tempfile
from pathlib import Path

from config import CFG, MM


def urdf_text() -> str:
    a4_x = CFG.a4_x_mm * MM
    a4_y = CFG.a4_y_mm * MM
    table_x = a4_x + 2 * CFG.table_margin_mm * MM
    table_y = a4_y + 2 * CFG.table_margin_mm * MM
    table_h = 0.024
    margin = CFG.table_margin_mm * MM
    z_up = CFG.z_travel_mm * MM
    mag_r = CFG.magnet_radius_mm * MM
    mag_h = CFG.magnet_height_mm * MM

    # Base origin is the A4 control origin. The table extends into +X/+Y
    # and a little into the negative margins so the paper sits on top.
    table_cx = a4_x * 0.5
    table_cy = a4_y * 0.5
    return f"""<?xml version="1.0"?>
<robot name="puzzle_gantry">
  <material name="frame"><color rgba="0.18 0.20 0.24 1"/></material>
  <material name="rail"><color rgba="0.55 0.58 0.62 1"/></material>
  <material name="carriage"><color rgba="0.22 0.45 0.78 1"/></material>
  <material name="servo"><color rgba="0.15 0.15 0.16 1"/></material>
  <material name="magnet"><color rgba="0.82 0.16 0.14 1"/></material>
  <material name="marker"><color rgba="0.95 0.82 0.12 1"/></material>

  <link name="base">
    <inertial>
      <mass value="8"/>
      <origin xyz="{table_cx} {table_cy} {-table_h * 0.5}"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
    <visual>
      <origin xyz="{table_cx} {table_cy} {-table_h * 0.5}"/>
      <geometry><box size="{table_x} {table_y} {table_h}"/></geometry>
      <material name="frame"/>
    </visual>
    <collision>
      <origin xyz="{table_cx} {table_cy} {-table_h * 0.5}"/>
      <geometry><box size="{table_x} {table_y} {table_h}"/></geometry>
    </collision>
    <visual>
      <origin xyz="{-margin} {table_cy} 0.06"/>
      <geometry><box size="0.018  {table_y} 0.12"/></geometry>
      <material name="frame"/>
    </visual>
    <visual>
      <origin xyz="{a4_x + margin} {table_cy} 0.06"/>
      <geometry><box size="0.018 {table_y} 0.12"/></geometry>
      <material name="frame"/>
    </visual>
    <visual>
      <origin xyz="{table_cx} {-margin} 0.11"/>
      <geometry><box size="{table_x} 0.016 0.016"/></geometry>
      <material name="rail"/>
    </visual>
    <visual>
      <origin xyz="{table_cx} {a4_y + margin} 0.11"/>
      <geometry><box size="{table_x} 0.016 0.016"/></geometry>
      <material name="rail"/>
    </visual>
  </link>

  <link name="x_beam">
    <inertial>
      <mass value="0.6"/>
      <origin xyz="0 {table_cy} 0"/>
      <inertia ixx="0.02" ixy="0" ixz="0" iyy="0.02" iyz="0" izz="0.02"/>
    </inertial>
    <visual>
      <origin xyz="0 {table_cy} 0"/>
      <geometry><box size="0.022 {table_y + 0.04} 0.022"/></geometry>
      <material name="rail"/>
    </visual>
  </link>
  <joint name="x_joint" type="prismatic">
    <parent link="base"/>
    <child link="x_beam"/>
    <origin xyz="0 0 0.11"/>
    <axis xyz="1 0 0"/>
    <limit lower="0" upper="{a4_x}" effort="80" velocity="0.4"/>
  </joint>

  <link name="y_carriage">
    <inertial>
      <mass value="0.25"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.004" ixy="0" ixz="0" iyy="0.004" iyz="0" izz="0.004"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.05 0.04 0.03"/></geometry>
      <material name="carriage"/>
    </visual>
  </link>
  <joint name="y_joint" type="prismatic">
    <parent link="x_beam"/>
    <child link="y_carriage"/>
    <origin xyz="0 0 0.0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="{a4_y}" effort="80" velocity="0.4"/>
  </joint>

  <link name="z_slider">
    <inertial>
      <mass value="0.12"/>
      <origin xyz="0 0 {-0.02}"/>
      <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.002" iyz="0" izz="0.002"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {-0.01}"/>
      <geometry><box size="0.016 0.016 0.07"/></geometry>
      <material name="carriage"/>
    </visual>
  </link>
  <joint name="z_joint" type="prismatic">
    <parent link="y_carriage"/>
    <child link="z_slider"/>
    <!-- y_carriage is at z=0.11. Offsets below the carriage are chosen so
         magnet-tip height above the paper equals the z_joint value. -->
    <origin xyz="0 0 -0.043"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="{z_up}" effort="40" velocity="0.2"/>
  </joint>

  <link name="theta_servo">
    <inertial>
      <mass value="0.08"/>
      <origin xyz="0 0 0"/>
      <inertia ixx="0.0004" ixy="0" ixz="0" iyy="0.0004" iyz="0" izz="0.0004"/>
    </inertial>
    <visual>
      <origin xyz="0 0 0"/>
      <geometry><box size="0.028 0.022 0.018"/></geometry>
      <material name="servo"/>
    </visual>
    <visual>
      <origin xyz="0 0.014 0"/>
      <geometry><box size="0.006 0.018 0.006"/></geometry>
      <material name="marker"/>
    </visual>
  </link>
  <joint name="theta_joint" type="revolute">
    <parent link="z_slider"/>
    <child link="theta_servo"/>
    <origin xyz="0 0 {-0.045}"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="{3.141592653589793}" effort="5" velocity="8"/>
  </joint>

  <link name="magnet">
    <inertial>
      <mass value="0.04"/>
      <origin xyz="0 0 {-mag_h * 0.5}"/>
      <inertia ixx="0.0002" ixy="0" ixz="0" iyy="0.0002" iyz="0" izz="0.0002"/>
    </inertial>
    <visual>
      <origin xyz="0 0 {-mag_h * 0.5}"/>
      <geometry><cylinder radius="{mag_r}" length="{mag_h}"/></geometry>
      <material name="magnet"/>
    </visual>
    <collision>
      <origin xyz="0 0 {-mag_h * 0.5}"/>
      <geometry><cylinder radius="{mag_r}" length="{mag_h}"/></geometry>
    </collision>
  </link>
  <joint name="magnet_fixed" type="fixed">
    <parent link="theta_servo"/>
    <child link="magnet"/>
    <origin xyz="0 0 {-0.01}"/>
  </joint>
</robot>
"""


def write_urdf(path: Path | None = None) -> Path:
    # PyBullet cannot load URDF from paths that contain non-ASCII characters,
    # so the default location is the system temp directory.
    out = path or (Path(tempfile.gettempdir()) / "puzzle_gantry.urdf")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(urdf_text(), encoding="utf-8")
    project_copy = Path(__file__).resolve().parent / "urdf" / "puzzle_gantry.urdf"
    project_copy.parent.mkdir(parents=True, exist_ok=True)
    project_copy.write_text(urdf_text(), encoding="utf-8")
    return out
