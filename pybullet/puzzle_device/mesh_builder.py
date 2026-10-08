"""Write ASCII-path OBJ/MTL meshes so PyBullet can texture the photo board."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from config import CFG, MM

ASSETS = Path(__file__).resolve().parent / "assets"


def _mesh_dir() -> Path:
    out = Path(tempfile.gettempdir()) / "puzzle_device_meshes"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _copy_texture(src_name: str, dest_name: str) -> Path:
    src = ASSETS / src_name
    dest = _mesh_dir() / dest_name
    dest.write_bytes(src.read_bytes())
    return dest


def _write_mtl(path: Path, material: str, texture: str) -> None:
    path.write_text(
        f"newmtl {material}\nKd 1 1 1\nillum 1\nmap_Kd {texture}\n",
        encoding="ascii",
    )


def write_board_mesh(texture_name: str = "a4_empty.jpg") -> Path:
    """Thin box covering the A4, UV-mapped from the warped photo."""
    mesh_dir = _mesh_dir()
    tex = _copy_texture(texture_name, "a4_empty.jpg")
    _write_mtl(mesh_dir / "a4_board.mtl", "board", tex.name)
    hx, hy = CFG.a4_x_mm * MM * 0.5, CFG.a4_y_mm * MM * 0.5
    z = 0.0012
    # Mesh is centred so the collision box and visual share the same origin.
    # UV: warped photo x-right = +Y, y-down = -X.
    obj = f"""mtllib a4_board.mtl
usemtl board
o a4_board
v {-hx} {-hy} {-z * 0.5}
v {hx} {-hy} {-z * 0.5}
v {hx} {hy} {-z * 0.5}
v {-hx} {hy} {-z * 0.5}
v {-hx} {-hy} {z * 0.5}
v {hx} {-hy} {z * 0.5}
v {hx} {hy} {z * 0.5}
v {-hx} {hy} {z * 0.5}
vt 0 0
vt 0 1
vt 1 1
vt 1 0
f 5/1 6/2 7/3 8/4
f 1/1 4/4 3/3 2/2
"""
    out = mesh_dir / "a4_board.obj"
    out.write_text(obj, encoding="ascii")
    return out


def _uv_for_vertex(x_mm: float, y_mm: float, crop_origin, tex_size) -> tuple[float, float]:
    img_x = y_mm * 4.0 - crop_origin[0]
    img_y = (CFG.a4_x_mm - x_mm) * 4.0 - crop_origin[1]
    u = float(np.clip(img_x / max(tex_size[0] - 1, 1), 0.0, 1.0))
    v = float(np.clip(1.0 - img_y / max(tex_size[1] - 1, 1), 0.0, 1.0))
    return u, v


def write_piece_mesh(piece: dict, thickness_m: float) -> Path:
    mesh_dir = _mesh_dir()
    name = piece["name"].lower()
    textured = bool(piece.get("texture"))
    if textured:
        tex = _copy_texture(piece["texture"], f"{name}.png")
        _write_mtl(mesh_dir / f"{name}.mtl", name, tex.name)
    verts = np.asarray(piece["vertices_mm"], dtype=np.float32)
    center = np.mean(verts, axis=0)
    local = verts - center
    z = thickness_m
    lines = ([f"mtllib {name}.mtl", f"usemtl {name}"] if textured else []) + [f"o {name}"]
    for x, y in local:
        lines.append(f"v {x * MM:.6f} {y * MM:.6f} {-z * 0.5:.6f}")
    for x, y in local:
        lines.append(f"v {x * MM:.6f} {y * MM:.6f} {z * 0.5:.6f}")
    uvs = []
    for x, y in verts:
        uvs.append(_uv_for_vertex(float(x), float(y), piece["crop_origin_px"], piece["texture_size_px"])
                   if textured else (0.0, 0.0))
    for u, v in uvs:
        lines.append(f"vt {u:.5f} {v:.5f}")
    n = len(local)
    top = " ".join(f"{i + 1 + n}/{i + 1}" for i in range(n))
    bottom = " ".join(f"{n - i}/{n - i}" for i in range(n))
    lines.append(f"f {top}")
    lines.append(f"f {bottom}")
    for i in range(n):
        j = (i + 1) % n
        a, b = i + 1, j + 1
        c, d = j + 1 + n, i + 1 + n
        lines.append(f"f {a}/{i + 1} {b}/{j + 1} {c}/{j + 1} {d}/{i + 1}")
    out = mesh_dir / f"{name}.obj"
    out.write_text("\n".join(lines) + "\n", encoding="ascii")
    return out


def load_layout() -> dict:
    path = ASSETS / "board_layout.json"
    if not path.exists():
        from board_from_photo import build_layout
        return build_layout()
    return json.loads(path.read_text(encoding="utf-8"))
