"""A4工作区内的最短正交路径规划。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：与普通版相同：A4 内最短正交路径，先 X 后 Y，无避障。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约36行  OrthogonalPathPlan：路径结果
#   - 约128行  plan_shortest_orthogonal_path：生成 L 形或直线
# =============================================================================
# 输入起点、终点和工作区像素宽高，输出OrthogonalPathPlan：points是折线顶点，planner是方法名，status是结果状态。
# 只沿X或Y轴运动时，两点最短距离是|终点x-起点x|+|终点y-起点y|，称为曼哈顿距离。
# 例如(10,20)到(30,50)，走(10,20)→(30,20)→(30,50)，总长20+30=50个像素。
# 此模块没有障碍物建模；前提是允许沿该L形路径搬运。
# =============================================================================
# 【分区】导入与类型别名
# 功能：引入数学库和路径结果数据结构。扑克版与普通版算法相同。
# 可修改：无现场参数。
# 看情况改：若路径点要带额外字段，下游 plan.json 也要一起改。
# 不要改：from __future__ import annotations；不要改成可变的三维点，机械只走 XY。
# =============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass

Point = tuple[float, float]


# =============================================================================
# 【分区】路径结果数据结构
# 功能：保存折线顶点、规划方法名和状态。
# 可修改：无。
# 看情况改：加长度字段可以，但 serial_transport 只读 points。
# 不要改：frozen=True、字段名 points/planner/status。
# =============================================================================
@dataclass(frozen=True)
class OrthogonalPathPlan:
    """一条仅由X/Y轴平行线段组成的搬运路径。"""

    points: list[Point]
    planner: str
    status: str


# =============================================================================
# 【分区】工作区尺寸换算
# 功能：A4 毫米宽高 × 像素/毫米 → 规划用像素工作区。
# 可修改：纸张尺寸改 config.json，不要改本函数。
# 看情况改：非 A4 必须同步改配置、标定和机械行程。
# 不要改：必须为正数；不要把 round 改成截断。
# =============================================================================
def workspace_size_px(paper_width_mm: float, paper_height_mm: float,
                      pixels_per_mm: float) -> tuple[int, int]:
    """按A4标定区域返回规划工作区像素尺寸。"""
    values = (paper_width_mm, paper_height_mm, pixels_per_mm)
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        raise ValueError("A4尺寸和像素比例必须为正数")
    return (
        int(round(paper_width_mm * pixels_per_mm)),
        int(round(paper_height_mm * pixels_per_mm)),
    )


# =============================================================================
# 【分区】正交几何工具
# 功能：判断单轴线段、压缩共线点、曼哈顿长度、点是否在工作区。
# 可修改：容差 1e-6 一般不动。
# 看情况改：压缩时保留折返端点，不要为“好看”删掉折返。
# 不要改：同向共线才删中间点；改错会把 L 形收成斜线。
# =============================================================================
def axis_aligned(start: Point, end: Point, tolerance: float = 1e-6) -> bool:
    """判断线段是否仅沿X轴或Y轴。"""
    return abs(start[0] - end[0]) <= tolerance or abs(start[1] - end[1]) <= tolerance


# 连续重复点可删；同一直线上且夹在首尾之间的中间点可删；折返端点不能删，否则会改变运动过程。
def compact_orthogonal_path(points: list[Point]) -> list[Point]:
    """删除重复点和同向共线中间点，同时保留折返端点。"""
    compact: list[Point] = []
    for point in points:
        normalized = (float(point[0]), float(point[1]))
        if compact and math.dist(compact[-1], normalized) <= 1e-6:
            continue
        compact.append(normalized)
    if len(compact) <= 2:
        return compact

    result = [compact[0]]
    for point in compact[1:]:
        result.append(point)
        while len(result) >= 3:
            first, middle, last = result[-3:]
            same_x = abs(first[0] - middle[0]) <= 1e-6 and abs(middle[0] - last[0]) <= 1e-6
            same_y = abs(first[1] - middle[1]) <= 1e-6 and abs(middle[1] - last[1]) <= 1e-6
            middle_between_y = min(first[1], last[1]) - 1e-6 <= middle[1] <= max(first[1], last[1]) + 1e-6
            middle_between_x = min(first[0], last[0]) - 1e-6 <= middle[0] <= max(first[0], last[0]) + 1e-6
            if not ((same_x and middle_between_y) or (same_y and middle_between_x)):
                break
            result.pop(-2)
    return result


def manhattan_length(points: list[Point]) -> float:
    """计算正交路径总长度。"""
    return sum(
        abs(end[0] - start[0]) + abs(end[1] - start[1])
        for start, end in zip(points, points[1:])
    )


def _point_in_workspace(point: Point, size_px: tuple[int, int],
                        tolerance: float = 1e-6) -> bool:
    x, y = point
    width, height = size_px
    return (
        math.isfinite(x) and math.isfinite(y)
        and -tolerance <= x < width
        and -tolerance <= y < height
    )


# =============================================================================
# 【分区】最短正交路径（先 X 后 Y）
# 功能：工作区内两点生成 L 形或直线；出界返回 out_of_workspace。
# 可修改：折点 (goal[0], start[1]) 为先 X 后 Y；现场先 Y 更安全时可改成 (start[0], goal[1])。
# 看情况改：无避障。片互相挡住应改搬运顺序，不要只改这里。
# 不要改：出界/已到达/共线三种提前返回；status 拼写。
# =============================================================================
def plan_shortest_orthogonal_path(start: Point, goal: Point,
                                  workspace_px: tuple[int, int]) -> OrthogonalPathPlan:
    """生成A4范围内的最短曼哈顿路径；固定先移动X轴，再移动Y轴。"""
    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    if not _point_in_workspace(start, workspace_px) or not _point_in_workspace(goal, workspace_px):
        return OrthogonalPathPlan([start], "shortest_xy", "out_of_workspace")
    if math.dist(start, goal) <= 1e-6:
        return OrthogonalPathPlan([start], "already_at_target", "ok")
    if axis_aligned(start, goal):
        return OrthogonalPathPlan([start, goal], "axis_direct", "ok")

    # 无障碍约束时，任意单折点L形路径都达到理论最短曼哈顿距离。
    path = compact_orthogonal_path([start, (goal[0], start[1]), goal])
    return OrthogonalPathPlan(path, "shortest_xy", "ok")
