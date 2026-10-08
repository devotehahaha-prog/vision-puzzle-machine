"""A4工作区内的最短正交路径规划。"""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：A4 工作区内两点之间的最短正交（曼哈顿）路径，固定先走 X 再走 Y，无避障。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约49行  OrthogonalPathPlan：折线顶点 + 方法名 + 状态
#   - 约72行  workspace_size_px：毫米纸面 × ppm → 像素工作区
#   - 约100行  axis_aligned：是否单轴线段
#   - 约110行  compact_orthogonal_path：删共线中间点，保留折返
#   - 约160行  manhattan_length：正交路径总长
#   - 约203行  plan_shortest_orthogonal_path：生成 L 形或直线；出界返回 out_of_workspace
# =============================================================================
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 输入起点、终点和工作区像素宽高，输出OrthogonalPathPlan：points是折线顶点，planner是方法名，status是结果状态。
# 只沿X或Y轴运动时，两点最短距离是|终点x-起点x|+|终点y-起点y|，称为曼哈顿距离。
# 例如(10,20)到(30,50)，走(10,20)→(30,20)→(30,50)，总长20+30=50个像素。
# 此模块没有障碍物建模；前提是允许沿该L形路径搬运。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入与类型别名
# 功能：引入数学库和路径结果数据结构。
# 可修改：无现场参数；本文件几乎不依赖硬件。
# 看情况改：若路径点要带额外字段，可扩展 Point 别名，但下游 plan.json 也要一起改。
# 不要改：from __future__ import annotations；不要改成可变的三维点，机械只走 XY。
# =============================================================================
# 从 __future__ 导入需要的类型或工具。
from __future__ import annotations

# 导入：math。
import math
# 从 dataclasses 导入需要的类型或工具。
from dataclasses import dataclass

# 计算并保存到 Point。
Point = tuple[float, float]


# =============================================================================
# 【分区】路径结果数据结构
# 功能：保存折线顶点、规划方法名和状态（ok / out_of_workspace 等）。
# 可修改：无。
# 看情况改：若要记录路径长度，可在数据结构里加字段，但 serial_transport 只读 points。
# 不要改：frozen=True、字段名 points/planner/status；主程序和串口按这些名字取值。
# =============================================================================
# 使用dataclass装饰器，按下方字段自动生成__init__、__repr__和比较方法；frozen=True限制字段重新赋值，但不冻结字段内部的列表/数组。
@dataclass(frozen=True)
# 【类：OrthogonalPathPlan】一条仅由X/Y轴平行线段组成的搬运路径。
class OrthogonalPathPlan:
    """一条仅由X/Y轴平行线段组成的搬运路径。"""

    # 声明字段：points（参与当前计算的一组坐标点）。
    points: list[Point]
    # 声明字段：planner（路径规划方法标记）。
    planner: str
    # 声明字段：status（处理结果的状态标记）。
    status: str


# =============================================================================
# 【分区】工作区尺寸换算
# 功能：把 A4 毫米宽高 × 像素/毫米，得到规划用的像素工作区。
# 可修改：无。纸张尺寸应改 config.json 的 paper_width_mm / paper_height_mm / pixels_per_mm。
# 看情况改：若改用非 A4 纸，必须同步改配置、标定和机械行程。
# 不要改：必须为正数才换算；不要把 round 改成截断，否则边界差 1 像素。
# =============================================================================
# 【函数：workspace_size_px】按A4标定区域返回规划工作区像素尺寸。
# 参数 paper_width_mm（纸面·宽度·毫米）：浮点数。
# 参数 paper_height_mm（纸面·高度·毫米）：浮点数。
# 参数 pixels_per_mm（每毫米对应的像素数）：浮点数。
# 返回类型：元组（依次为整数、整数）；箭头->是类型提示，不会替你转换实际返回值。
def workspace_size_px(paper_width_mm: float, paper_height_mm: float,
                      pixels_per_mm: float) -> tuple[int, int]:
    """按A4标定区域返回规划工作区像素尺寸。"""
    # 计算并保存到 values（当前数值集合）。
    values = (paper_width_mm, paper_height_mm, pixels_per_mm)
    # 判断条件；满足时执行下面代码：not all((math.isfinite(value) and value > 0.0 for value in values))。
    if not all(math.isfinite(value) and value > 0.0 for value in values):
        # 抛出异常，通知上层处理：ValueError('A4尺寸和像素比例必须为正数')。
        raise ValueError("A4尺寸和像素比例必须为正数")
    # 返回结果：创建数据容器。
    return (
        int(round(paper_width_mm * pixels_per_mm)),
        int(round(paper_height_mm * pixels_per_mm)),
    )


# =============================================================================
# 【分区】正交几何工具
# 功能：判断是否单轴线段、压缩共线点、计算曼哈顿长度、检查点是否在工作区内。
# 可修改：容差 1e-6 一般不用动；只有浮点抖动导致误删拐点时才略放大。
# 看情况改：compact 保留折返端点是为了真实往返路径，不要为“好看”删掉折返。
# 不要改：压缩逻辑（同向共线才删中间点）；改错会把 L 形路径收成斜线，电机无法斜走。
# =============================================================================
# 【函数：axis_aligned】判断线段是否仅沿X轴或Y轴。
# 参数 start（起始值或起点）：Point。
# 参数 end（终止值或终点）：Point。
# 参数 tolerance（用于浮点比较或几何判断的容差）：浮点数；省略时使用1e-06。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def axis_aligned(start: Point, end: Point, tolerance: float = 1e-6) -> bool:
    """判断线段是否仅沿X轴或Y轴。"""
    # 返回结果：组合多个条件：abs(start[0] - end[0]) <= tolerance or abs(start[1] - end[1]) <= tolerance。
    return abs(start[0] - end[0]) <= tolerance or abs(start[1] - end[1]) <= tolerance


# 【函数：compact_orthogonal_path】删除重复点和同向共线中间点，同时保留折返端点。
# 连续重复点可删；同一直线上且夹在首尾之间的中间点可删；折返端点不能删，否则会改变运动过程。
# 参数 points（参与当前计算的一组坐标点）：Point列表/序列。
# 返回类型：Point列表/序列；箭头->是类型提示，不会替你转换实际返回值。
def compact_orthogonal_path(points: list[Point]) -> list[Point]:
    """删除重复点和同向共线中间点，同时保留折返端点。"""
    # 计算并保存到 compact。
    compact: list[Point] = []
    # 遍历数据，逐项处理：points。
    for point in points:
        # 计算并保存到 normalized（经过本函数规范化处理的数据）。
        normalized = (float(point[0]), float(point[1]))
        # 判断条件；满足时执行下面代码：compact and math.dist(compact[-1], normalized) <= 1e-06。
        if compact and math.dist(compact[-1], normalized) <= 1e-6:
            # 跳过本轮，进入下一轮循环。
            continue
        # 调用函数：compact.append。
        compact.append(normalized)
    # 判断条件；满足时执行下面代码：len(compact) <= 2。
    if len(compact) <= 2:
        # 返回结果：compact。
        return compact

    # 计算并保存到 result（当前步骤得到的结果）。
    result = [compact[0]]
    # 遍历数据，逐项处理：compact[1:]。
    for point in compact[1:]:
        # 调用函数：result.append。
        result.append(point)
        # 只要条件成立就重复执行：len(result) >= 3。
        while len(result) >= 3:
            # 计算并保存到 first（第一项数据）、middle（中间项数据）、last（最后一项数据）。
            first, middle, last = result[-3:]
            # 计算并保存到 same_x（same·X轴）。
            same_x = abs(first[0] - middle[0]) <= 1e-6 and abs(middle[0] - last[0]) <= 1e-6
            # 计算并保存到 same_y（same·Y轴）。
            same_y = abs(first[1] - middle[1]) <= 1e-6 and abs(middle[1] - last[1]) <= 1e-6
            # 计算并保存到 middle_between_y（中间·between·Y轴）。
            middle_between_y = min(first[1], last[1]) - 1e-6 <= middle[1] <= max(first[1], last[1]) + 1e-6
            # 计算并保存到 middle_between_x（中间·between·X轴）。
            middle_between_x = min(first[0], last[0]) - 1e-6 <= middle[0] <= max(first[0], last[0]) + 1e-6
            # 判断条件；满足时执行下面代码：not (same_x and middle_between_y or (same_y and middle_between_x))。
            if not ((same_x and middle_between_y) or (same_y and middle_between_x)):
                # 立即结束当前循环。
                break
            # 调用函数：result.pop。
            result.pop(-2)
    # 返回结果：result（当前步骤得到的结果）。
    return result


# 【函数：manhattan_length】计算正交路径总长度。
# 参数 points（参与当前计算的一组坐标点）：Point列表/序列。
# 返回类型：浮点数；箭头->是类型提示，不会替你转换实际返回值。
def manhattan_length(points: list[Point]) -> float:
    """计算正交路径总长度。"""
    # 返回结果：调用 sum。
    return sum(
        abs(end[0] - start[0]) + abs(end[1] - start[1])
        # 这是上一条推导式的遍历部分；按这里的变量和范围逐项生成前面指定的结果。
        for start, end in zip(points, points[1:])
    )


# 【函数：_point_in_workspace】检查坐标是否有限并落在规划工作区允许范围内。
# 参数 point（当前坐标点）：Point。
# 参数 size_px（尺寸·像素）：元组（依次为整数、整数）。
# 参数 tolerance（用于浮点比较或几何判断的容差）：浮点数；省略时使用1e-06。
# 返回类型：布尔值True/False；箭头->是类型提示，不会替你转换实际返回值。
def _point_in_workspace(point: Point, size_px: tuple[int, int],
                        tolerance: float = 1e-6) -> bool:
    # 计算并保存到 x（当前横坐标或横向数据）、y（当前纵坐标或纵向数据）。
    x, y = point
    # 计算并保存到 width（当前区域宽度）、height（当前区域高度）。
    width, height = size_px
    # 返回结果：组合多个条件：math.isfinite(x) and math.isfinite(y) and (-tolerance <= x < width) and (-…。
    return (
        math.isfinite(x) and math.isfinite(y)
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and -tolerance <= x < width
        # 续接上一行的条件表达式；当前行仍属于同一次判断，所有括号闭合后才决定分支。
        and -tolerance <= y < height
    )


# =============================================================================
# 【分区】最短正交路径（先 X 后 Y）
# 功能：工作区内两点生成 L 形或直线路径；出界返回 status=out_of_workspace。
# 可修改：折点 (goal[0], start[1]) 表示先走 X 再走 Y。若现场先 Y 更不易碰片，可改成 (start[0], goal[1])。
# 看情况改：本模块无避障。碎片互相挡住时不要只改这里，应改搬运顺序或加障碍规划。
# 不要改：出界/已到达/共线三种提前返回；status 字符串被上层判断，不要改拼写。
# =============================================================================
# 【函数：plan_shortest_orthogonal_path】生成A4范围内的最短曼哈顿路径；固定先移动X轴，再移动Y轴。
# 参数 start（起始值或起点）：Point。
# 参数 goal（路径目标点）：Point。
# 参数 workspace_px（工作区·像素）：元组（依次为整数、整数）。
# 返回类型：OrthogonalPathPlan；箭头->是类型提示，不会替你转换实际返回值。
def plan_shortest_orthogonal_path(start: Point, goal: Point,
                                  workspace_px: tuple[int, int]) -> OrthogonalPathPlan:
    """生成A4范围内的最短曼哈顿路径；固定先移动X轴，再移动Y轴。"""
    # 计算并保存到 start（起始值或起点）。
    start = (float(start[0]), float(start[1]))
    # 计算并保存到 goal（路径目标点）。
    goal = (float(goal[0]), float(goal[1]))
    # 判断条件；满足时执行下面代码：not _point_in_workspace(start, workspace_px) or not _point_in_workspace(goal, w…。
    if not _point_in_workspace(start, workspace_px) or not _point_in_workspace(goal, workspace_px):
        # 返回结果：调用 OrthogonalPathPlan。
        return OrthogonalPathPlan([start], "shortest_xy", "out_of_workspace")
    # 判断条件；满足时执行下面代码：math.dist(start, goal) <= 1e-06。
    if math.dist(start, goal) <= 1e-6:
        # 返回结果：调用 OrthogonalPathPlan。
        return OrthogonalPathPlan([start], "already_at_target", "ok")
    # 判断条件；满足时执行下面代码：axis_aligned(start, goal)。
    if axis_aligned(start, goal):
        # 返回结果：调用 OrthogonalPathPlan。
        return OrthogonalPathPlan([start, goal], "axis_direct", "ok")

    # 无障碍约束时，任意单折点L形路径都达到理论最短曼哈顿距离。
    # 计算并保存到 path（当前文件路径或几何路径）。
    path = compact_orthogonal_path([start, (goal[0], start[1]), goal])
    # 返回结果：调用 OrthogonalPathPlan。
    return OrthogonalPathPlan(path, "shortest_xy", "ok")
