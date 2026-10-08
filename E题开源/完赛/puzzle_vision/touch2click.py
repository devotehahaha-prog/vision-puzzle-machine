#!/usr/bin/env python3
"""触摸屏触摸转鼠标点击守护: 触摸时移动光标到触摸位置, 抬起时模拟左键点击."""
# =============================================================================
# 【文件导读】先看这里，再按「约xxxx行」跳到对应函数
# 本文件功能：Linux/X11 触摸转单击：读 /dev/input 触摸事件，按下移光标，抬起模拟左键。导入即进入死循环。
# 主要代码位置（约等于当前行号，后面再加注释时可能差几行）：
#   - 约30行  TOUCH_DEV：触摸设备路径，常改 event 编号
#   - 约70行  move_mouse：XTest 移动光标
#   - 约77行  click：左键按下再松开
# =============================================================================
# 【中文阅读导引】以下新增#注释解释代码用途；原有字符串、计算、默认值和执行顺序保持不变。
# 这是Linux/X11触摸辅助脚本，不是拼图求解器：读取触摸事件，把按下位置映射到屏幕，松开时模拟一次左键点击。
# 它直接访问/dev/input/event0和X11显示:0；导入文件也会执行设备初始化与while循环，因为这里没有main入口保护。
# 事件记录按本程序假定的24字节解析；llHHi使用本机C类型尺寸与对齐，不是所有系统架构都固定24字节。
# 类型3是EV_ABS；代码53/54是多点触摸X/Y位置；类型1代码330是BTN_TOUCH按下/抬起标记。
# 坐标按触摸量程与屏幕尺寸做线性比例换算。本程序在按下事件时移动一次光标，没有实现持续拖动。
# 语法约定：缩进决定代码归属；=赋值，==比较；列表和数组下标从0开始；None表示没有值；冒号后的类型主要用于阅读和检查。
# 跨行表达式属于同一条语句，括号结束前不会另起一条指令；字典条目和命名实参也在相邻注释中解释。
# =============================================================================
# 【分区】导入与触摸/屏幕参数
# 功能：打开 Linux 触摸设备和 X11 :0，把触摸坐标线性映射到屏幕像素。
# 可修改：TOUCH_DEV（常见 /dev/input/event0~eventN）、SCREEN_W/H（屏幕分辨率）、TOUCH_W/H（触摸量程）。
# 看情况改：本文件导入即进入死循环，没有 if __name__ 保护。调试请直接运行脚本，不要 from 导入。
# 不要改：事件解析 llHHi、类型 3 代码 53/54、类型 1 代码 330；改错会点到随机位置。
# =============================================================================
# 导入：os、struct、select、time、ctypes、ctypes.util。
import os, struct, select, time, ctypes, ctypes.util

# 计算并保存到 TOUCH_DEV（Linux触摸输入设备文件）。
TOUCH_DEV = "/dev/input/event0"
# 计算并保存到 SCREEN_W（屏幕像素宽度）、SCREEN_H（屏幕像素高度）。
SCREEN_W, SCREEN_H = 1920, 1080
# 计算并保存到 TOUCH_W（触摸设备横坐标量程）、TOUCH_H（触摸设备纵坐标量程）。
TOUCH_W, TOUCH_H = 1024, 600

# 计算并保存到 xlib（X11动态库）。
xlib = ctypes.CDLL(ctypes.util.find_library('X11'))
# 计算并保存到 xtst（XTest扩展动态库）。
xtst = ctypes.CDLL(ctypes.util.find_library('Xtst'))
# 计算并保存到 xlib.XOpenDisplay.argtypes（C函数的参数类型列表）。
xlib.XOpenDisplay.argtypes = [ctypes.c_char_p]
# 计算并保存到 xlib.XOpenDisplay.restype（C函数的返回类型）。
xlib.XOpenDisplay.restype = ctypes.c_void_p
# 计算并保存到 dpy（X11显示连接的指针）。
dpy = xlib.XOpenDisplay(b":0")
# 判断条件；满足时执行下面代码：not dpy。
if not dpy:
    # 调用函数：print。
    print("无法打开X显示: :0")
    # 抛出异常，通知上层处理：SystemExit(1)。
    raise SystemExit(1)

# 计算并保存到 xtst.XTestFakeMotionEvent.argtypes（C函数的参数类型列表）。
xtst.XTestFakeMotionEvent.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_ulong]
# 计算并保存到 xtst.XTestFakeButtonEvent.argtypes（C函数的参数类型列表）。
xtst.XTestFakeButtonEvent.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_int, ctypes.c_ulong]
# 计算并保存到 xlib.XFlush.argtypes（C函数的参数类型列表）。
xlib.XFlush.argtypes = [ctypes.c_void_p]

# =============================================================================
# 【分区】XTest 鼠标移动与单击
# 功能：移动光标；抬起时按下再松开左键，间隔 0.03s。
# 可修改：click 间隔 0.03；太短可能点不上，太长手感钝。
# 看情况改：当前按下只移一次光标，不支持拖动。若要拖动需在 ABS 事件里持续 move_mouse。
# 不要改：XTestFakeButtonEvent 按钮号 1（左键）；不要在这里发右键/中键。
# =============================================================================
# 【函数：move_mouse】调用XTest将鼠标光标移动到屏幕上的指定坐标。
# 参数 x（当前横坐标或横向数据）：未限定类型，由调用方传入。
# 参数 y（当前纵坐标或纵向数据）：未限定类型，由调用方传入。
def move_mouse(x, y):
    # 调用函数：xtst.XTestFakeMotionEvent。
    xtst.XTestFakeMotionEvent(ctypes.c_void_p(dpy), -1, int(x), int(y), 0)
    # 调用函数：xlib.XFlush。
    xlib.XFlush(ctypes.c_void_p(dpy))

# 【函数：click】模拟一次鼠标左键按下和松开，中间间隔0.03秒。
def click():
    # 调用函数：xtst.XTestFakeButtonEvent。
    xtst.XTestFakeButtonEvent(ctypes.c_void_p(dpy), 1, 1, 0)
    # 调用函数：xlib.XFlush。
    xlib.XFlush(ctypes.c_void_p(dpy))
    # 调用函数：time.sleep。
    time.sleep(0.03)
    # 调用函数：xtst.XTestFakeButtonEvent。
    xtst.XTestFakeButtonEvent(ctypes.c_void_p(dpy), 1, 0, 0)
    # 调用函数：xlib.XFlush。
    xlib.XFlush(ctypes.c_void_p(dpy))

# =============================================================================
# 【分区】触摸事件主循环
# 功能：非阻塞读 24 字节事件；按下移到映射坐标，抬起单击；设备断开则退出。
# 可修改：select 超时 0.2s、一次最多读 24*64 字节。
# 看情况改：坐标公式 val/TOUCH_W*SCREEN_W；若 X/Y 对调或翻转，只改这里的映射，不要改事件码。
# 不要改：24 字节步进和 BTN_TOUCH 边沿检测（val==1 按下、val==0 抬起）。
# =============================================================================
# 计算并保存到 fd（触摸设备文件描述符）。
fd = os.open(TOUCH_DEV, os.O_RDONLY)
# 调用函数：os.set_blocking。
os.set_blocking(fd, False)
# 调用函数：print。
print("touch2click 守护运行中")
# 计算并保存到 cur（最近收到的触摸坐标）。
cur = {}
# 计算并保存到 pressed（是否处于按下状态）。
pressed = False
# 只要条件成立就重复执行：True。
while True:
    # 执行可能出错的代码，并由后面的异常分支处理错误。
    try:
        # 计算并保存到 r、_（此处不需要使用的返回值或循环占位变量）、_（此处不需要使用的返回值或循环占位变量）。
        r, _, _ = select.select([fd], [], [], 0.2)
        # 判断条件；满足时执行下面代码：r。
        if r:
            # 计算并保存到 data（数据）。
            data = os.read(fd, 24 * 64)
            # 遍历数据，逐项处理：range(0, len(data), 24)。
            for i in range(0, len(data), 24):
                # 判断条件；满足时执行下面代码：i + 24 > len(data)。
                if i + 24 > len(data):
                    # 立即结束当前循环。
                    break
                # 计算并保存到 sec（输入事件时间戳的秒部分）、usec（输入事件时间戳的微秒部分）、et（Linux输入事件类型）、cd（Linux输入事件代码）、val（Linux输入事件携带的值）。
                sec, usec, et, cd, val = struct.unpack('llHHi', data[i:i + 24])
                # 判断条件；满足时执行下面代码：et == 3。
                if et == 3:
                    # 判断条件；满足时执行下面代码：cd == 53。
                    if cd == 53:
                        # 计算并保存到 cur['x']。
                        cur['x'] = val
                    # 判断条件；满足时执行下面代码：cd == 54。
                    elif cd == 54:
                        # 计算并保存到 cur['y']。
                        cur['y'] = val
                # 判断条件；满足时执行下面代码：et == 1 and cd == 330。
                elif et == 1 and cd == 330:
                    # 判断条件；满足时执行下面代码：val == 1 and (not pressed)。
                    if val == 1 and not pressed:
                        # 计算并保存到 pressed（是否处于按下状态）。
                        pressed = True
                        # 计算并保存到 sx。
                        sx = cur.get('x', 0) / TOUCH_W * SCREEN_W
                        # 计算并保存到 sy。
                        sy = cur.get('y', 0) / TOUCH_H * SCREEN_H
                        # 调用函数：move_mouse。
                        move_mouse(sx, sy)
                    # 判断条件；满足时执行下面代码：val == 0 and pressed。
                    elif val == 0 and pressed:
                        # 计算并保存到 pressed（是否处于按下状态）。
                        pressed = False
                        # 调用函数：click。
                        click()
    # 捕获BlockingIOError异常，转入下面的处理代码。
    except BlockingIOError:
        # 调用函数：time.sleep。
        time.sleep(0.05)
    # 捕获OSError异常，转入下面的处理代码。
    except OSError:
        # 立即结束当前循环。
        break