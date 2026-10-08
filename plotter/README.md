# 通用写字机控制端

该工具通过 Grbl 文本协议操作已有运动控制板，不依赖原厂桌面软件。

| 命令 | 功能 | 是否访问硬件 |
| --- | --- | --- |
| `machine --json` | 查看默认或用户提供的机器配置 | 否 |
| `svg 输入.svg 输出.nc` | 将直线路径转换为 G-code | 否 |
| `send 端口 文件.nc` | 打印发送预览 | 否 |
| `probe 端口` | 查询固件、参数和状态 | 是，不主动发送运动指令 |
| `send 端口 文件.nc --execute` | 逐行发送并等待应答 | 是，可能运动 |

在仓库根目录运行：

```bash
python -m pip install -r plotter/requirements.txt
python -m plotter.grbl_writer machine
python -m plotter.grbl_writer svg plotter/example.svg preview.nc --z-up 0 --z-down -1 --feed 600
python -m plotter.grbl_writer send COM3 preview.nc
python -m plotter.grbl_writer probe COM3
```

仅在确认端口、坐标、Z 方向、行程和速度后，给 `send` 加 `--execute`。此工具的历史 Z 约定是下降为负，与拼图项目的 Z 约定可能不同，不能混用配置。

串口使用 115200、8N1。普通命令以 LF 结尾，等待 `ok` 或 `error:n`；`?`、`!`、`~`、`0x18` 是实时单字节命令。查询使用 `$I`、`$G`、`$$`、`$#` 与 `?`。连接操作本身可能受控制板 DTR/RTS 电路影响，查询不发运动指令不等于连接对板卡绝无副作用。

SVG 支持 line、polyline、polygon 以及 M/L/H/V/Z 路径，支持缩放、XY 偏移和 Y 翻转。不提供字体库、文字排版、任意曲线或完整 SVG transform 支持。建议先在文件中将图形转换为明确的直线路径。

自定义配置可通过 `--settings-dir` 或 `PLOTTER_SETTINGS_DIR` 提供。也会搜索 `.plotter` 目录。可选文件为 `Grbl.json`、`settings.json`、`MachineControllerData.json`、`AxesData.json`，字段读取逻辑见 `load_machine_profile`。不附带任何原厂配置包或授权数据。
