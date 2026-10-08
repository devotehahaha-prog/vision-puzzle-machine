# 部署与配置

## 环境与入口

主要类型：Python 视觉与串口应用、Linux 用户态 SPI/I2C 界面、C 语言辅助固件。

Python 无编译步骤；核心入口为 `spi_control/start_page.py`、`E题开源/完赛/puzzle_vision_poker/run_real.py` 和 `plotter/grbl_writer.py`。辅助固件原工程使用 ARMCC 5，具体集成范围见 `firmware/README.md`。

## 桌面离线使用

```bash
python -m venv .venv
# 按当前系统激活虚拟环境后：
python -m pip install -r requirements.txt
python -m plotter.grbl_writer --help
python -m plotter.grbl_writer svg plotter/example.svg preview.nc --z-up 0 --z-down -1 --feed 600
python -m plotter.grbl_writer send COM3 preview.nc
```

`COM3` 仅为端口示例，预览发送不访问它。相机规划需要真实相机和本机标定，不能把缺少标定的启动失败解释为安装成功后的功能验收。

## Linux 触屏环境

启动脚本固定使用系统 `/usr/bin/python3` 运行界面，视觉使用仓库根目录 `.venv/bin/python`。分别安装依赖：

```bash
sudo apt install python3-venv python3-numpy python3-opencv python3-serial python3-pil python3-spidev python3-smbus fonts-noto-cjk
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install -r requirements-screen.txt
```

系统解释器还需 `smbus2`。如果发行版提供 `python3-smbus2`，安装该包；否则按发行版支持方式给系统解释器安装。`smbus` 和 `smbus2` 不是同一个 Python 模块。先确认 `/usr/bin/python3 -c "import serial, PIL, spidev, smbus2"` 成功。

从 [已有驱动仓库](https://github.com/devotehahaha-prog/orangepi-lt758x-spi-touch) 下载驱动，放在 `external/spi-display-driver` 或设置：

```bash
export SPI_DRIVER_DIR=/absolute/path/to/spi-display-driver
bash start-screen.sh --check-env
bash start-screen.sh
```

启用本机对应的 SPI/I2C 设备树配置、GPIO 工具和设备访问权限，按驱动的硬件说明核对接线。屏幕尺寸为 800×480，默认节点为 `/dev/spidev0.0` 与 `/dev/i2c-1`，不能假定其他开发板引脚编号一致。

## 配置自己的机构

1. 修改 `spi_control/manual_config.json` 中两路串口与速度档位。
2. 修改三片项目 `production_config.json` 的相机设备、串口和硬件参数。
3. 在真实分辨率下保存四角标定，核对 A4 物理尺寸、原点与坐标方向。
4. 分别实测 XY 变换、工具偏移、Z 安全/拾取/释放高度及 R 正方向。
5. 建立本轮基准，完成空载检查后才进入实物流程。

公开配置主动清除了现场角点、透视矩阵、验收范围和操作员确认，并将 `calibration_complete` 及分项标志设为 false、`require_empty_run` 设为 true。剩余数值仍包含历史机构示例，不应直接用来驱动其他机器。

界面普通四片模式与三片模式共享相机与机械标定。普通四片的实机正式入口是触屏调用 `ordinary_camera.py`；旧普通目录中的 `run_real.py` 仅作为历史代码保留。

## 软件检查

```bash
python -m pip install -r requirements-dev.txt
cd spi_control
python -m pytest -q tests
cd ../E题开源/完赛/puzzle_vision_poker
python -m pytest -q --ignore=test_sim.py
```

两套测试分别运行，避免同名 `main`、`config` 等模块发生导入冲突。当前仿真缺少历史 `assets/` 场景，仿真测试不计入上述检查。软件结果及平台限制见 `VALIDATION.md`。
