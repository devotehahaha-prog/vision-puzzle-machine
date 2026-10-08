# 辅助控制器应用层

此目录发布自定义 `Drive/` 与 `main.c`，用于步进运动、R 旋转、电磁铁、辅助输出和串口协议。目标硬件为 STM32F407，原工程使用标准外设库和 ARMCC 5。

未收录第三方芯片 SDK、启动汇编、系统时钟文件、厂商中断模板、工程管理文件或二进制。因此这里不是可直接一键烧录的完整 SDK 工程，也没有在本次开源整理中进行固件编译或硬件测试。

## 集成

在匹配的 SDK 工程中添加 `Drive/Source/*.c` 和 `main.c`，头文件路径加入 `Drive/Include`。配置启动文件、芯片型号、时钟与外设库；实现 SysTick 毫秒时基，以及串口接收和步进定时器的中断转发。请逐项对照 `delay.h`、`usart.h`、`a4988.h` 中的接口，不能仅复制主循环而忽略中断。

参考引脚：X 左 STEP PA0/DIR PE0、X 右 STEP PA8/DIR PE8、Y STEP PA6/DIR PE2、Z STEP PB6/DIR PE4、R STEP PA1/DIR PE6，共用 EN PE1。正式拼图架构只使用该控制器的 R 与辅助输出，XYZ 由另一控制器承担。USART1 使用 PA9/PA10。

`motion_config.h` 中 XYZ 脉冲标定默认未填写；R 默认 3200 脉冲/圈。`servo_config.h` 定义舵机参数。不得将示例值当作现场标定结果。

## 协议概览

使用 115200、8N1，ASCII 行以 LF 结束。主要命令为 `PING`、`STATUS`、`IOSTATUS`、`MOTION,0/1`、`ZERO`、`GOTO`、`MOVE`、`ROTATE`、`AUX,0/1`、`MAGNET,0/1`、`SERVO` 与 `STOP`。以 `motion_protocol.c` 为具体语法及应答依据。

上电进入 SIM（运动输出未使能），启用实动需显式命令；磁铁还受 AUX 和通信租约约束。运动租约与 AUX 输出租约分别检查，建议状态查询间隔不超过 0.8 秒。`STOP` 应关闭运动和输出，主机必须检查应答及状态。

保留外部 SDK 原有版权和许可；本仓库 MIT 许可仅适用于这里发布的应用层，不授予外部 SDK 的许可。
