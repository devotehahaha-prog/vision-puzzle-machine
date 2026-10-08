#ifndef __MOTION_CONFIG_H
#define __MOTION_CONFIG_H

/**
  * @file    motion_config.h
  * @brief   运动标定参数配置
  *
  * @description
  *   本文件集中存放各轴的标定参数与行程限位，在装机测量后填写。
  *   脉冲与单位的换算关系（X100 表示 0.01 单位）：
  *     pulses = position_x100 * PULSES_NUM / UNITS_X100_DEN
  *   X/Y/Z 轴单位为 0.01 mm，R 轴单位为 0.01 度。
  *   若分子或分母为 0，表示该轴尚未标定。
  */

/* X 轴：每 0.01 mm 对应的脉冲换算系数（标定后填写） */
#define MOTION_X_PULSES_NUM         0u
#define MOTION_X_UNITS_X100_DEN     0u

/* Y 轴：每 0.01 mm 对应的脉冲换算系数（标定后填写） */
#define MOTION_Y_PULSES_NUM         0u
#define MOTION_Y_UNITS_X100_DEN     0u

/* Z 轴：每 0.01 mm 对应的脉冲换算系数（标定后填写） */
#define MOTION_Z_PULSES_NUM         0u
#define MOTION_Z_UNITS_X100_DEN     0u

/*
 * R axis motor and mechanism defaults.
 * A typical 42 mm motor is 1.8 degrees/step (200 full steps/rev).
 * A4988 MS1=MS2=MS3=HIGH selects 1/16 microstepping.
 * The default assumes a direct 1:1 coupling to the output shaft.
 * For a gearbox, set GEAR_NUM/GEAR_DEN to motor revs/output revs.
 */
#define MOTION_R_FULL_STEPS_PER_REV 200u
#define MOTION_R_MICROSTEPS         16u
#define MOTION_R_GEAR_NUM           1u
#define MOTION_R_GEAR_DEN           1u
#define MOTION_R_PULSES_PER_REV     (MOTION_R_FULL_STEPS_PER_REV * MOTION_R_MICROSTEPS)

/* R axis: pulses = angle_x100 * PULSES_NUM / UNITS_X100_DEN */
#define MOTION_R_PULSES_NUM         (MOTION_R_PULSES_PER_REV * MOTION_R_GEAR_NUM)
#define MOTION_R_UNITS_X100_DEN     (36000u * MOTION_R_GEAR_DEN)

/* 旧版 Z 轴 UP/DOWN 使用固定行程，测量机构行程后填写（单位：脉冲） */
#define MOTION_Z_STROKE_PULSES      0u

/*
 * 各轴正方向定义。A4988_DIR_FORWARD=0，A4988_DIR_REVERSE=1。
 * 根据实际接线方向调整，使坐标值增大的方向为正方向。
 */
#define MOTION_X_POSITIVE_DIRECTION 0u
#define MOTION_X_RIGHT_DIRECTION_INVERT 1u
#define MOTION_Y_POSITIVE_DIRECTION 0u
#define MOTION_Z_POSITIVE_DIRECTION 0u
#define MOTION_R_POSITIVE_DIRECTION 0u

/*
 * 各轴行程限位（单位：0.01 mm / 0.01 度）。
 * 将 ENABLE 置 1 后限位生效，GOTO 等指令会校验目标是否在范围内。
 */
#define MOTION_X_LIMIT_ENABLE       0u
#define MOTION_X_LIMIT_MIN_X100     0L
#define MOTION_X_LIMIT_MAX_X100     0L

#define MOTION_Y_LIMIT_ENABLE       0u
#define MOTION_Y_LIMIT_MIN_X100     0L
#define MOTION_Y_LIMIT_MAX_X100     0L

#define MOTION_Z_LIMIT_ENABLE       0u
#define MOTION_Z_LIMIT_MIN_X100     0L
#define MOTION_Z_LIMIT_MAX_X100     0L

#define MOTION_R_LIMIT_ENABLE       0u
#define MOTION_R_LIMIT_MIN_X100     0L
#define MOTION_R_LIMIT_MAX_X100     0L

/* 按键演示与串口协议互斥：1=启用按键演示，0=启用串口运动协议 */
#define APP_ENABLE_BUTTON_DEMO      0u

/* REAL mode holds shared EN between moves. Exact PING/STATUS/IOSTATUS
 * heartbeats must arrive within this interval; expiry disarms the session. */
#define MOTION_HOLD_LEASE_MS        2000u

#endif /* __MOTION_CONFIG_H */
