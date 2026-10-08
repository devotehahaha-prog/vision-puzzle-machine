/**
  *************************************************************************************************
  * @file    a4988.h
  * @brief   A4988 步进电机驱动接口（五电机四逻辑轴版本）
  *************************************************************************************************
  * @description
  *
  *   本头文件向应用层提供 A4988 步进电机驱动的全部 API：
  *     - 初始化 5 路步进电机（X左 / X右 / Y / Z / 旋转轴）
  *     - 设置各轴加减速参数（起始频率、最大频率、加减速步数）
  *     - 单轴运动 A4988_Move，以及多轴同步运动 A4988_MoveGroup
  *     - 单轴停止 A4988_Stop，全部停止 A4988_StopAll
  *     - 忙状态查询、已发送脉冲数查询
  *
  *   硬件对应 CNC Shield v3 扩展板引脚分配：
  *     X      STEP=PA0 (TIM2_CH1)，DIR=PE0
  *     Y      STEP=PA6 (TIM3_CH1)，DIR=PE2
  *     Z      STEP=PB6 (TIM4_CH1)，DIR=PE4
  *     ROTATE STEP=PA1 (TIM5_CH2)，DIR=PE6
  *     EN     =PE1（所有步进共用，低电平有效）
  *
  ************************************************************************************************
  */

#ifndef __A4988_H
#define __A4988_H

#include "stm32f4xx.h"

/* 硬件资源限制：最多支持 5 路电机，X右使用 TIM1 */
#define A4988_MAX_MOTORS       5u
/* 实际使用的电机数量，取值范围 1~A4988_MAX_MOTORS */
#define A4988_ACTIVE_MOTORS    5u

/* 轴编号，对应 CNC Shield 上的各轴接口 */
#define A4988_AXIS_X_LEFT      0u   /* X 左电机  */
#define A4988_AXIS_Y           1u   /* Y 轴      */
#define A4988_AXIS_Z           2u   /* Z 轴      */
#define A4988_AXIS_ROTATE      3u   /* 旋转轴    */
#define A4988_AXIS_X_RIGHT     4u   /* X 右电机  */
#define A4988_AXIS_X           A4988_AXIS_X_LEFT

/* 兼容旧代码的别名，直接映射到轴编号 */
#define A4988_MOTOR_1          A4988_AXIS_X_LEFT
#define A4988_MOTOR_2          A4988_AXIS_X_RIGHT
#define A4988_MOTOR_3          A4988_AXIS_Y
#define A4988_MOTOR_4          A4988_AXIS_Z
#define A4988_MOTOR_5          A4988_AXIS_ROTATE

/*
 * Generic full-step reference used by the linear-axis demo.
 * The R axis has its own 1/16-microstep count in motion_config.h.
 */
#define A4988_PULSES_PER_REV   200u

/* 编译期检查：电机数量必须在 [1, A4988_MAX_MOTORS] 范围内 */
#if (A4988_ACTIVE_MOTORS < 1u) || (A4988_ACTIVE_MOTORS > A4988_MAX_MOTORS)
#error "A4988_ACTIVE_MOTORS must be between 1 and A4988_MAX_MOTORS"
#endif

/**
  * @brief 驱动 API 返回状态
  */
typedef enum
{
	A4988_OK = 0,        /* 操作成功                          */
	A4988_ERROR_ID,     /* 电机/轴 ID 越界                   */
	A4988_ERROR_BUSY,   /* 指定电机正在运动中，无法修改或启动 */
	A4988_ERROR_PARAM   /* 参数不合法（如 0 脉冲、重复轴等） */
} A4988_Status;

/**
  * @brief 电机运动方向
  */
typedef enum
{
	A4988_DIR_FORWARD = 0, /* 正转（DIR=0） */
	A4988_DIR_REVERSE = 1  /* 反转（DIR=1） */
} A4988_Direction;

/**
  * @brief 单轴运动指令，A4988_MoveGroup 的数组元素
  */
typedef struct
{
	uint8_t id;                    /* 轴号，取 A4988_AXIS_x 等      */
	A4988_Direction direction;     /* 运动方向                        */
	uint32_t pulses;               /* 目标脉冲数                      */
} A4988_MoveCommand;

/*---------------------- 对外 API ----------------------*/

/**
  * @brief  初始化所有 EN 引脚和各电机硬件资源（GPIO、定时器、NVIC）
  */
void A4988_Init(void);

/* Hold the shared active-low EN during a leased real-motion session.
 * STEP remains low while idle; StopAll always clears this hold. */
void A4988_SetHoldEnabled(uint8_t enabled);

/**
  * @brief  设置指定轴的加减速曲线参数
  * @param  id:          轴号
  * @param  start_hz:    起始频率
  * @param  max_hz:      最大运行频率
  * @param  accel_steps: 加减速步数
  * @retval A4988_OK=成功；A4988_ERROR_ID=ID 非法；A4988_ERROR_BUSY=运动中；
  *         A4988_ERROR_PARAM=参数不合法
  */
A4988_Status A4988_SetProfile(uint8_t id, uint32_t start_hz,
									 uint32_t max_hz, uint32_t accel_steps);

/**
  * @brief  单轴运动（包装单条指令并在内部调用 A4988_MoveGroup）
  * @param  id:        轴号
  * @param  direction: 方向
  * @param  pulses:    目标脉冲数
  * @retval A4988_Status
  */
A4988_Status A4988_Move(uint8_t id, A4988_Direction direction,
								 uint32_t pulses);

A4988_Status A4988_MoveXPair(A4988_Direction direction, uint32_t pulses);

/**
  * @brief  多轴同步运动：先准备全部指令，再统一使能、同时启动各轴定时器
  * @param  commands: 指令数组指针（不可为 0，同一批指令中轴号不得重复）
  * @param  count:    指令数量（1~A4988_ACTIVE_MOTORS）
  * @retval A4988_OK=成功；任一指令非法时拒绝执行，返回对应错误码
  */
A4988_Status A4988_MoveGroup(const A4988_MoveCommand *commands, uint8_t count);

/**
  * @brief  立即停止指定电机（非减速停），完成后根据全局状态更新 EN
  * @param  id: 轴号
  */
void A4988_Stop(uint8_t id);

/**
  * @brief  立即停止全部电机并释放共用 EN
  */
void A4988_StopAll(void);

/**
  * @brief  查询指定轴是否正在运动
  * @param  id: 轴号
  * @retval 1=忙碌，0=空闲
  */
uint8_t A4988_IsBusy(uint8_t id);

/**
  * @brief  查询是否有任一轴正在运动
  * @retval 1=至少一个轴忙碌，0=全部空闲
  */
uint8_t A4988_AnyBusy(void);

/**
  * @brief  查询指定轴已发送的脉冲数
  * @param  id: 轴号
  * @retval 已发送脉冲数
  */
uint32_t A4988_GetCompletedPulses(uint8_t id);

/**
  * @brief  定时器更新中断处理函数
  *         在 TIM2/TIM3/TIM4/TIM5 的 IRQHandler 中调用，传入对应轴号
  * @param  id: 轴号
  */
void A4988_TimerIRQHandler(uint8_t id);

#endif
