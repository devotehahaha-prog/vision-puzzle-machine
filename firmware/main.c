/**
  *************************************************************************************************
  * @file    main.c
  * @brief   步进电机驱动工程主程序
  *************************************************************************************************
  * @description
  *
  *   系统上电后依次初始化延时、LED、串口、A4988 驱动、继电器，
  *   并为各轴配置默认加减速参数。
  *   根据 APP_ENABLE_BUTTON_DEMO 选择运行模式：
  *     - 1：按键演示模式，按键触发 4 轴同步转动一圈，LED 指示方向翻转
  *     - 0：串口运动协议模式，循环接收并执行上位机运动指令
  *
  ************************************************************************************************
  */

#include "stm32f4xx.h"

#include "a4988.h"
#include "delay.h"
#include "key.h"
#include "led.h"
#include "motion_config.h"
#include "motion_protocol.h"
#include "relay.h"
#include "servo.h"
#include "usart.h"

/**
  * @brief  为所有轴配置默认加减速参数（起始 200Hz，最大 1000Hz，50 步加减速）
  */
static void ConfigureMotorProfiles(void)
{
	A4988_Status status;
	uint8_t axis;

	for (axis = 0u; axis < A4988_ACTIVE_MOTORS; axis++)
	{
		status = A4988_SetProfile(axis, 200u, 1000u, 50u);
		if (status != A4988_OK)
		{
			printf("ERR,PROFILE,%u,%d\n", (unsigned int)axis, (int)status);
		}
	}
}

#if APP_ENABLE_BUTTON_DEMO
/**
  * @brief  按键演示模式：按键触发 4 轴同步转动一圈，完成后翻转方向
  */
static void RunButtonDemo(void)
{
	A4988_MoveCommand commands[A4988_ACTIVE_MOTORS];
	A4988_Direction direction = A4988_DIR_FORWARD;
	A4988_Status status;
	uint8_t was_busy = 0u;
	uint8_t axis;

	KEY_Init();
	/* Use the configured full-turn pulse count for each axis. */
	for (axis = 0u; axis < A4988_ACTIVE_MOTORS; axis++)
	{
		commands[axis].id = axis;
		commands[axis].direction = direction;
		commands[axis].pulses = (axis == A4988_AXIS_ROTATE) ?
			MOTION_R_PULSES_PER_REV : A4988_PULSES_PER_REV;
	}
	Usart_SendString("BOOT,BUTTON_DEMO,READY\n");

	while (1)
	{
		uint8_t busy = A4988_AnyBusy();
		uint8_t key_state = KEY_Scan();

		/* 运动刚结束时翻转 LED 与运动方向 */
		if ((was_busy != 0u) && (busy == 0u))
		{
			LED1_Toggle;
			direction = (direction == A4988_DIR_FORWARD) ?
				A4988_DIR_REVERSE : A4988_DIR_FORWARD;
		}
		was_busy = busy;
		/* 空闲且检测到按键按下时启动同步运动 */
		if ((busy == 0u) && (key_state == KEY_ON))
		{
		for (axis = 0u; axis < A4988_ACTIVE_MOTORS; axis++)
		{
			commands[axis].direction = direction;
		}
		commands[A4988_AXIS_X_RIGHT].direction =
			(A4988_Direction)(direction ^ MOTION_X_RIGHT_DIRECTION_INVERT);
			status = A4988_MoveGroup(commands, A4988_ACTIVE_MOTORS);
			if (status == A4988_OK)
			{
				was_busy = 1u;
			}
		}
	}
}
#endif

/**
  * @brief  主函数
  *         初始化各外设，根据配置选择按键演示或串口协议模式
  */
int main(void)
{
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_4); /* 4 位抢占优先级 */
	Delay_Init();        /* SysTick 1ms 时基 */
	LED_Init();          /* LED 初始化 */
	Usart_Config();      /* 串口 115200 */
	Relay_Init();
	Servo_Init();
	A4988_Init();        /* 步进电机驱动初始化 */
	ConfigureMotorProfiles(); /* 配置各轴加减速参数 */

#if APP_ENABLE_BUTTON_DEMO
	RunButtonDemo();
#else
	/* 串口运动协议模式 */
	MotionProtocol_Init();
	Usart_SendString("BOOT,MOTION,READY\n");
	while (1)
	{
		MotionProtocol_Process();
	}
#endif
}
