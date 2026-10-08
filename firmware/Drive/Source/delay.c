/***
	***************************************************************************
	*	@file  	delay.c
	*	@brief   delay 接口函数实现
   ***************************************************************************
   *  @description
	*
	*  配置 SysTick 定时器为 1ms 中断，实现毫秒级阻塞延时。
	*
	***************************************************************************
***/

#include "delay.h"

/* 延时计数器，由 SysTick 中断递减 */
static __IO uint32_t TimingDelay;

/**
  * @brief  延时模块初始化
  *         配置 SysTick 为 1ms 中断，作为毫秒延时的时基
  */
void Delay_Init(void)
{
	SysTick_Config(SystemCoreClock / 1000);  /* 配置 SysTick 为 1ms 中断 */
}

/**
  * @brief  延时计数器递减函数
  *         在 SysTick 中断服务函数中被调用
  */
void TimingDelay_Decrement(void)
{
	if (TimingDelay != 0)
	{
		TimingDelay--;
	}
}

/**
  * @brief  毫秒级阻塞延时
  * @param  nTime: 延时时间，单位 ms
  * @note   每次调用重新给 TimingDelay 赋值，实现 n 毫秒延时
  *         最大延时 4294967295 ms
  */
void Delay_ms(uint32_t nTime)
{
	TimingDelay = nTime;

	while (TimingDelay != 0)
		;
}
