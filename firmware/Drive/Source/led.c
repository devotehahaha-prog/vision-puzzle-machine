/***
	*************************************************************************************************
	*	@version V1.0
	*	@brief   LED 接口函数
   *************************************************************************************************
   *  @description
	*
	*	初始化 LED 的 IO 口，配置为推挽输出、速度等级 2MHz。
	*
	************************************************************************************************
***/

#include "led.h"

/*************************************************************************************************
*	函数名:	LED_Init
*
*	功能描述:	LED IO 口初始化
*
*************************************************************************************************/

void LED_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure; /* GPIO 初始化结构体 */
	RCC_AHB1PeriphClockCmd(LED1_CLK, ENABLE); /* 使能 LED1 端口时钟 */

	GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_OUT;    /* 输出模式 */
	GPIO_InitStructure.GPIO_OType = GPIO_OType_PP;    /* 推挽输出 */
	GPIO_InitStructure.GPIO_PuPd  = GPIO_PuPd_UP;     /* 上拉 */
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz;  /* 速度等级 2MHz */

	/* 初始化 LED1 引脚 */
	GPIO_InitStructure.GPIO_Pin = LED1_PIN;
	GPIO_Init(LED1_PORT, &GPIO_InitStructure);

	GPIO_ResetBits(LED1_PORT, LED1_PIN); /* 输出低电平，点亮 LED1 */
}
