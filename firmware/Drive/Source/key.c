/***
	*************************************************************************************************
	*	@version V1.0
	*	@brief   按键接口函数
   *************************************************************************************************
   *  @description
	*
	*	按键 IO 口初始化，配置为输入上拉、速度等级 2MHz。
	*
	************************************************************************************************
***/

#include "key.h"

/*************************************************************************************************
*	函数名:	KEY_Init
*
*	功能描述:	按键 IO 口初始化
*
*************************************************************************************************/

void KEY_Init(void)
{
	GPIO_InitTypeDef GPIO_InitStructure; /* GPIO 初始化结构体 */
	RCC_AHB1PeriphClockCmd(KEY_CLK, ENABLE); /* 使能 KEY 端口时钟 */

	GPIO_InitStructure.GPIO_Mode  = GPIO_Mode_IN;    /* 输入模式 */
	GPIO_InitStructure.GPIO_PuPd  = GPIO_PuPd_UP;    /* 上拉 */
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_2MHz; /* 速度等级 2MHz */
	GPIO_InitStructure.GPIO_Pin   = KEY_PIN;

	GPIO_Init(KEY_PORT, &GPIO_InitStructure);
}

/*************************************************************************************************
*	函数名:	KEY_Scan
*
*	返回值:	KEY_ON - 按键按下，KEY_OFF - 按键松开
*
*	功能描述:	按键扫描
*
*************************************************************************************************/

uint8_t KEY_Scan(void)
{
	if (GPIO_ReadInputDataBit(KEY_PORT, KEY_PIN) == 0) /* 检测按键是否被按下 */
	{
		Delay_ms(10); /* 延时消抖 */
		if (GPIO_ReadInputDataBit(KEY_PORT, KEY_PIN) == 0) /* 再次确认是否为低电平 */
		{
			while (GPIO_ReadInputDataBit(KEY_PORT, KEY_PIN) == 0)
				; /* 等待按键松开 */
			return KEY_ON; /* 返回按键按下标志 */
		}
	}
	return KEY_OFF;
}
