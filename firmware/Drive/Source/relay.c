/**
  *************************************************************************************************
  * @file    relay.c
  * @brief   继电器（电磁铁）控制实现
  *************************************************************************************************
  * @description
  *
  *   通过 PB12 推挽输出控制继电器线圈，高电平吸合、低电平释放。
  *   用于驱动电磁铁的通断。
  *
  ************************************************************************************************
  */

#include "relay.h"

/* 继电器硬件引脚：PB12，高电平吸合 */
#define RELAY_GPIO_PORT GPIOB
#define RELAY_GPIO_PIN  GPIO_Pin_12
#define RELAY_GPIO_CLK  RCC_AHB1Periph_GPIOB

/**
  * @brief  初始化继电器 IO 口（PB12，推挽输出，下拉，默认释放）
  */
void Relay_Init(void)
{
	GPIO_InitTypeDef gpio;

	RCC_AHB1PeriphClockCmd(RELAY_GPIO_CLK, ENABLE);
	GPIO_ResetBits(RELAY_GPIO_PORT, RELAY_GPIO_PIN); /* 先输出低电平，默认释放 */
	gpio.GPIO_Pin = RELAY_GPIO_PIN;
	gpio.GPIO_Mode = GPIO_Mode_OUT;
	gpio.GPIO_OType = GPIO_OType_PP;
	gpio.GPIO_Speed = GPIO_Speed_2MHz;
	gpio.GPIO_PuPd = GPIO_PuPd_DOWN;
	GPIO_Init(RELAY_GPIO_PORT, &gpio);
}

/**
  * @brief  设置继电器状态
  * @param  energized: 非 0=吸合（高电平），0=释放（低电平）
  */
void Relay_Set(uint8_t energized)
{
	if (energized != 0u)
	{
		GPIO_SetBits(RELAY_GPIO_PORT, RELAY_GPIO_PIN);
	}
	else
	{
		GPIO_ResetBits(RELAY_GPIO_PORT, RELAY_GPIO_PIN);
	}
}

/**
  * @brief  查询继电器当前状态
  * @retval 1=已吸合，0=已释放
  */
uint8_t Relay_IsEnergized(void)
{
	return (uint8_t)(GPIO_ReadOutputDataBit(RELAY_GPIO_PORT, RELAY_GPIO_PIN) != Bit_RESET);
}
