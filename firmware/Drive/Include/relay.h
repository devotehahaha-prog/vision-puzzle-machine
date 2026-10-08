#ifndef __RELAY_H
#define __RELAY_H

#include "stm32f4xx.h"

/**
  * @file    relay.h
  * @brief   继电器（电磁铁）控制接口
  *
  * @description
  *   控制一路 GPIO 输出驱动继电器，用于吸合/释放电磁铁。
  *   硬件：PB12 推挽输出，高电平吸合。
  */

/**
  * @brief  初始化继电器 IO 口（PB12，推挽输出，默认释放）
  */
void Relay_Init(void);

/**
  * @brief  设置继电器状态
  * @param  energized: 非 0=吸合，0=释放
  */
void Relay_Set(uint8_t energized);

/**
  * @brief  查询继电器当前状态
  * @retval 1=已吸合，0=已释放
  */
uint8_t Relay_IsEnergized(void);

#endif /* __RELAY_H */
