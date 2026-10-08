#ifndef __DELAY_H
#define __DELAY_H

#include "stm32f4xx.h"

/**
  * @brief  初始化 SysTick 为 1ms 周期中断，作为毫秒延时的时基
  */
void Delay_Init(void);

/**
  * @brief  毫秒级阻塞延时
  * @param  nTime: 延时时间，单位 ms
  */
void Delay_ms(uint32_t nTime);

#endif /* __DELAY_H */
