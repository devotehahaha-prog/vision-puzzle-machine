#ifndef __LED_H
#define __LED_H

#include "stm32f4xx.h"

/*---------------------- LED 硬件引脚 ------------------------*/

#define LED1_PIN    GPIO_Pin_13            /* LED1 引脚 */
#define LED1_PORT   GPIOC                  /* LED1 GPIO 端口 */
#define LED1_CLK    RCC_AHB1Periph_GPIOC   /* LED1 GPIO 端口时钟 */

/*---------------------- LED 控制宏 ------------------------*/

#define LED1_ON      GPIO_ResetBits(LED1_PORT, LED1_PIN)   /* 输出低电平点亮 LED1 */
#define LED1_OFF     GPIO_SetBits(LED1_PORT, LED1_PIN)     /* 输出高电平熄灭 LED1 */
#define LED1_Toggle  GPIO_ToggleBits(LED1_PORT, LED1_PIN)  /* LED1 状态翻转 */

/*---------------------- 对外函数 ----------------------------*/

/**
  * @brief  初始化 LED IO 口（推挽输出，默认点亮）
  */
void LED_Init(void);

#endif /* __LED_H */
