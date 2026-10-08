#ifndef __KEY_H
#define __KEY_H

#include "stm32f4xx.h"
#include "delay.h"

/* 按键状态：按下 / 松开 */
#define KEY_ON   1   /* 按键按下 */
#define KEY_OFF  0   /* 按键松开 */

/*---------------------- 按键硬件引脚 ------------------------*/

#define KEY_PIN           GPIO_Pin_15         /* KEY 引脚 */
#define KEY_PORT          GPIOA               /* KEY GPIO 端口 */
#define KEY_CLK           RCC_AHB1Periph_GPIOA /* KEY GPIO 端口时钟 */

/*---------------------- 对外函数 ----------------------------*/

/**
  * @brief  初始化按键 IO 口（输入、上拉）
  */
void KEY_Init(void);

/**
  * @brief  按键扫描，带消抖与松手检测
  * @retval KEY_ON=检测到按下并松开，KEY_OFF=未按下
  */
uint8_t KEY_Scan(void);

#endif /* __KEY_H */
