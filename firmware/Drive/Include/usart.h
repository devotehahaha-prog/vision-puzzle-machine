#ifndef __USART_H
#define __USART_H

#include "stdio.h"
#include "stm32f4xx.h"

/**
  * @file    usart.h
  * @brief   USART1 串口收发接口（含环形接收缓冲）
  *
  * @description
  *   配置 USART1 为 115200 8N1，PA9(TX)/PA10(RX) 复用功能。
  *   接收采用中断 + 软件环形缓冲区，发送采用阻塞轮询。
  *   重定向了 fputc，使 printf 输出到 USART1。
  */

/* 波特率 */
#define USART1_BaudRate          115200u

/* TX 引脚：PA9 */
#define USART1_TX_PIN            GPIO_Pin_9
#define USART1_TX_PORT           GPIOA
#define USART1_TX_CLK            RCC_AHB1Periph_GPIOA
#define USART1_TX_PinSource      GPIO_PinSource9

/* RX 引脚：PA10 */
#define USART1_RX_PIN            GPIO_Pin_10
#define USART1_RX_PORT           GPIOA
#define USART1_RX_CLK            RCC_AHB1Periph_GPIOA
#define USART1_RX_PinSource      GPIO_PinSource10

/**
  * @brief  配置 USART1 引脚、波特率、接收中断并使能
  */
void Usart_Config(void);

/**
  * @brief  阻塞发送字符串（直到遇到 '\0'）
  * @param  text: 待发送字符串指针
  */
void Usart_SendString(const char *text);

/**
  * @brief  从环形缓冲区读取一个字节
  * @param  byte: 输出字节指针
  * @retval 1=成功读取，0=缓冲区为空或指针无效
  */
uint8_t Usart_ReadByte(uint8_t *byte);

/**
  * @brief  取出并清零接收溢出标志
  * @retval 1=曾发生溢出，0=无溢出
  */
uint8_t Usart_TakeRxOverflow(void);

/**
  * @brief  USART1 接收中断处理函数，在 USART1_IRQHandler 中调用
  */
void Usart_RxIRQHandler(void);

#endif /* __USART_H */
