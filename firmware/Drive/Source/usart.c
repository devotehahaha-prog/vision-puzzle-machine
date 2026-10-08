/**
  *************************************************************************************************
  * @file    usart.c
  * @brief   USART1 串口收发实现（中断接收 + 环形缓冲区）
  *************************************************************************************************
  * @description
  *
  *   配置 USART1 为 115200 8N1，PA9(TX)/PA10(RX) 复用功能。
  *   接收：开启 RXNE 中断，在中断中把字节写入软件环形缓冲区。
  *   发送：阻塞轮询 TXE 标志，逐字节发送。
  *   重定向 fputc，使 printf 输出到 USART1。
  *
  ************************************************************************************************
  */

#include "usart.h"

/* 接收环形缓冲区大小（2 的幂，便于用位与代替取模） */
#define USART_RX_BUFFER_SIZE 256u
#define USART_RX_BUFFER_MASK (USART_RX_BUFFER_SIZE - 1u)

/* 接收环形缓冲区及读写指针、溢出标志（均为 volatile，中断与主循环共享） */
static volatile uint8_t usart_rx_buffer[USART_RX_BUFFER_SIZE];
static volatile uint16_t usart_rx_head;   /* 写入位置（中断更新） */
static volatile uint16_t usart_rx_tail;   /* 读取位置（主循环更新） */
static volatile uint8_t usart_rx_overflow; /* 接收溢出标志 */

/**
  * @brief  配置 USART1 的 TX/RX GPIO 引脚为复用功能
  */
static void USART_GPIO_Config(void)
{
	GPIO_InitTypeDef gpio_init;

	RCC_AHB1PeriphClockCmd(USART1_TX_CLK | USART1_RX_CLK, ENABLE);
	GPIO_StructInit(&gpio_init);
	gpio_init.GPIO_Mode = GPIO_Mode_AF;
	gpio_init.GPIO_OType = GPIO_OType_PP;
	gpio_init.GPIO_PuPd = GPIO_PuPd_UP;
	gpio_init.GPIO_Speed = GPIO_Speed_2MHz;

	/* TX 引脚 PA9 */
	gpio_init.GPIO_Pin = USART1_TX_PIN;
	GPIO_Init(USART1_TX_PORT, &gpio_init);
	/* RX 引脚 PA10 */
	gpio_init.GPIO_Pin = USART1_RX_PIN;
	GPIO_Init(USART1_RX_PORT, &gpio_init);

	GPIO_PinAFConfig(USART1_TX_PORT, USART1_TX_PinSource, GPIO_AF_USART1);
	GPIO_PinAFConfig(USART1_RX_PORT, USART1_RX_PinSource, GPIO_AF_USART1);
}

/**
  * @brief  配置 USART1：引脚、波特率、接收中断、NVIC 并使能
  */
void Usart_Config(void)
{
	USART_InitTypeDef usart_init;
	NVIC_InitTypeDef nvic_init;

	/* 初始化环形缓冲区状态 */
	usart_rx_head = 0u;
	usart_rx_tail = 0u;
	usart_rx_overflow = 0u;

	/* 使能 USART1 时钟并配置 GPIO */
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_USART1, ENABLE);
	USART_GPIO_Config();

	/* USART 参数：115200，8 数据位，1 停止位，无校验，收发模式，无硬件流控 */
	USART_StructInit(&usart_init);
	usart_init.USART_BaudRate = USART1_BaudRate;
	usart_init.USART_WordLength = USART_WordLength_8b;
	usart_init.USART_StopBits = USART_StopBits_1;
	usart_init.USART_Parity = USART_Parity_No;
	usart_init.USART_Mode = USART_Mode_Rx | USART_Mode_Tx;
	usart_init.USART_HardwareFlowControl = USART_HardwareFlowControl_None;
	USART_Init(USART1, &usart_init);

	/* 配置接收中断 NVIC（抢占优先级 2，子优先级 0） */
	nvic_init.NVIC_IRQChannel = USART1_IRQn;
	nvic_init.NVIC_IRQChannelPreemptionPriority = 2u;
	nvic_init.NVIC_IRQChannelSubPriority = 0u;
	nvic_init.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&nvic_init);

	/* 使能接收中断与 USART1 */
	USART_ITConfig(USART1, USART_IT_RXNE, ENABLE);
	USART_Cmd(USART1, ENABLE);
}

/**
  * @brief  阻塞发送字符串，直到遇到字符串结束符 '\0'
  * @param  text: 待发送字符串指针
  */
void Usart_SendString(const char *text)
{
	if (text == 0)
	{
		return;
	}
	while (*text != '\0')
	{
		USART_SendData(USART1, (uint8_t)*text);
		/* 等待发送数据寄存器空 */
		while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET)
		{
		}
		text++;
	}
}

/**
  * @brief  从环形缓冲区读取一个字节
  * @param  byte: 输出字节指针
  * @retval 1=成功读取，0=缓冲区为空或指针无效
  */
uint8_t Usart_ReadByte(uint8_t *byte)
{
	if ((byte == 0) || (usart_rx_tail == usart_rx_head))
	{
		return 0u;
	}
	*byte = usart_rx_buffer[usart_rx_tail];
	/* 读取指针前移，用位与实现环形回绕 */
	usart_rx_tail = (uint16_t)((usart_rx_tail + 1u) & USART_RX_BUFFER_MASK);
	return 1u;
}

/**
  * @brief  取出并清零接收溢出标志
  * @retval 1=曾发生溢出，0=无溢出
  */
uint8_t Usart_TakeRxOverflow(void)
{
	uint8_t overflow = usart_rx_overflow;

	usart_rx_overflow = 0u;
	return overflow;
}

/**
  * @brief  USART1 接收中断处理函数，由 USART1_IRQHandler 调用
  *
  *         读取状态寄存器判断中断源，读取数据寄存器清除 RXNE 标志。
  *         若发生帧错误/噪声/溢出/奇偶校验错误，置溢出标志。
  *         若缓冲区满，置溢出标志并丢弃当前字节。
  */
void Usart_RxIRQHandler(void)
{
	uint16_t status = USART1->SR;
	uint8_t byte;
	uint16_t next;

	/* 无相关中断源则返回 */
	if ((status & (USART_SR_RXNE | USART_SR_ORE | USART_SR_NE | USART_SR_FE)) == 0u)
	{
		return;
	}
	/* 读 DR 清除 RXNE 标志 */
	byte = (uint8_t)USART1->DR;

	/* 检测错误标志，置溢出标记 */
	if ((status & (USART_SR_ORE | USART_SR_NE | USART_SR_FE | USART_SR_PE)) != 0u)
	{
		usart_rx_overflow = 1u;
	}
	/* 非 RXNE 中断则不写入缓冲区 */
	if ((status & USART_SR_RXNE) == 0u)
	{
		return;
	}
	/* 计算下一个写入位置 */
	next = (uint16_t)((usart_rx_head + 1u) & USART_RX_BUFFER_MASK);
	if (next == usart_rx_tail)
	{
		/* 缓冲区已满，丢弃当前字节并标记溢出 */
		usart_rx_overflow = 1u;
		return;
	}
	usart_rx_buffer[usart_rx_head] = byte;
	usart_rx_head = next;
}

/**
  * @brief  重定向 fputc，使 printf 输出到 USART1
  * @param  c:  待发送字符
  * @param  fp: 文件指针（忽略）
  * @retval 已发送的字符
  */
int fputc(int c, FILE *fp)
{
	(void)fp;
	USART_SendData(USART1, (uint8_t)c);
	while (USART_GetFlagStatus(USART1, USART_FLAG_TXE) == RESET)
	{
	}
	return c;
}
