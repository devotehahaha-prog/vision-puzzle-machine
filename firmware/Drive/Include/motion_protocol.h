#ifndef __MOTION_PROTOCOL_H
#define __MOTION_PROTOCOL_H

/**
  * @file    motion_protocol.h
  * @brief   串口运动控制协议接口
  *
  * @description
  *   通过 USART1 接收以换行结尾的文本指令，控制 4 轴步进电机运动。
  *   支持绝对定位（GOTO）、相对运动（MOVE/Z/ROTATE）、急停（STOP）、
  *   电磁铁（MAGNET）、状态查询（STATUS/PING）等指令。
  *   具体协议格式见项目根目录 SERIAL_PROTOCOL.md。
  */

/**
  * @brief  协议状态机初始化，清零位置与所有标志位
  */
void MotionProtocol_Init(void);

/**
  * @brief  协议主循环处理：读取串口字节、解析指令、轮询运动完成
  *         需在主循环中持续调用
  */
void MotionProtocol_Process(void);

/* Called by SysTick; only advances the communication-lease timer. */
void MotionProtocol_Tick1ms(void);

#endif /* __MOTION_PROTOCOL_H */
