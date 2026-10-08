#ifndef SERVO_H
#define SERVO_H
#include <stdint.h>
void Servo_Init(void);
void Servo_Stop(void);
void Servo_Jog(int16_t speed, uint16_t duration_ms);
void Servo_Tick1ms(void);
int16_t Servo_Speed(void);
#endif
