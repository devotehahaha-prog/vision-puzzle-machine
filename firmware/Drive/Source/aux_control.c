#include "stm32f4xx.h"
#include "aux_control.h"
#include "servo.h"
#include "servo_config.h"
#include "relay.h"
#include "usart.h"
#include <stdio.h>
#include <string.h>

static volatile uint8_t armed;
static volatile uint16_t lease_ms;

void Aux_Stop(void)
{
    uint32_t mask = __get_PRIMASK();
    __disable_irq();
    armed = 0u;
    lease_ms = 0u;
    Servo_Stop();
    Relay_Set(0u);
    __set_PRIMASK(mask);
}

void Aux_Tick1ms(void)
{
    Servo_Tick1ms();
    if (lease_ms && --lease_ms == 0) Aux_Stop();
}

/* Strict decimal integers, bounded before multiplication; no atoi overflow. */
static uint8_t ReadInt(const char **cursor, int32_t *value, char terminator)
{
    const char *p = *cursor;
    int32_t n = 0, sign = 1;
    if (*p == '-') { sign = -1; p++; }
    if (*p < '0' || *p > '9') return 0u;
    while (*p >= '0' && *p <= '9') {
        if (n > 10000) return 0u;
        n = n * 10 + (*p++ - '0');
    }
    if (*p != terminator) return 0u;
    *value = n * sign;
    *cursor = p + (terminator != '\0');
    return 1u;
}

uint8_t Aux_HandleLine(const char *line)
{
    const char *p;
    int32_t speed, duration;
    uint32_t mask;
    uint8_t enabled, magnet;
    int16_t current_speed;
    if (strcmp(line, "IOSTATUS") == 0) {
        mask = __get_PRIMASK(); __disable_irq();
        if (armed) lease_ms = AUX_LEASE_MS;
        enabled = armed; current_speed = Servo_Speed(); magnet = Relay_IsEnergized();
        __set_PRIMASK(mask);
        printf("ACK,IOSTATUS,AUX,%u,SERVO,%d,MAGNET,%u\n", enabled, current_speed, magnet);
        return 1u;
    }
    if (strcmp(line, "AUX,0") == 0 || strcmp(line, "SERVO,STOP") == 0 || strcmp(line, "MAGNET,0") == 0) {
        if (line[0] == 'A') Aux_Stop();
        else if (line[0] == 'S') Servo_Stop();
        else Relay_Set(0u);
        printf("ACK,%s\n", line);
        return 1u;
    }
    if (strcmp(line, "AUX,1") == 0) {
        mask = __get_PRIMASK(); __disable_irq();
        armed = 1u; lease_ms = AUX_LEASE_MS;
        __set_PRIMASK(mask);
        Usart_SendString("ACK,AUX,1\n"); return 1u;
    }
    if (strcmp(line, "MAGNET,1") == 0) {
        mask = __get_PRIMASK(); __disable_irq();
        enabled = armed;
        if (enabled) { lease_ms = AUX_LEASE_MS; Relay_Set(1u); }
        __set_PRIMASK(mask);
        Usart_SendString(enabled ? "ACK,MAGNET,1\n" : "ERR,ARM\n"); return 1u;
    }
    if (strncmp(line, "SERVO,JOG,", 10) == 0) {
        p = line + 10;
        if (!ReadInt(&p, &speed, ',') || !ReadInt(&p, &duration, '\0')) {
            Usart_SendString("ERR,PARAM\n"); return 1u;
        }
        if (speed < -100 || speed > 100 || duration < 1 || (uint32_t)duration > SERVO_MAX_JOG_MS) {
            Usart_SendString("ERR,RANGE\n"); return 1u;
        }
        mask = __get_PRIMASK(); __disable_irq();
        enabled = armed;
        if (enabled) { lease_ms = AUX_LEASE_MS; Servo_Jog((int16_t)speed, (uint16_t)duration); }
        __set_PRIMASK(mask);
        if (enabled) printf("ACK,SERVO,JOG,%ld,%ld\n", (long)speed, (long)duration);
        else Usart_SendString("ERR,ARM\n");
        return 1u;
    }
    if (strncmp(line,"AUX",3)==0 || strncmp(line,"SERVO",5)==0 || strncmp(line,"MAGNET",6)==0 || strncmp(line,"IOSTATUS",8)==0) {
        Usart_SendString("ERR,PARAM\n"); return 1u;
    }
    return 0u;
}
