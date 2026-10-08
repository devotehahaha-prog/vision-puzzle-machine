#ifndef SERVO_CONFIG_H
#define SERVO_CONFIG_H
/* JX PDI-6221MG-360 continuous rotation. Calibrate neutral unloaded. */
#define SERVO_PERIOD_US       3030u
#define SERVO_MIN_US          500u
#define SERVO_NEUTRAL_US      1520u
#define SERVO_MAX_US          2500u
#define SERVO_MAX_JOG_MS      1000u
#define AUX_LEASE_MS          2000u
#if SERVO_MIN_US >= SERVO_NEUTRAL_US || SERVO_NEUTRAL_US >= SERVO_MAX_US || SERVO_MAX_US >= SERVO_PERIOD_US
#error Invalid servo PWM configuration
#endif
#endif
