#include "stm32f4xx.h"
#include "servo.h"
#include "servo_config.h"

static volatile int16_t servo_speed;
static volatile uint16_t remaining_ms;
static volatile uint8_t initialized;

void Servo_Stop(void)
{
    uint32_t mask = __get_PRIMASK();
    __disable_irq();
    remaining_ms = 0;
    servo_speed = 0;
    if (initialized) TIM_SetCompare1(TIM12, SERVO_NEUTRAL_US);
    __set_PRIMASK(mask);
}

void Servo_Init(void)
{
    GPIO_InitTypeDef gpio;
    TIM_TimeBaseInitTypeDef base;
    TIM_OCInitTypeDef oc;
    RCC_ClocksTypeDef clocks;
    uint32_t timer_clock;
    RCC_AHB1PeriphClockCmd(RCC_AHB1Periph_GPIOB, ENABLE);
    RCC_APB1PeriphClockCmd(RCC_APB1Periph_TIM12, ENABLE);
    TIM_DeInit(TIM12);
    RCC_GetClocksFreq(&clocks);
    timer_clock = clocks.PCLK1_Frequency;
    if ((RCC->CFGR & RCC_CFGR_PPRE1) != 0u) timer_clock *= 2u;
    TIM_TimeBaseStructInit(&base);
    base.TIM_Prescaler = (uint16_t)(timer_clock / 1000000u - 1u);
    base.TIM_Period = SERVO_PERIOD_US - 1u;
    TIM_TimeBaseInit(TIM12, &base);
    TIM_OCStructInit(&oc);
    oc.TIM_OCMode = TIM_OCMode_PWM1;
    oc.TIM_OutputState = TIM_OutputState_Enable;
    oc.TIM_Pulse = SERVO_NEUTRAL_US;
    oc.TIM_OCPolarity = TIM_OCPolarity_High;
    TIM_OC1Init(TIM12, &oc);
    TIM_OC1PreloadConfig(TIM12, TIM_OCPreload_Enable);
    TIM_ARRPreloadConfig(TIM12, ENABLE);
    TIM_GenerateEvent(TIM12, TIM_EventSource_Update);
    GPIO_StructInit(&gpio);
    gpio.GPIO_Pin = GPIO_Pin_14;
    gpio.GPIO_Mode = GPIO_Mode_AF;
    gpio.GPIO_OType = GPIO_OType_PP;
    gpio.GPIO_Speed = GPIO_Speed_25MHz;
    gpio.GPIO_PuPd = GPIO_PuPd_DOWN;
    GPIO_PinAFConfig(GPIOB, GPIO_PinSource14, GPIO_AF_TIM12);
    GPIO_Init(GPIOB, &gpio);
    initialized = 1u;
    Servo_Stop();
    TIM_Cmd(TIM12, ENABLE);
}

void Servo_Jog(int16_t speed, uint16_t duration_ms)
{
    int32_t pulse;
    uint32_t mask;
    if (!initialized || speed < -100 || speed > 100 || duration_ms == 0 || duration_ms > SERVO_MAX_JOG_MS) return;
    pulse = (int32_t)SERVO_NEUTRAL_US + speed * (int32_t)(speed >= 0 ?
        SERVO_MAX_US - SERVO_NEUTRAL_US : SERVO_NEUTRAL_US - SERVO_MIN_US) / 100;
    mask = __get_PRIMASK();
    __disable_irq();
    servo_speed = speed;
    remaining_ms = duration_ms;
    TIM_SetCompare1(TIM12, (uint32_t)pulse);
    __set_PRIMASK(mask);
}

void Servo_Tick1ms(void)
{
    if (remaining_ms && --remaining_ms == 0) Servo_Stop();
}

int16_t Servo_Speed(void) { return servo_speed; }
