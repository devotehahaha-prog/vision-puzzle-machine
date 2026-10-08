/**
  *************************************************************************************************
  * @file    a4988.c
  * @brief   A4988 步进电机驱动实现（五电机四逻辑轴版本）
  *************************************************************************************************
  * @description
  *
  *   本文件为 A4988 步进电机驱动芯片的 STM32F4 实现层。
  *   通过 5 个定时器输出 PWM 产生步进脉冲（STEP），
  *   通过普通 GPIO 控制方向（DIR），共用使能信号 EN 控制 5 路步进。
  *   支持梯形加减速曲线（起始频率、最大频率、加减速步数）、急停等功能。
  *   同时提供多轴同步运动接口 A4988_MoveGroup。
  *
  *   硬件资源映射：
  *     X左    STEP=PA0 (TIM2_CH1)，DIR=PE0
  *     X右    STEP=PA8 (TIM1_CH1)，DIR=PE8
  *     Y      STEP=PA6 (TIM3_CH1)，DIR=PE2
  *     Z      STEP=PB6 (TIM4_CH1)，DIR=PE4
  *     ROTATE STEP=PA1 (TIM5_CH2)，DIR=PE6
  *     EN     =PE1（所有步进共用，低电平有效）
  *
  ************************************************************************************************
  */

#include "a4988.h"
#include "motion_config.h"

/* 定时器内部计数频率：1MHz。TIMx 时钟为 84MHz，预分频 83 得到 1MHz 时基 */
#define A4988_TIMER_TICK_HZ       1000000u
/* 预分频值：83，即 84MHz / (83+1) = 1MHz */
#define A4988_TIMER_PRESCALER     83u
/* 步进脉冲最低频率，过低会导致堵转 */
#define A4988_MIN_STEP_HZ         16u
/* 步进脉冲最高频率，过高会超出 A4988 响应能力 */
#define A4988_MAX_STEP_HZ         100000u
/* 默认起始频率，保守取值 */
#define A4988_DEFAULT_START_HZ    200u
/* 默认最大频率，保守取值 */
#define A4988_DEFAULT_MAX_HZ      1000u
/* 默认加减速步数，用于起始频率到最大频率之间的过渡 */
#define A4988_DEFAULT_ACCEL_STEPS 50u

/* CNC Shield 共用 EN 引脚，所有驱动芯片共用，低电平有效 */
#define A4988_ENABLE_PORT         GPIOE
#define A4988_ENABLE_CLOCK        RCC_AHB1Periph_GPIOE
#define A4988_ENABLE_PIN          GPIO_Pin_1

/**
  * @brief 单路电机硬件资源映射
  *        包含生成 STEP 脉冲所用的定时器/通道/中断号，以及 DIR 引脚。
  *        EN 为共用引脚，不为每路单独分配。
  */
typedef struct
{
	TIM_TypeDef *timer;            /* 生成 STEP 脉冲的定时器                */
	uint32_t timer_clock;         /* 定时器 RCC 时钟使能位                 */
	IRQn_Type timer_irq;          /* 定时器更新中断号                      */
	uint16_t timer_channel;       /* 使用的捕获比较通道（TIM_Channel_1/2） */
	GPIO_TypeDef *step_port;      /* STEP 信号所在 GPIO 端口              */
	uint32_t step_clock;         /* STEP 端口的 AHB1 时钟使能位          */
	uint16_t step_pin;            /* STEP 信号引脚                        */
	uint8_t step_pin_source;     /* STEP 引脚的 PinSource（AF 配置用）   */
	uint8_t step_af;              /* STEP 引脚的复用功能号                */
	GPIO_TypeDef *direction_port; /* DIR 信号所在 GPIO 端口              */
	uint32_t direction_clock;    /* DIR 端口的 AHB1 时钟使能位           */
	uint16_t direction_pin;       /* DIR 信号引脚                         */
} A4988_Hardware;

/**
  * @brief 单路电机运行状态
  *        volatile 成员会被中断改写，同时被主循环读取
  */
typedef struct
{
	volatile uint32_t target_pulses;    /* 目标脉冲数                    */
	volatile uint32_t completed_pulses;/* 已发送脉冲数                    */
	uint32_t start_hz;                 /* 起始频率                        */
	uint32_t max_hz;                   /* 最大运行频率                    */
	uint32_t accel_steps;              /* 加减速步数                        */
	volatile uint8_t busy;              /* 1=正在运动，0=空闲              */
} A4988_MotorState;

/**
  * @brief 硬件资源表（4 轴）
  *        注意：旋转轴使用 TIM5 的通道 2（PA1），其余轴使用通道 1
  */
static const A4988_Hardware a4988_hardware[A4988_MAX_MOTORS] =
{
	{
		TIM2, RCC_APB1Periph_TIM2, TIM2_IRQn, TIM_Channel_1,
		GPIOA, RCC_AHB1Periph_GPIOA, GPIO_Pin_0, GPIO_PinSource0, GPIO_AF_TIM2,
		GPIOE, RCC_AHB1Periph_GPIOE, GPIO_Pin_0
	},
	{
		TIM3, RCC_APB1Periph_TIM3, TIM3_IRQn, TIM_Channel_1,
		GPIOA, RCC_AHB1Periph_GPIOA, GPIO_Pin_6, GPIO_PinSource6, GPIO_AF_TIM3,
		GPIOE, RCC_AHB1Periph_GPIOE, GPIO_Pin_2
	},
	{
		TIM4, RCC_APB1Periph_TIM4, TIM4_IRQn, TIM_Channel_1,
		GPIOB, RCC_AHB1Periph_GPIOB, GPIO_Pin_6, GPIO_PinSource6, GPIO_AF_TIM4,
		GPIOE, RCC_AHB1Periph_GPIOE, GPIO_Pin_4
	},
	{
		TIM5, RCC_APB1Periph_TIM5, TIM5_IRQn, TIM_Channel_2,
		GPIOA, RCC_AHB1Periph_GPIOA, GPIO_Pin_1, GPIO_PinSource1, GPIO_AF_TIM5,
		GPIOE, RCC_AHB1Periph_GPIOE, GPIO_Pin_6
	},
	{
		TIM1, RCC_APB2Periph_TIM1, TIM1_UP_TIM10_IRQn, TIM_Channel_1,
		GPIOA, RCC_AHB1Periph_GPIOA, GPIO_Pin_8, GPIO_PinSource8, GPIO_AF_TIM1,
		GPIOE, RCC_AHB1Periph_GPIOE, GPIO_Pin_8
	}
};

/* 各电机状态数组，数量由 A4988_ACTIVE_MOTORS 决定 */
static A4988_MotorState a4988_state[A4988_ACTIVE_MOTORS];
static volatile uint8_t a4988_hold_enabled;

/**
  * @brief  判断电机 ID 是否在有效范围内
  * @param  id: 轴号
  * @retval 1=合法，0=越界
  */
static uint8_t A4988_IsValidId(uint8_t id)
{
	return (id < A4988_ACTIVE_MOTORS) ? 1u : 0u;
}

/**
  * @brief  设置共用 EN 引脚电平
  *         EN 低电平有效：enabled=非 0 时使能全部电机，=0 时释放全部电机
  * @param  enabled: 非 0 使能，0 释放
  */
static void A4988_SetSharedEnable(uint8_t enabled)
{
	if (enabled != 0u)
	{
		GPIO_ResetBits(A4988_ENABLE_PORT, A4988_ENABLE_PIN);
	}
	else
	{
		GPIO_SetBits(A4988_ENABLE_PORT, A4988_ENABLE_PIN);
	}
}

/**
  * @brief  查询是否有任意电机正在运动
  * @retval 1=至少一个忙碌，0=全部空闲
  */
uint8_t A4988_AnyBusy(void)
{
	uint8_t id;

	for (id = 0u; id < A4988_ACTIVE_MOTORS; id++)
	{
		if (a4988_state[id].busy != 0u)
		{
			return 1u;
		}
	}
	return 0u;
}

/**
  * @brief  根据当前忙碌状态刷新共用 EN：任一电机运动则使能，否则释放
  */
static void A4988_UpdateSharedEnable(void)
{
	A4988_SetSharedEnable((uint8_t)(a4988_hold_enabled || A4988_AnyBusy()));
}

void A4988_SetHoldEnabled(uint8_t enabled)
{
	uint32_t mask = __get_PRIMASK();
	__disable_irq();
	a4988_hold_enabled = (enabled != 0u);
	A4988_UpdateSharedEnable();
	__set_PRIMASK(mask);
}

/*==================== 定时器通道封装（兼容 CH1/CH2）====================*/

/**
  * @brief  根据硬件表中的通道号调用对应通道的 OC 初始化
  */
static void A4988_OCInit(const A4988_Hardware *hardware,
						 TIM_OCInitTypeDef *output_compare_init)
{
	if (hardware->timer_channel == TIM_Channel_1)
	{
		TIM_OC1Init(hardware->timer, output_compare_init);
	}
	else
	{
		TIM_OC2Init(hardware->timer, output_compare_init);
	}
}

/**
  * @brief  配置对应通道的 CCR 预装载使能/失能
  */
static void A4988_OCPreloadConfig(const A4988_Hardware *hardware,
								  uint16_t preload)
{
	if (hardware->timer_channel == TIM_Channel_1)
	{
		TIM_OC1PreloadConfig(hardware->timer, preload);
	}
	else
	{
		TIM_OC2PreloadConfig(hardware->timer, preload);
	}
}

/**
  * @brief  向对应通道写入比较寄存器值（决定 PWM 占空比）
  */
static void A4988_SetCompare(const A4988_Hardware *hardware, uint32_t compare)
{
	if (hardware->timer_channel == TIM_Channel_1)
	{
		TIM_SetCompare1(hardware->timer, compare);
	}
	else
	{
		TIM_SetCompare2(hardware->timer, compare);
	}
}

/**
  * @brief  将对应通道强制输出为无效电平，用于把 STEP 拉低
  */
static void A4988_ForceInactive(const A4988_Hardware *hardware)
{
	if (hardware->timer_channel == TIM_Channel_1)
	{
		TIM_ForcedOC1Config(hardware->timer, TIM_ForcedAction_InActive);
	}
	else
	{
		TIM_ForcedOC2Config(hardware->timer, TIM_ForcedAction_InActive);
	}
}

/*==================== 梯形加减速 ====================*/

/**
  * @brief  根据当前已发送脉冲数计算下一拍应使用的频率
  *
  *         梯形模型：前 accel_steps 拍从 start_hz 线性升到 max_hz（加速段），
  *         中间保持 max_hz（匀速段），最后 accel_steps 拍降回 start_hz（减速段）。
  *         若加减速步数超过目标脉冲的一半，则压缩到一半，保证加减速对称。
  *
  * @param  id:          轴号
  * @param  pulse_index: 当前已发送脉冲数（0 开始）
  * @retval 下一拍应使用的脉冲频率（Hz）
  */
static uint32_t A4988_GetPulseFrequency(uint8_t id, uint32_t pulse_index)
{
	A4988_MotorState *state = &a4988_state[id];
	uint32_t ramp_pulses = state->accel_steps;
	uint32_t half_pulses = state->target_pulses / 2u;
	uint32_t scale_index;
	uint32_t frequency_span;

	/* 若加减速步数超过总步数一半，则压缩到一半，保证加减速对称 */
	if (ramp_pulses > half_pulses)
	{
		ramp_pulses = half_pulses;
	}
	/* 加减速段不足一拍或起止频率相同，则全程使用起始频率 */
	if ((ramp_pulses <= 1u) || (state->start_hz == state->max_hz))
	{
		return state->start_hz;
	}

	/* 判断当前处于加速段 / 减速段 / 匀速段，得到斜坡内的索引值 */
	if (pulse_index < ramp_pulses)
	{
		scale_index = pulse_index;
	}
	else if (pulse_index >= (state->target_pulses - ramp_pulses))
	{
		scale_index = state->target_pulses - pulse_index - 1u;
	}
	else
	{
		scale_index = ramp_pulses - 1u;
	}

	/* 在 start_hz 到 max_hz 之间做线性插值 */
	frequency_span = state->max_hz - state->start_hz;
	return state->start_hz +
		   (frequency_span * scale_index) / (state->accel_steps - 1u);
}

/**
  * @brief  将指定频率写入定时器 ARR/CCR，PWM 占空比固定 50%
  *
  *         定时器为 1MHz，因此 ARR = 1000000 / frequency - 1，
  *         比较值取周期一半。ARR 预装载已关闭，频率修改立即生效。
  *
  * @param  id:        轴号
  * @param  frequency: 目标脉冲频率（Hz）
  */
static void A4988_ApplyFrequency(uint8_t id, uint32_t frequency)
{
	const A4988_Hardware *hardware = &a4988_hardware[id];
	uint32_t period = A4988_TIMER_TICK_HZ / frequency;

	TIM_SetAutoreload(hardware->timer, period - 1u);
	A4988_SetCompare(hardware, period / 2u);
}

/**
  * @brief  强制 STEP 为低并停止该轴定时器
  *
  *         A4988 的 STEP 上升沿触发一次运动，停止时须保证 STEP 为低电平，
  *         否则下次使能时可能被误判为一个脉冲。
  *
  * @param  id: 轴号
  */
static void A4988_ForceStepLow(uint8_t id)
{
	const A4988_Hardware *hardware = &a4988_hardware[id];

	A4988_ForceInactive(hardware);
	TIM_Cmd(hardware->timer, DISABLE);
	TIM_SetCounter(hardware->timer, 0u);
	TIM_ClearITPendingBit(hardware->timer, TIM_IT_Update);
}

/**
  * @brief  一次运动结束的收尾：强制 STEP 低、清忙碌标志、刷新共用 EN
  * @param  id: 轴号
  */
static void A4988_FinishMove(uint8_t id)
{
	A4988_ForceStepLow(id);
	a4988_state[id].busy = 0u;
	A4988_UpdateSharedEnable();
}

/*==================== 初始化 ====================*/

/**
  * @brief  初始化指定轴的全部硬件资源
  *
  *         1. 使能 STEP/DIR 端口与定时器时钟
  *         2. DIR 引脚置低（默认方向），配置为推挽输出
  *         3. STEP 引脚配置为复用功能，映射到对应 TIMx 通道
  *         4. 定时器时基：预分频 83，1MHz 计数，向上计数
  *         5. 对应通道配置为 PWM2 模式，占空比 50%
  *         6. 开启更新中断并配置 NVIC（抢占优先级 1）
  *         7. 状态初始化为空闲，使用默认加减速参数
  *
  * @param  id: 轴号
  */
static void A4988_InitMotor(uint8_t id)
{
	const A4988_Hardware *hardware = &a4988_hardware[id];
	GPIO_InitTypeDef gpio_init;
	TIM_TimeBaseInitTypeDef timer_init;
	TIM_OCInitTypeDef output_compare_init;
	NVIC_InitTypeDef nvic_init;

	/* 1. 使能 GPIO 与定时器时钟（EN 引脚时钟在 A4988_Init 中统一开启） */
	RCC_AHB1PeriphClockCmd(hardware->step_clock | hardware->direction_clock, ENABLE);
	if (hardware->timer == TIM1)
	{
		RCC_APB2PeriphClockCmd(hardware->timer_clock, ENABLE);
	}
	else
	{
		RCC_APB1PeriphClockCmd(hardware->timer_clock, ENABLE);
	}

	/* 2. DIR 默认置低（正方向），配置为推挽输出 */
	GPIO_ResetBits(hardware->direction_port, hardware->direction_pin);
	GPIO_StructInit(&gpio_init);
	gpio_init.GPIO_Pin = hardware->direction_pin;
	gpio_init.GPIO_Mode = GPIO_Mode_OUT;
	gpio_init.GPIO_OType = GPIO_OType_PP;
	gpio_init.GPIO_PuPd = GPIO_PuPd_NOPULL;
	gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(hardware->direction_port, &gpio_init);

	/* 3. STEP 引脚配置为复用功能，映射到对应的 TIMx */
	gpio_init.GPIO_Pin = hardware->step_pin;
	gpio_init.GPIO_Mode = GPIO_Mode_AF;
	gpio_init.GPIO_PuPd = GPIO_PuPd_DOWN;
	GPIO_Init(hardware->step_port, &gpio_init);
	GPIO_PinAFConfig(hardware->step_port, hardware->step_pin_source,
					hardware->step_af);

	/* 4. 定时器时基：1MHz 计数，向上计数，初始周期=起始频率对应周期 */
	TIM_TimeBaseStructInit(&timer_init);
	timer_init.TIM_Prescaler = A4988_TIMER_PRESCALER;
	timer_init.TIM_Period = (A4988_TIMER_TICK_HZ / A4988_DEFAULT_START_HZ) - 1u;
	timer_init.TIM_CounterMode = TIM_CounterMode_Up;
	timer_init.TIM_ClockDivision = TIM_CKD_DIV1;
	TIM_TimeBaseInit(hardware->timer, &timer_init);
	/* ARR 不预装载：频率修改立即生效 */
	TIM_ARRPreloadConfig(hardware->timer, DISABLE);

	/* 5. 对应通道 PWM2 模式（CNT<CCR 时输出有效电平），50% 占空比 */
	TIM_OCStructInit(&output_compare_init);
	output_compare_init.TIM_OCMode = TIM_OCMode_PWM2;
	output_compare_init.TIM_OutputState = TIM_OutputState_Enable;
	output_compare_init.TIM_Pulse =
		(A4988_TIMER_TICK_HZ / A4988_DEFAULT_START_HZ) / 2u;
	output_compare_init.TIM_OCPolarity = TIM_OCPolarity_High;
	A4988_OCInit(hardware, &output_compare_init);
	if (hardware->timer == TIM1)
	{
		TIM_CtrlPWMOutputs(TIM1, ENABLE);
	}
	/* CCR 不预装载：占空比修改立即生效 */
	A4988_OCPreloadConfig(hardware, TIM_OCPreload_Disable);

	/* 6. 开启更新中断、清标志、定时器暂不启动 */
	TIM_ITConfig(hardware->timer, TIM_IT_Update, ENABLE);
	TIM_ClearITPendingBit(hardware->timer, TIM_IT_Update);
	TIM_Cmd(hardware->timer, DISABLE);

	/* 7. NVIC 配置（抢占优先级 1，子优先级 0），4 个定时器同步时无嵌套 */
	nvic_init.NVIC_IRQChannel = hardware->timer_irq;
	nvic_init.NVIC_IRQChannelPreemptionPriority = 1u;
	nvic_init.NVIC_IRQChannelSubPriority = 0u;
	nvic_init.NVIC_IRQChannelCmd = ENABLE;
	NVIC_Init(&nvic_init);

	/* 8. 电机状态初始化为空闲，使用默认加减速参数 */
	a4988_state[id].target_pulses = 0u;
	a4988_state[id].completed_pulses = 0u;
	a4988_state[id].start_hz = A4988_DEFAULT_START_HZ;
	a4988_state[id].max_hz = A4988_DEFAULT_MAX_HZ;
	a4988_state[id].accel_steps = A4988_DEFAULT_ACCEL_STEPS;
	a4988_state[id].busy = 0u;
}

/**
  * @brief  A4988 驱动总初始化
  *         初始化 4 路步进与共用 EN 引脚（默认释放），再逐个初始化各轴，
  *         最后确保 EN 处于释放状态（高电平）。
  */
void A4988_Init(void)
{
	GPIO_InitTypeDef gpio_init;
	uint8_t id;
	a4988_hold_enabled = 0u;

	/* 配置共用 EN 引脚：推挽输出、上拉，默认高电平（释放全部电机） */
	RCC_AHB1PeriphClockCmd(A4988_ENABLE_CLOCK, ENABLE);
	GPIO_SetBits(A4988_ENABLE_PORT, A4988_ENABLE_PIN);
	GPIO_StructInit(&gpio_init);
	gpio_init.GPIO_Pin = A4988_ENABLE_PIN;
	gpio_init.GPIO_Mode = GPIO_Mode_OUT;
	gpio_init.GPIO_OType = GPIO_OType_PP;
	gpio_init.GPIO_PuPd = GPIO_PuPd_UP;
	gpio_init.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(A4988_ENABLE_PORT, &gpio_init);

	/* 逐轴初始化 STEP/DIR/定时器/中断 */
	for (id = 0u; id < A4988_ACTIVE_MOTORS; id++)
	{
		A4988_InitMotor(id);
	}
	/* 确保上电后处于释放（高阻）状态 */
	A4988_SetSharedEnable(0u);
}

/**
  * @brief  设置指定轴的加减速曲线参数
  * @param  id:          轴号
  * @param  start_hz:    起始频率（不低于 A4988_MIN_STEP_HZ）
  * @param  max_hz:      最大运行频率（不高于 A4988_MAX_STEP_HZ）
  * @param  accel_steps: 加减速步数（大于 0）
  * @retval A4988_OK=成功；A4988_ERROR_ID=ID 非法；A4988_ERROR_BUSY=运动中；
  *         A4988_ERROR_PARAM=参数不合法
  */
A4988_Status A4988_SetProfile(uint8_t id, uint32_t start_hz,
									 uint32_t max_hz, uint32_t accel_steps)
{
	if (A4988_IsValidId(id) == 0u)
	{
		return A4988_ERROR_ID;
	}
	/* 运动中不允许修改参数，直接拒绝 */
	if (a4988_state[id].busy != 0u)
	{
		return A4988_ERROR_BUSY;
	}
	/* 参数合法区间校验 */
	if ((start_hz < A4988_MIN_STEP_HZ) || (max_hz == 0u) ||
		(start_hz > max_hz) || (max_hz > A4988_MAX_STEP_HZ) ||
		(accel_steps == 0u))
	{
		return A4988_ERROR_PARAM;
	}

	a4988_state[id].start_hz = start_hz;
	a4988_state[id].max_hz = max_hz;
	a4988_state[id].accel_steps = accel_steps;
	return A4988_OK;
}

/*==================== 运动控制 ====================*/

/**
  * @brief  准备一次单轴运动指令（只配置不启动，不使能 EN）
  *
  *         设置状态为忙碌、写入 DIR、恢复通道 PWM2 模式、写入第 0 拍频率。
  *         设计为多轴准备接口，供 A4988_MoveGroup 准备完全部轴后，
  *         再统一使能 EN、统一启动定时器，保证多轴同步起转。
  *
  * @param  command: 运动指令指针
  */
static void A4988_PrepareMove(const A4988_MoveCommand *command)
{
	uint8_t id = command->id;
	const A4988_Hardware *hardware = &a4988_hardware[id];

	a4988_state[id].target_pulses = command->pulses;
	a4988_state[id].completed_pulses = 0u;
	a4988_state[id].busy = 1u;

	/* 设置方向：反转拉高 DIR，正转拉低 DIR */
	if (command->direction == A4988_DIR_REVERSE)
	{
		GPIO_SetBits(hardware->direction_port, hardware->direction_pin);
	}
	else
	{
		GPIO_ResetBits(hardware->direction_port, hardware->direction_pin);
	}

	/* ForceStepLow 曾把通道置为强制模式，此处恢复为 PWM2 并使能通道 */
	TIM_SelectOCxM(hardware->timer, hardware->timer_channel, TIM_OCMode_PWM2);
	TIM_CCxCmd(hardware->timer, hardware->timer_channel, TIM_CCx_Enable);
	/* 写入第 0 拍频率（起始频率），清计数器与中断标志，等待统一启动 */
	A4988_ApplyFrequency(id, A4988_GetPulseFrequency(id, 0u));
	TIM_SetCounter(hardware->timer, 0u);
	TIM_ClearITPendingBit(hardware->timer, TIM_IT_Update);
}

/**
  * @brief  多轴同步运动
  *
  *         执行三阶段：
  *           1. 校验全部指令（空指针、数量、ID、忙碌状态、脉冲/方向、重复轴）；
  *              任一非法则整体拒绝，不启动任何轴。
  *           2. 对每条指令调用 PrepareMove（只配置，不启动）。
  *           3. 统一释放共用 EN 使能全部电机，再统一启动各轴定时器，
  *              使各轴几乎同时输出第一个脉冲。
  *
  * @param  commands: 指令数组
  * @param  count:    指令数量（1~A4988_ACTIVE_MOTORS）
  * @retval A4988_Status
  */
A4988_Status A4988_MoveGroup(const A4988_MoveCommand *commands, uint8_t count)
{
	uint8_t index;
	uint8_t previous;
	uint8_t expanded_count = 0u;
	uint8_t has_x_left = 0u;
	uint8_t has_x_right = 0u;
	A4988_MoveCommand expanded[A4988_MAX_MOTORS];

	/* 阶段 1：参数合法性校验 */
	if ((commands == 0) || (count == 0u) || (count > A4988_ACTIVE_MOTORS))
	{
		return A4988_ERROR_PARAM;
	}

	for (index = 0u; index < count; index++)
	{
		if (A4988_IsValidId(commands[index].id) == 0u)
		{
			return A4988_ERROR_ID;
		}
		if (a4988_state[commands[index].id].busy != 0u)
		{
			return A4988_ERROR_BUSY;
		}
		if ((commands[index].pulses == 0u) ||
			((commands[index].direction != A4988_DIR_FORWARD) &&
			 (commands[index].direction != A4988_DIR_REVERSE)))
		{
			return A4988_ERROR_PARAM;
		}
		/* 同一批指令中轴号不得重复 */
		for (previous = 0u; previous < index; previous++)
		{
			if (commands[previous].id == commands[index].id)
			{
				return A4988_ERROR_PARAM;
			}
		}
		if (commands[index].id == A4988_AXIS_X_LEFT) has_x_left = 1u;
		if (commands[index].id == A4988_AXIS_X_RIGHT) has_x_right = 1u;
	}
	if ((has_x_right != 0u) && (has_x_left == 0u))
	{
		return A4988_ERROR_PARAM;
	}
	for (index = 0u; index < count; index++)
	{
		expanded[expanded_count++] = commands[index];
		if ((commands[index].id == A4988_AXIS_X_LEFT) && (has_x_right == 0u))
		{
			expanded[expanded_count].id = A4988_AXIS_X_RIGHT;
			expanded[expanded_count].direction =
				(A4988_Direction)(commands[index].direction ^ MOTION_X_RIGHT_DIRECTION_INVERT);
			expanded[expanded_count].pulses = commands[index].pulses;
			expanded_count++;
		}
	}
	for (index = 0u; index < expanded_count; index++)
	{
		if (a4988_state[expanded[index].id].busy != 0u)
		{
			return A4988_ERROR_BUSY;
		}
	}

	/* 阶段 2：逐轴准备（DIR/PWM/频率），置忙碌但定时器不启动 */
	for (index = 0u; index < expanded_count; index++)
	{
		A4988_PrepareMove(&expanded[index]);
	}
	/* 阶段 3：先使能共用 EN，再统一启动各轴定时器，保证同步起转 */
	A4988_SetSharedEnable(1u);
	for (index = 0u; index < expanded_count; index++)
	{
		TIM_Cmd(a4988_hardware[expanded[index].id].timer, ENABLE);
	}

	return A4988_OK;
}

/**
  * @brief  单轴运动（将单条指令包装后调用 A4988_MoveGroup）
  * @param  id:        轴号
  * @param  direction: 方向
  * @param  pulses:    目标脉冲数（大于 0）
  * @retval A4988_Status
  */
A4988_Status A4988_Move(uint8_t id, A4988_Direction direction,
								 uint32_t pulses)
{
	A4988_MoveCommand command;

	command.id = id;
	command.direction = direction;
	command.pulses = pulses;
	return A4988_MoveGroup(&command, 1u);
}

A4988_Status A4988_MoveXPair(A4988_Direction direction, uint32_t pulses)
{
	A4988_MoveCommand commands[2];
	if (pulses == 0u)
	{
		return A4988_ERROR_PARAM;
	}
	commands[0].id = A4988_AXIS_X_LEFT;
	commands[0].direction = direction;
	commands[0].pulses = pulses;
	commands[1].id = A4988_AXIS_X_RIGHT;
	commands[1].direction = (A4988_Direction)(direction ^ MOTION_X_RIGHT_DIRECTION_INVERT);
	commands[1].pulses = pulses;
	return A4988_MoveGroup(commands, 2u);
}

/**
  * @brief  立即停止指定电机（非减速停）
  *         停止后根据全局忙碌状态刷新共用 EN，全部停完时释放电机
  * @param  id: 轴号
  */
void A4988_Stop(uint8_t id)
{
	if (id == A4988_AXIS_X_LEFT)
	{
		A4988_ForceStepLow(A4988_AXIS_X_LEFT);
		A4988_ForceStepLow(A4988_AXIS_X_RIGHT);
		a4988_state[A4988_AXIS_X_LEFT].busy = 0u;
		a4988_state[A4988_AXIS_X_RIGHT].busy = 0u;
		a4988_state[A4988_AXIS_X_LEFT].target_pulses = 0u;
		a4988_state[A4988_AXIS_X_RIGHT].target_pulses = 0u;
		A4988_UpdateSharedEnable();
		return;
	}
	if (A4988_IsValidId(id) == 0u)
	{
		return;
	}

	A4988_ForceStepLow(id);
	a4988_state[id].busy = 0u;
	a4988_state[id].target_pulses = 0u;
	A4988_UpdateSharedEnable();
}

/**
  * @brief  立即停止全部电机并强制释放共用 EN（全部断电）
  */
void A4988_StopAll(void)
{
	uint8_t id;
	uint32_t mask = __get_PRIMASK();
	__disable_irq();
	a4988_hold_enabled = 0u;

	for (id = 0u; id < A4988_ACTIVE_MOTORS; id++)
	{
		A4988_ForceStepLow(id);
		a4988_state[id].busy = 0u;
		a4988_state[id].target_pulses = 0u;
	}
	A4988_SetSharedEnable(0u);
	__set_PRIMASK(mask);
}

/**
  * @brief  查询指定轴是否正在运动
  * @param  id: 轴号
  * @retval 1=忙碌，0=空闲（ID 非法也返回 0）
  */
uint8_t A4988_IsBusy(uint8_t id)
{
	if (A4988_IsValidId(id) == 0u)
	{
		return 0u;
	}
	return a4988_state[id].busy;
}

/**
  * @brief  查询指定轴已发送的脉冲数
  * @param  id: 轴号
  * @retval 已发送脉冲数（ID 非法返回 0）
  */
uint32_t A4988_GetCompletedPulses(uint8_t id)
{
	if (A4988_IsValidId(id) == 0u)
	{
		return 0u;
	}
	return a4988_state[id].completed_pulses;
}

/**
  * @brief  定时器更新中断处理函数，由 TIM2/3/4/5 的 IRQHandler 调用
  *
  *         每次更新中断代表发出一个 STEP 脉冲，处理流程：
  *           1. 清更新中断标志
  *           2. 已发送脉冲数 +1
  *           3. 达到目标脉冲数 -> FinishMove 收尾停止并刷新共用 EN
  *           4. 否则按当前已发脉冲数计算频率并写入 ARR/CCR，实现梯形加减速
  *
  * @param  id: 轴号
  */
void A4988_TimerIRQHandler(uint8_t id)
{
	TIM_TypeDef *timer;
	A4988_MotorState *state;

	if (A4988_IsValidId(id) == 0u)
	{
		return;
	}

	timer = a4988_hardware[id].timer;
	/* 非本轴中断直接返回，避免误触发 */
	if (TIM_GetITStatus(timer, TIM_IT_Update) == RESET)
	{
		return;
	}
	TIM_ClearITPendingBit(timer, TIM_IT_Update);

	state = &a4988_state[id];
	/* 已被 Stop/StopAll 清忙碌则直接返回 */
	if (state->busy == 0u)
	{
		return;
	}

	/* 累计一个脉冲 */
	state->completed_pulses++;
	if (state->completed_pulses >= state->target_pulses)
	{
		/* 达到目标脉冲数，收尾停止 */
		A4988_FinishMove(id);
		return;
	}

	/* 根据当前已发脉冲数计算下一拍频率，实现梯形加减速 */
	A4988_ApplyFrequency(id,
		A4988_GetPulseFrequency(id, state->completed_pulses));
}
