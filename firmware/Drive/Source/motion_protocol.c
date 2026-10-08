/**
  *************************************************************************************************
  * @file    motion_protocol.c
  * @brief   串口运动控制协议实现
  *************************************************************************************************
  * @description
  *
  *   通过 USART1 接收以换行结尾的文本指令，解析并控制 4 轴步进电机（X/Y/Z/R）运动。
  *   支持两类指令：
  *     - 原生指令：PING、STATUS、STOP、MOTION、ZERO、GOTO（4 轴绝对定位）
  *     - 兼容旧版指令：MOVE（XY 相对移动）、Z（上下）、ROTATE（旋转）、
  *                    MAGNET（电磁铁）、PLAN/PIECE/ENDPIECE/ENDPLAN（加工计划）
  *
  *   坐标体系：X/Y/Z 单位为 0.01 mm，R 单位为 0.01 度。
  *   位置追踪：motion_position_x100[] 为当前坐标，motion_pulse_position[] 为对应脉冲数。
  *   运动模式：当 motion_enabled=0 时为仿真模式（只更新坐标不驱动电机）。
  *   异步机制：指令启动运动后置 motion_pending，主循环通过 PollMove 等待电机停转后回包。
  *
  ************************************************************************************************
  */

#include "motion_protocol.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "a4988.h"
#include "motion_config.h"
#include "relay.h"
#include "aux_control.h"
#include "usart.h"

/* 单条指令行最大长度（不含结束符） */
#define MOTION_LINE_MAX 96u
/* 单个字段最大长度 */
#define MOTION_FIELD_MAX 24u
/* 轴数量：X/Y/Z/R */
#define MOTION_AXIS_COUNT 4u
/* 脉冲位置绝对值上限，防止换算溢出 */
#define MOTION_MAX_ABS_PULSE_POSITION 2147483647LL

/**
  * @brief 单轴标定配置
  *        由 motion_config.h 中的宏填充
  */
typedef struct
{
	uint32_t pulses_num;           /* 脉冲数分子（标定系数） */
	uint32_t units_x100_den;       /* 0.01 单位分母（标定系数） */
	uint8_t positive_direction;    /* 坐标增大对应的电机方向 */
	uint8_t limit_enable;          /* 是否启用行程限位 */
	int32_t limit_min_x100;        /* 行程下限（0.01 单位） */
	int32_t limit_max_x100;        /* 行程上限（0.01 单位） */
} MotionAxisConfig;

/**
  * @brief 数值解析结果
  */
typedef enum
{
	PARSE_INVALID = 0, /* 格式非法 */
	PARSE_OK,          /* 解析成功 */
	PARSE_RANGE        /* 数值超出范围 */
} ParseResult;

/* 各轴标定配置表，从 motion_config.h 读取 */
static const MotionAxisConfig motion_axis_config[MOTION_AXIS_COUNT] =
{
	{MOTION_X_PULSES_NUM, MOTION_X_UNITS_X100_DEN,
	 MOTION_X_POSITIVE_DIRECTION, MOTION_X_LIMIT_ENABLE,
	 MOTION_X_LIMIT_MIN_X100, MOTION_X_LIMIT_MAX_X100},
	{MOTION_Y_PULSES_NUM, MOTION_Y_UNITS_X100_DEN,
	 MOTION_Y_POSITIVE_DIRECTION, MOTION_Y_LIMIT_ENABLE,
	 MOTION_Y_LIMIT_MIN_X100, MOTION_Y_LIMIT_MAX_X100},
	{MOTION_Z_PULSES_NUM, MOTION_Z_UNITS_X100_DEN,
	 MOTION_Z_POSITIVE_DIRECTION, MOTION_Z_LIMIT_ENABLE,
	 MOTION_Z_LIMIT_MIN_X100, MOTION_Z_LIMIT_MAX_X100},
	{MOTION_R_PULSES_NUM, MOTION_R_UNITS_X100_DEN,
	 MOTION_R_POSITIVE_DIRECTION, MOTION_R_LIMIT_ENABLE,
	 MOTION_R_LIMIT_MIN_X100, MOTION_R_LIMIT_MAX_X100}
};

/* 指令行缓冲与长度 */
static char motion_line[MOTION_LINE_MAX + 1u];
static uint16_t motion_line_length;
/* 是否丢弃当前行（溢出或超长时置 1） */
static uint8_t motion_discard_line;
/* 运动使能：0=仿真模式，1=真实驱动电机 */
static uint8_t motion_enabled;
static volatile uint16_t motion_lease_ms;
static volatile uint8_t motion_lease_expired;
/* 是否已回零（ZERO 指令后置 1） */
static uint8_t motion_zeroed;
/* 是否有待完成的运动（启动后置 1，完成回包后清 0） */
static uint8_t motion_pending;

/*------ 旧版加工计划状态 ------*/
static uint8_t legacy_plan_active;   /* 计划进行中 */
static uint8_t legacy_plan_total;    /* 计划总件数 */
static uint8_t legacy_piece_active;  /* 当前工件加工中 */
static uint8_t legacy_piece_id;      /* 当前工件编号 */
static uint8_t legacy_piece_done;    /* 已完成工件数 */

/*------ 当前待完成运动的上下文 ------*/
static uint8_t pending_kind;   /* 待完成运动类型，见下方枚举 */
static uint8_t pending_axis;   /* 待完成运动的轴（单轴运动时） */
static uint8_t legacy_axis;    /* 旧版 MOVE 的轴 */
static int32_t pending_target[MOTION_AXIS_COUNT];    /* 本次运动的目标坐标 */
static int32_t legacy_xy_target[2];                   /* 旧版 GOTO 的 XY 目标 */
static int32_t pending_delta;                         /* 本次相对运动的增量 */
static char pending_z_text[5];                        /* 旧版 Z 指令的参数文本 */

/*------ 位置追踪 ------*/
static int32_t motion_position_x100[MOTION_AXIS_COUNT];      /* 当前坐标（0.01 单位） */
static int64_t motion_pulse_position[MOTION_AXIS_COUNT];     /* 当前脉冲位置 */
static int64_t motion_pending_pulse_position[MOTION_AXIS_COUNT]; /* 运动完成后的脉冲位置 */

/**
  * @brief 待完成运动类型枚举
  */
enum
{
	PENDING_NATIVE = 1u,        /* 原生 GOTO（4 轴同步） */
	PENDING_LEGACY_GOTO_X,      /* 旧版 GOTO 的 X 轴段 */
	PENDING_LEGACY_GOTO_Y,      /* 旧版 GOTO 的 Y 轴段 */
	PENDING_LEGACY_MOVE,        /* 旧版 MOVE（XY 相对） */
	PENDING_LEGACY_Z,           /* 旧版 Z（上下） */
	PENDING_LEGACY_ROTATE       /* 旧版 ROTATE */
};

/* 前向声明 */
static void MotionProtocol_ReplyLegacyMove(void);

/**
  * @brief  清零全部坐标与脉冲位置
  */
static void MotionProtocol_ClearPosition(void)
{
	uint8_t axis;

	for (axis = 0u; axis < MOTION_AXIS_COUNT; axis++)
	{
		motion_position_x100[axis] = 0;
		motion_pulse_position[axis] = 0;
		motion_pending_pulse_position[axis] = 0;
	}
}

void MotionProtocol_Tick1ms(void)
{
	/* No serial I/O in SysTick. Main-loop expiry handling runs before ACKs. */
	if (motion_lease_ms != 0u && --motion_lease_ms == 0u)
		motion_lease_expired = 1u;
}

static void MotionProtocol_Disarm(void)
{
	uint32_t mask = __get_PRIMASK();
	__disable_irq();
	motion_lease_ms = 0u;
	motion_lease_expired = 0u;
	A4988_StopAll();
	Aux_Stop();
	motion_enabled = 0u;
	motion_pending = 0u;
	pending_kind = 0u;
	motion_zeroed = 0u;
	legacy_plan_active = 0u;
	legacy_piece_active = 0u;
	MotionProtocol_ClearPosition();
	__set_PRIMASK(mask);
}

static uint8_t MotionProtocol_CheckLease(void)
{
	uint8_t byte;
	if (motion_lease_expired == 0u) return 0u;
	MotionProtocol_Disarm();
	/* Discard commands queued before loss of the session. Never ACK a
	 * partially executed rotation as complete, or auto-rearm stale input. */
	while (Usart_ReadByte(&byte) != 0u) { }
	motion_line_length = 0u;
	motion_discard_line = 0u;
	Usart_SendString("ERR,MOTION_LEASE\n");
	return 1u;
}

static void MotionProtocol_RefreshLease(void)
{
	uint32_t mask = __get_PRIMASK();
	__disable_irq();
	if (motion_enabled != 0u && motion_lease_expired == 0u)
		motion_lease_ms = MOTION_HOLD_LEASE_MS;
	__set_PRIMASK(mask);
}

/**
  * @brief  统计一行中以逗号分隔的字段数量
  * @param  line: 指令行
  * @retval 字段数量（至少为 1）
  */
static uint8_t MotionProtocol_FieldCount(const char *line)
{
	uint8_t count = 1u;

	while (*line != '\0')
	{
		if (*line == ',')
		{
			count++;
		}
		line++;
	}
	return count;
}

/**
  * @brief  从指令行中提取指定索引的字段
  * @param  line:       指令行
  * @param  index:      字段索引（0 开始）
  * @param  field:      输出缓冲区
  * @param  field_size: 输出缓冲区大小
  * @retval 1=成功，0=索引越界或字段为空/超长
  */
static uint8_t MotionProtocol_GetField(const char *line, uint8_t index,
									  char *field, uint16_t field_size)
{
	uint8_t current = 0u;
	uint16_t length = 0u;

	/* 跳过前面的字段 */
	while ((current < index) && (*line != '\0'))
	{
		if (*line == ',')
		{
			current++;
		}
		line++;
	}
	if ((current != index) || (*line == '\0'))
	{
		return 0u;
	}
	/* 统计当前字段长度 */
	while ((line[length] != '\0') && (line[length] != ','))
	{
		length++;
	}
	if ((length == 0u) || (length >= field_size))
	{
		return 0u;
	}
	memcpy(field, line, length);
	field[length] = '\0';
	return 1u;
}

/**
  * @brief  判断指令行的命令字是否匹配
  * @param  line:    指令行
  * @param  command: 期望的命令字
  * @retval 1=匹配，0=不匹配
  */
static uint8_t MotionProtocol_CommandIs(const char *line, const char *command)
{
	uint16_t length = 0u;

	while ((line[length] != '\0') && (line[length] != ','))
	{
		length++;
	}
	return (uint8_t)((strlen(command) == length) &&
						 (strncmp(line, command, length) == 0));
}

/**
  * @brief  将十进制文本解析为 0.01 单位的整数（支持正负号与两位小数）
  * @param  text:  文本
  * @param  value: 输出值（0.01 单位）
  * @retval PARSE_OK / PARSE_INVALID / PARSE_RANGE
  */
static ParseResult MotionProtocol_ParseX100(const char *text, int32_t *value)
{
	int64_t integer = 0;
	int64_t fraction = 0;
	int64_t result;
	int32_t sign = 1;
	uint8_t integer_digits = 0u;
	uint8_t fraction_digits = 0u;

	/* 处理符号 */
	if ((*text == '+') || (*text == '-'))
	{
		if (*text == '-')
		{
			sign = -1;
		}
		text++;
	}
	/* 整数部分 */
	while ((*text >= '0') && (*text <= '9'))
	{
		integer = integer * 10 + (int64_t)(*text - '0');
		if (integer > 21474836LL)
		{
			return PARSE_RANGE;
		}
		integer_digits++;
		text++;
	}
	/* 小数部分（最多两位） */
	if (*text == '.')
	{
		text++;
		while ((*text >= '0') && (*text <= '9'))
		{
			if (fraction_digits >= 2u)
			{
				return PARSE_INVALID;
			}
			fraction = fraction * 10 + (int64_t)(*text - '0');
			fraction_digits++;
			text++;
		}
	}
	/* 必须有数字且无多余字符 */
	if (((integer_digits == 0u) && (fraction_digits == 0u)) || (*text != '\0'))
	{
		return PARSE_INVALID;
	}
	/* 小数位不足两位补零 */
	if (fraction_digits == 0u)
	{
		fraction = 0;
	}
	else if (fraction_digits == 1u)
	{
		fraction *= 10;
	}
	result = (integer * 100 + fraction) * sign;
	if ((result > INT32_MAX) || (result < INT32_MIN))
	{
		return PARSE_RANGE;
	}
	*value = (int32_t)result;
	return PARSE_OK;
}

/**
  * @brief  判断指定轴是否已标定（系数非零、方向合法、限位范围合理）
  * @param  axis: 轴号
  * @retval 1=已标定，0=未标定
  */
static uint8_t MotionProtocol_AxisCalibrated(uint8_t axis)
{
	const MotionAxisConfig *config = &motion_axis_config[axis];
	return (uint8_t)((config->pulses_num != 0u) &&
					 (config->units_x100_den != 0u) &&
					 (config->positive_direction <= 1u) &&
					 ((config->limit_enable == 0u) ||
					  (config->limit_min_x100 <= config->limit_max_x100)));
}

/**
  * @brief  检查目标位置中所有需要运动的轴是否已标定
  * @param  target: 目标坐标数组
  * @retval 1=全部已标定，0=存在未标定且需要运动的轴
  */
static uint8_t MotionProtocol_CalibratedForTarget(const int32_t *target)
{
	uint8_t axis;
	for (axis = 0u; axis < MOTION_AXIS_COUNT; axis++)
	{
		if ((target[axis] != motion_position_x100[axis]) &&
			(MotionProtocol_AxisCalibrated(axis) == 0u))
		{
			return 0u;
		}
	}
	return 1u;
}

/**
  * @brief  检查指定坐标是否在该轴的行程限位范围内
  * @param  axis:          轴号
  * @param  position_x100: 坐标（0.01 单位）
  * @retval 1=在范围内或未启用限位，0=越限
  */
static uint8_t MotionProtocol_InRange(uint8_t axis, int32_t position_x100)
{
	const MotionAxisConfig *config = &motion_axis_config[axis];

	if (config->limit_enable == 0u)
	{
		return 1u;
	}
	return (uint8_t)((position_x100 >= config->limit_min_x100) &&
						 (position_x100 <= config->limit_max_x100));
}

/**
  * @brief  将坐标（0.01 单位）换算为脉冲数（四舍五入）
  * @param  axis:          轴号
  * @param  position_x100: 坐标
  * @param  pulses:        输出脉冲数
  * @retval 1=成功，0=未标定或结果溢出
  */
static uint8_t MotionProtocol_ToPulses(uint8_t axis, int32_t position_x100,
									 int64_t *pulses)
{
	const MotionAxisConfig *config = &motion_axis_config[axis];
	int64_t product;
	int64_t result;

	if ((config->pulses_num == 0u) || (config->units_x100_den == 0u))
	{
		return 0u;
	}
	product = (int64_t)position_x100 * (int64_t)config->pulses_num;
	/* 四舍五入除法（正负数分别处理） */
	if (product >= 0)
	{
		result = (product + ((int64_t)config->units_x100_den / 2)) /
				 (int64_t)config->units_x100_den;
	}
	else
	{
		result = (product - ((int64_t)config->units_x100_den / 2)) /
				 (int64_t)config->units_x100_den;
	}
	if ((result > MOTION_MAX_ABS_PULSE_POSITION) ||
		(result < -MOTION_MAX_ABS_PULSE_POSITION))
	{
		return 0u;
	}
	*pulses = result;
	return 1u;
}

/**
  * @brief  启动多轴绝对运动
  * @param  target:        目标坐标数组
  * @param  sequential_xy: 非 0 时 X/Y 轴顺序运动（旧版 GOTO），0 时全部同步
  * @retval 0=失败（已回错误包），1=已启动，2=无需运动（目标与当前相同）
  */
static uint8_t MotionProtocol_StartTargets(const int32_t *target,
										  uint8_t sequential_xy)
{
	A4988_MoveCommand commands[MOTION_AXIS_COUNT];
	A4988_Status status;
	int64_t delta;
	uint8_t axis;
	uint8_t count = 0u;

	/* 所有需要运动的轴必须已标定 */
	if (MotionProtocol_CalibratedForTarget(target) == 0u)
	{
		Usart_SendString("ERR,CAL\n");
		return 0u;
	}
	/* 逐轴校验范围并构造运动指令 */
	for (axis = 0u; axis < MOTION_AXIS_COUNT; axis++)
	{
		if (MotionProtocol_InRange(axis, target[axis]) == 0u)
		{
			Usart_SendString("ERR,RANGE\n");
			return 0u;
		}
		/* 目标与当前相同，记录脉冲位置后跳过 */
		if (target[axis] == motion_position_x100[axis])
		{
			motion_pending_pulse_position[axis] = motion_pulse_position[axis];
			continue;
		}
		/* 坐标换算为脉冲 */
		if (MotionProtocol_ToPulses(axis, target[axis],
								   &motion_pending_pulse_position[axis]) == 0u)
		{
			Usart_SendString("ERR,CAL\n");
			return 0u;
		}
		delta = motion_pending_pulse_position[axis] - motion_pulse_position[axis];
		/* 脉冲增量为 0，或顺序模式下非 X 轴先跳过 */
		if ((delta == 0) || ((sequential_xy != 0u) && (axis > A4988_AXIS_X)))
		{
			continue;
		}
		/* 构造单轴运动指令 */
		commands[count].id = axis;
		commands[count].direction = (A4988_Direction)(
			(delta > 0) ? motion_axis_config[axis].positive_direction :
			(motion_axis_config[axis].positive_direction ^ 1u));
		commands[count].pulses = (uint32_t)((delta > 0) ? delta : -delta);
		count++;
	}
	/* 所有轴都无需运动 */
	if (count == 0u)
	{
		return 2u;
	}
	/* 顺序模式：只先启动 X 或 Y 中需要运动的那一轴 */
	if (sequential_xy != 0u)
	{
		axis = (uint8_t)((motion_pending_pulse_position[A4988_AXIS_X] !=
					  motion_pulse_position[A4988_AXIS_X]) ? A4988_AXIS_X : A4988_AXIS_Y);
		/* X、Y 都无需运动则直接返回 */
		if ((motion_pending_pulse_position[A4988_AXIS_X] == motion_pulse_position[A4988_AXIS_X]) &&
			(motion_pending_pulse_position[A4988_AXIS_Y] == motion_pulse_position[A4988_AXIS_Y]))
		{
			return 2u;
		}
		delta = motion_pending_pulse_position[axis] - motion_pulse_position[axis];
		status = A4988_Move(axis,
			(A4988_Direction)((delta > 0) ? motion_axis_config[axis].positive_direction :
			(motion_axis_config[axis].positive_direction ^ 1u)),
			(uint32_t)((delta > 0) ? delta : -delta));
		pending_axis = axis;
	}
	else
	{
		/* 同步模式：多轴同时启动 */
		status = A4988_MoveGroup(commands, count);
	}
	if (status != A4988_OK)
	{
		Usart_SendString((status == A4988_ERROR_BUSY) ? "ERR,BUSY\n" : "ERR,PARAM\n");
		return 0u;
	}
	memcpy(pending_target, target, sizeof(pending_target));
	motion_pending = 1u;
	return 1u;
}

/**
  * @brief  启动单轴相对运动
  * @param  axis:      轴号
  * @param  delta_x100: 相对增量（0.01 单位）
  * @retval 0=失败，1=已启动，2=增量为 0 无需运动
  */
static uint8_t MotionProtocol_StartRelative(uint8_t axis, int32_t delta_x100)
{
	int32_t target[MOTION_AXIS_COUNT];
	int64_t target_pulses;
	int64_t delta_pulses;
	A4988_Status status;

	/* 暂存当前脉冲位置与坐标 */
	memcpy(motion_pending_pulse_position, motion_pulse_position,
		   sizeof(motion_pending_pulse_position));
	memcpy(target, motion_position_x100, sizeof(target));

	/* 校验标定与增量换算 */
	if ((MotionProtocol_AxisCalibrated(axis) == 0u) ||
		(MotionProtocol_ToPulses(axis, delta_x100, &target_pulses) == 0u))
	{
		Usart_SendString("ERR,CAL\n");
		return 0u;
	}
	target[axis] += delta_x100;
	if (MotionProtocol_InRange(axis, target[axis]) == 0u)
	{
		Usart_SendString("ERR,RANGE\n");
		return 0u;
	}
	/* 目标坐标换算为脉冲 */
	if (MotionProtocol_ToPulses(axis, target[axis],
							&motion_pending_pulse_position[axis]) == 0u)
	{
		Usart_SendString("ERR,CAL\n");
		return 0u;
	}
	delta_pulses = motion_pending_pulse_position[axis] - motion_pulse_position[axis];
	if (delta_pulses == 0)
	{
		memcpy(pending_target, target, sizeof(pending_target));
		return 2u;
	}
	/* 启动单轴运动 */
	status = A4988_Move(axis,
		(A4988_Direction)((delta_pulses > 0) ? motion_axis_config[axis].positive_direction :
		(motion_axis_config[axis].positive_direction ^ 1u)),
		(uint32_t)((delta_pulses > 0) ? delta_pulses : -delta_pulses));
	if (status != A4988_OK)
	{
		Usart_SendString((status == A4988_ERROR_BUSY) ? "ERR,BUSY\n" : "ERR,PARAM\n");
		return 0u;
	}
	memcpy(pending_target, target, sizeof(pending_target));
	pending_axis = axis;
	motion_pending = 1u;
	return 1u;
}

/**
  * @brief  以指定前缀回复当前 4 轴坐标
  * @param  prefix:   回复前缀字符串
  * @param  position: 坐标数组
  */
static void MotionProtocol_ReplyPosition(const char *prefix,
									 const int32_t *position)
{
	printf("%s,%ld,%ld,%ld,%ld\n", prefix,
		   (long)position[A4988_AXIS_X],
		   (long)position[A4988_AXIS_Y],
		   (long)position[A4988_AXIS_Z],
		   (long)position[A4988_AXIS_ROTATE]);
}

/**
  * @brief  PING 指令：存活检测
  */
static void MotionProtocol_CommandPing(const char *line)
{
	if (MotionProtocol_FieldCount(line) != 1u)
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	Usart_SendString("ACK,PING\n");
}

/**
  * @brief  STATUS 指令：查询当前状态与坐标
  */
static void MotionProtocol_CommandStatus(const char *line)
{
	if (MotionProtocol_FieldCount(line) != 1u)
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	printf("ACK,STATUS,%s,%s,%s,%ld,%ld,%ld,%ld\n",
		   (motion_enabled != 0u) ? "REAL" : "SIM",
		   ((motion_pending != 0u) || (A4988_AnyBusy() != 0u)) ? "BUSY" : "IDLE",
		   (motion_zeroed != 0u) ? "ZEROED" : "UNHOMED",
		   (long)motion_position_x100[A4988_AXIS_X],
		   (long)motion_position_x100[A4988_AXIS_Y],
		   (long)motion_position_x100[A4988_AXIS_Z],
		   (long)motion_position_x100[A4988_AXIS_ROTATE]);
}

/**
  * @brief  STOP 指令：急停全部电机、断开电磁铁、复位所有状态与坐标
  */
static void MotionProtocol_CommandStop(const char *line)
{
	if (MotionProtocol_FieldCount(line) != 1u)
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	MotionProtocol_Disarm();
	Usart_SendString("ACK,STOP\n");
}

/**
  * @brief  MOTION 指令：使能/禁用真实运动（0=仿真，1=真实）
  */
static void MotionProtocol_CommandMotion(const char *line)
{
	char field[MOTION_FIELD_MAX];
	uint8_t requested;
	uint32_t mask;

	if ((MotionProtocol_FieldCount(line) != 2u) ||
		(MotionProtocol_GetField(line, 1u, field, sizeof(field)) == 0u) ||
		((strcmp(field, "0") != 0) && (strcmp(field, "1") != 0)))
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	requested = (uint8_t)(field[0] == '1');
	/* 禁用时先断开电磁铁 */
	if (requested == 0u)
	{
		Aux_Stop();
	}
	/* 运动中不允许切换模式 */
	if (A4988_AnyBusy() != 0u)
	{
		Usart_SendString("ERR,BUSY\n");
		return;
	}
	if (requested != motion_enabled)
	{
		motion_enabled = requested;
		if (requested == 0u)
		{
			Aux_Stop();
		}
		/* 切换模式后清零回零状态与坐标 */
		motion_zeroed = 0u;
		MotionProtocol_ClearPosition();
	}
	mask = __get_PRIMASK();
	__disable_irq();
	/* Do not let a lease expiring during command parsing be rearmed by
	 * a MOTION command that was already queued in the old session. */
	if (motion_lease_expired != 0u)
	{
		__set_PRIMASK(mask);
		MotionProtocol_CheckLease();
		return;
	}
	motion_lease_expired = 0u;
	motion_lease_ms = requested ? MOTION_HOLD_LEASE_MS : 0u;
	A4988_SetHoldEnabled(requested);
	__set_PRIMASK(mask);
	printf("ACK,MOTION,%u\n", (unsigned int)motion_enabled);
}

/**
  * @brief  ZERO 指令：将当前位置设为坐标原点
  */
static void MotionProtocol_CommandZero(const char *line)
{
	if (MotionProtocol_FieldCount(line) != 1u)
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	if (A4988_AnyBusy() != 0u)
	{
		Usart_SendString("ERR,BUSY\n");
		return;
	}
	MotionProtocol_ClearPosition();
	motion_zeroed = 1u;
	Usart_SendString("ACK,ZERO\n");
}

/**
  * @brief  GOTO 指令：绝对定位
  *         3 字段为旧版（仅 X,Y），5 字段为原生（X,Y,Z,R）
  */
static void MotionProtocol_CommandGoto(const char *line)
{
	char field[MOTION_FIELD_MAX];
	int32_t target[MOTION_AXIS_COUNT];
	ParseResult parse_result;
	uint8_t axis;
	uint8_t fields = MotionProtocol_FieldCount(line);
	uint8_t legacy;

	if ((fields != 3u) && (fields != 5u))
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	legacy = (uint8_t)(fields == 3u);
	if (A4988_AnyBusy() != 0u)
	{
		Usart_SendString("ERR,BUSY\n");
		return;
	}
	/* 原生 GOTO 必须先回零 */
	if ((legacy == 0u) && (motion_zeroed == 0u))
	{
		Usart_SendString("ERR,ORIGIN\n");
		return;
	}

	/* 先以当前坐标填充，再覆盖指定轴 */
	memcpy(target, motion_position_x100, sizeof(target));
	for (axis = 0u; axis < (legacy ? 2u : MOTION_AXIS_COUNT); axis++)
	{
		if (MotionProtocol_GetField(line, (uint8_t)(axis + 1u),
									field, sizeof(field)) == 0u)
		{
			Usart_SendString("ERR,PARAM\n");
			return;
		}
		parse_result = MotionProtocol_ParseX100(field, &target[axis]);
		if (parse_result == PARSE_RANGE)
		{
			Usart_SendString("ERR,RANGE\n");
			return;
		}
		if (parse_result != PARSE_OK)
		{
			Usart_SendString("ERR,PARAM\n");
			return;
		}
		/* 旧版 GOTO 坐标必须非负，且不超过工作台范围 */
		if ((legacy != 0u) && (target[axis] < 0))
		{
			Usart_SendString("ERR,RANGE\n");
			return;
		}
		if ((legacy != 0u) &&
			(((axis == A4988_AXIS_X) && (target[axis] > 21000)) ||
			 ((axis == A4988_AXIS_Y) && (target[axis] > 29700))))
		{
			Usart_SendString("ERR,RANGE\n");
			return;
		}
		if (MotionProtocol_InRange(axis, target[axis]) == 0u)
		{
			Usart_SendString("ERR,RANGE\n");
			return;
		}
	}

	/* 旧版 GOTO 分支 */
	if (legacy != 0u)
	{
		/* 仿真模式：直接更新坐标并回包 */
		if (motion_enabled == 0u)
		{
			motion_position_x100[0] = target[0];
			motion_position_x100[1] = target[1];
			printf("ACK,GOTO,%ld,%ld\n", (long)target[0], (long)target[1]);
			return;
		}
		legacy_xy_target[0] = target[0];
		legacy_xy_target[1] = target[1];
		/* 顺序启动 X/Y，若无需运动则立即回包 */
		if (MotionProtocol_StartTargets(target, 1u) == 2u)
		{
			memcpy(motion_position_x100, target, sizeof(target));
			motion_pulse_position[0] = motion_pending_pulse_position[0];
			motion_pulse_position[1] = motion_pending_pulse_position[1];
			printf("ACK,GOTO,%ld,%ld\n", (long)target[0], (long)target[1]);
		}
		else if (motion_pending != 0u)
		{
			/* 标记当前启动的是 X 还是 Y，完成后继续下一轴 */
			pending_kind = (pending_axis == A4988_AXIS_X) ?
				PENDING_LEGACY_GOTO_X : PENDING_LEGACY_GOTO_Y;
		}
		return;
	}
	/* 原生 GOTO 分支 */
	if (motion_enabled == 0u)
	{
		memcpy(motion_position_x100, target, sizeof(target));
		MotionProtocol_ReplyPosition("ACK,GOTO", motion_position_x100);
		return;
	}
	if (MotionProtocol_StartTargets(target, 0u) == 2u)
	{
		memcpy(motion_position_x100, target, sizeof(target));
		memcpy(motion_pulse_position, motion_pending_pulse_position,
			   sizeof(motion_pulse_position));
		MotionProtocol_ReplyPosition("ACK,GOTO", motion_position_x100);
	}
	else if (motion_pending != 0u)
	{
		pending_kind = PENDING_NATIVE;
	}
}

/**
  * @brief  MOVE 指令：X/Y 轴相对移动（旧版）
  *         格式：MOVE,X/Y,增量
  */
static void MotionProtocol_CommandLegacyMove(const char *line)
{
	char axis_field[MOTION_FIELD_MAX];
	char value_field[MOTION_FIELD_MAX];
	int32_t delta;
	uint8_t axis;
	ParseResult result;

	if ((MotionProtocol_FieldCount(line) != 3u) ||
		(MotionProtocol_GetField(line, 1u, axis_field, sizeof(axis_field)) == 0u) ||
		(MotionProtocol_GetField(line, 2u, value_field, sizeof(value_field)) == 0u) ||
		((strcmp(axis_field, "X") != 0) && (strcmp(axis_field, "Y") != 0)))
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	result = MotionProtocol_ParseX100(value_field, &delta);
	if (result != PARSE_OK)
	{
		Usart_SendString((result == PARSE_RANGE) ? "ERR,RANGE\n" : "ERR,PARAM\n");
		return;
	}
	if ((delta == 0) || (A4988_AnyBusy() != 0u) || (motion_pending != 0u))
	{
		Usart_SendString((delta == 0) ? "ERR,PARAM\n" : "ERR,BUSY\n");
		return;
	}
	axis = (uint8_t)((axis_field[0] == 'X') ? A4988_AXIS_X : A4988_AXIS_Y);
	/* 旧版 MOVE 范围检查：X[0,21000]，Y[0,29700] */
	if (((axis == A4988_AXIS_X) &&
		 ((int64_t)motion_position_x100[axis] + delta < 0 ||
		  (int64_t)motion_position_x100[axis] + delta > 21000)) ||
		((axis == A4988_AXIS_Y) &&
		 ((int64_t)motion_position_x100[axis] + delta < 0 ||
		  (int64_t)motion_position_x100[axis] + delta > 29700)))
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	/* 仿真模式：直接更新坐标并回包 */
	if (motion_enabled == 0u)
	{
		motion_position_x100[axis] += delta;
		printf("ACK,MOVE,%s,%ld,%ld,%ld\n", axis_field, (long)delta,
			   (long)motion_position_x100[0], (long)motion_position_x100[1]);
		return;
	}
	pending_delta = delta;
	legacy_axis = axis;
	{
		uint8_t started = MotionProtocol_StartRelative(axis, delta);
		if (started == 1u)
		{
			pending_kind = PENDING_LEGACY_MOVE;
		}
		else if (started == 2u)
		{
			/* 增量为 0，直接更新坐标并回包 */
			motion_position_x100[axis] = pending_target[axis];
			motion_pulse_position[axis] = motion_pending_pulse_position[axis];
			MotionProtocol_ReplyLegacyMove();
		}
	}
}

/**
  * @brief  Z 指令：Z 轴上下运动（旧版，固定行程）
  *         格式：Z,UP 或 Z,DOWN
  */
static void MotionProtocol_CommandLegacyZ(const char *line)
{
	char field[MOTION_FIELD_MAX];
	int32_t target[MOTION_AXIS_COUNT];
	uint8_t down;
	int32_t stroke_x100;
	int64_t pulse_target;
	int64_t pulse_delta;
	A4988_Status status;

	if ((MotionProtocol_FieldCount(line) != 2u) ||
		(MotionProtocol_GetField(line, 1u, field, sizeof(field)) == 0u))
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	if (strcmp(field, "DOWN") == 0) down = 1u;
	else if (strcmp(field, "UP") == 0) down = 0u;
	else { Usart_SendString("ERR,PARAM\n"); return; }

	if ((motion_pending != 0u) || (A4988_AnyBusy() != 0u))
	{
		Usart_SendString("ERR,BUSY\n");
		return;
	}
	/* 仿真模式直接回包 */
	if (motion_enabled == 0u)
	{
		printf("ACK,Z,%s\n", field);
		return;
	}
	/* 必须已配置 Z 轴行程且已标定 */
	if ((MOTION_Z_STROKE_PULSES == 0u) ||
		(MotionProtocol_AxisCalibrated(A4988_AXIS_Z) == 0u))
	{
		Usart_SendString("ERR,CAL\n");
		return;
	}
	/* 将固定行程脉冲数换算为坐标增量 */
	memcpy(motion_pending_pulse_position, motion_pulse_position,
		   sizeof(motion_pending_pulse_position));
	stroke_x100 = (int32_t)(((int64_t)MOTION_Z_STROKE_PULSES *
		motion_axis_config[A4988_AXIS_Z].units_x100_den +
		motion_axis_config[A4988_AXIS_Z].pulses_num / 2u) /
		motion_axis_config[A4988_AXIS_Z].pulses_num);
	memcpy(target, motion_position_x100, sizeof(target));
	target[A4988_AXIS_Z] += down ? stroke_x100 : -stroke_x100;
	/* 校验范围与换算 */
	if ((MotionProtocol_InRange(A4988_AXIS_Z, target[A4988_AXIS_Z]) == 0u) ||
		(MotionProtocol_ToPulses(A4988_AXIS_Z, target[A4988_AXIS_Z], &pulse_target) == 0u) ||
		(MotionProtocol_ToPulses(A4988_AXIS_Z,
			motion_position_x100[A4988_AXIS_Z], &pulse_delta) == 0u))
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	pulse_delta = pulse_target - pulse_delta;
	/* 校验脉冲增量方向与指令一致 */
	if ((pulse_delta == 0) || ((down != 0u) && (pulse_delta < 0)) ||
		((down == 0u) && (pulse_delta > 0)))
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	/* 启动 Z 轴运动 */
	status = A4988_Move(A4988_AXIS_Z,
		(A4988_Direction)((pulse_delta > 0) ? motion_axis_config[A4988_AXIS_Z].positive_direction :
		(motion_axis_config[A4988_AXIS_Z].positive_direction ^ 1u)),
		(uint32_t)((pulse_delta > 0) ? pulse_delta : -pulse_delta));
	if (status != A4988_OK)
	{
		Usart_SendString((status == A4988_ERROR_BUSY) ? "ERR,BUSY\n" : "ERR,PARAM\n");
		return;
	}
	strncpy(pending_z_text, field, sizeof(pending_z_text) - 1u);
	pending_z_text[sizeof(pending_z_text) - 1u] = '\0';
	memcpy(pending_target, target, sizeof(pending_target));
	pending_kind = PENDING_LEGACY_Z;
	motion_pending = 1u;
}

/**
  * @brief  ROTATE 指令：旋转轴相对转动（旧版）
  *         格式：ROTATE,角度（0.01 度）
  */
static void MotionProtocol_CommandLegacyRotate(const char *line)
{
	char field[MOTION_FIELD_MAX];
	int32_t angle;
	int64_t pulses;
	int64_t current_pulses;
	A4988_Status status;
	ParseResult result;

	if ((MotionProtocol_FieldCount(line) != 2u) ||
		(MotionProtocol_GetField(line, 1u, field, sizeof(field)) == 0u))
	{
		Usart_SendString("ERR,PARAM\n");
		return;
	}
	result = MotionProtocol_ParseX100(field, &angle);
	if (result != PARSE_OK)
	{
		Usart_SendString((result == PARSE_RANGE) ? "ERR,RANGE\n" : "ERR,PARAM\n");
		return;
	}
	/* 角度范围 [-180, 180] 度 */
	if ((angle < -18000) || (angle > 18000))
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	if ((motion_pending != 0u) || (A4988_AnyBusy() != 0u))
	{
		Usart_SendString("ERR,BUSY\n");
		return;
	}
	/* 仿真模式 */
	if (motion_enabled == 0u)
	{
		if (((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle < -18000) ||
			((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle > 18000))
		{
			Usart_SendString("ERR,RANGE\n");
			return;
		}
		motion_position_x100[A4988_AXIS_ROTATE] += angle;
		printf("ACK,ROTATE,%ld\n", (long)angle);
		return;
	}
	/* 必须已标定 */
	if (MotionProtocol_AxisCalibrated(A4988_AXIS_ROTATE) == 0u)
	{
		Usart_SendString("ERR,CAL\n");
		return;
	}
	/* 校验合成角度在范围内 */
	if (((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle < -18000) ||
		((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle > 18000) ||
		(MotionProtocol_InRange(A4988_AXIS_ROTATE,
			(int32_t)((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle)) == 0u))
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	if (MotionProtocol_ToPulses(A4988_AXIS_ROTATE,
		(int32_t)((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle),
		&current_pulses) == 0u)
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	memcpy(motion_pending_pulse_position, motion_pulse_position,
		   sizeof(motion_pending_pulse_position));
	if (MotionProtocol_ToPulses(A4988_AXIS_ROTATE,
		(int32_t)((int64_t)motion_position_x100[A4988_AXIS_ROTATE] + angle),
		&motion_pending_pulse_position[A4988_AXIS_ROTATE]) == 0u)
	{
		Usart_SendString("ERR,RANGE\n");
		return;
	}
	pulses = motion_pending_pulse_position[A4988_AXIS_ROTATE] -
		motion_pulse_position[A4988_AXIS_ROTATE];
	if (pulses == 0)
	{
		/* 脉冲增量为 0，直接更新坐标回包 */
		motion_position_x100[A4988_AXIS_ROTATE] += angle;
		printf("ACK,ROTATE,%ld\n", (long)angle);
		return;
	}
	/* 启动旋转轴运动 */
	memcpy(pending_target, motion_position_x100, sizeof(pending_target));
	pending_target[A4988_AXIS_ROTATE] += angle;
	status = A4988_Move(A4988_AXIS_ROTATE,
		(A4988_Direction)((pulses > 0) ? motion_axis_config[A4988_AXIS_ROTATE].positive_direction :
		(motion_axis_config[A4988_AXIS_ROTATE].positive_direction ^ 1u)),
		(uint32_t)((pulses > 0) ? pulses : -pulses));
	if (status != A4988_OK)
	{
		Usart_SendString((status == A4988_ERROR_BUSY) ? "ERR,BUSY\n" : "ERR,PARAM\n");
		return;
	}
	pending_delta = angle;
	pending_kind = PENDING_LEGACY_ROTATE;
	motion_pending = 1u;
}

/**
  * @brief  旧版加工计划指令处理：PLAN/PIECE/ENDPIECE/ENDPLAN
  *         用于多工件加工流程的顺序控制
  */
static void MotionProtocol_HandleLegacyPlan(const char *line)
{
	char field[MOTION_FIELD_MAX];
	uint8_t id;

	if (MotionProtocol_CommandIs(line, "PLAN"))
	{
		/* 开始计划：指定总件数（1~4） */
		if ((MotionProtocol_FieldCount(line) != 2u) ||
			(MotionProtocol_GetField(line, 1u, field, sizeof(field)) == 0u) ||
			((field[0] < '1') || (field[0] > '4') || (field[1] != '\0')))
		{
			Usart_SendString("ERR,PARAM\n"); return;
		}
		if (legacy_plan_active != 0u) { Usart_SendString("ERR,SEQUENCE\n"); return; }
		legacy_plan_total = (uint8_t)(field[0] - '0');
		legacy_piece_done = 0u; legacy_piece_active = 0u; legacy_plan_active = 1u;
		printf("ACK,PLAN,%u\n", (unsigned int)legacy_plan_total); return;
	}
	if (MotionProtocol_CommandIs(line, "PIECE"))
	{
		/* 开始单件加工 */
		if ((MotionProtocol_FieldCount(line) != 2u) ||
			(MotionProtocol_GetField(line, 1u, field, sizeof(field)) == 0u) ||
			((field[0] < '1') || (field[0] > '4') || (field[1] != '\0')))
		{ Usart_SendString("ERR,PARAM\n"); return; }
		id = (uint8_t)(field[0] - '0');
		if ((legacy_plan_active == 0u) || (legacy_piece_active != 0u) ||
			(legacy_piece_done >= legacy_plan_total))
		{ Usart_SendString("ERR,SEQUENCE\n"); return; }
		legacy_piece_active = 1u; legacy_piece_id = id;
		printf("ACK,PIECE,%u\n", (unsigned int)id); return;
	}
	if (MotionProtocol_CommandIs(line, "ENDPIECE"))
	{
		/* 结束单件加工：编号须匹配且无运动进行中 */
		if ((MotionProtocol_FieldCount(line) != 2u) ||
			(MotionProtocol_GetField(line, 1u, field, sizeof(field)) == 0u) ||
			((field[0] < '1') || (field[0] > '4') || (field[1] != '\0')))
		{ Usart_SendString("ERR,PARAM\n"); return; }
		id = (uint8_t)(field[0] - '0');
		if ((legacy_piece_active == 0u) || (id != legacy_piece_id) ||
			(motion_pending != 0u))
		{ Usart_SendString((motion_pending != 0u) ? "ERR,BUSY\n" : "ERR,SEQUENCE\n"); return; }
		legacy_piece_active = 0u; legacy_piece_done++;
		printf("ACK,ENDPIECE,%u\n", (unsigned int)id); return;
	}
	if (MotionProtocol_CommandIs(line, "ENDPLAN"))
	{
		/* 结束计划：所有件须已完成 */
		if ((MotionProtocol_FieldCount(line) != 1u) || (legacy_plan_active == 0u) ||
			(legacy_piece_active != 0u) || (legacy_piece_done != legacy_plan_total))
		{ Usart_SendString("ERR,SEQUENCE\n"); return; }
		legacy_plan_active = 0u;
		printf("ACK,ENDPLAN,%u\n", (unsigned int)legacy_piece_done); return;
	}
}

/**
  * @brief  回复旧版 MOVE 指令完成包
  */
static void MotionProtocol_ReplyLegacyMove(void)
{
	printf("ACK,MOVE,%c,%ld,%ld,%ld\n",
		(legacy_axis == A4988_AXIS_X) ? 'X' : 'Y', (long)pending_delta,
		(long)motion_position_x100[A4988_AXIS_X],
		(long)motion_position_x100[A4988_AXIS_Y]);
}

/**
  * @brief  指令分发：根据命令字调用对应的处理函数
  * @param  line: 指令行
  */
static void MotionProtocol_HandleLine(const char *line)
{
	if (MotionProtocol_CheckLease() != 0u) return;
	if (line[0] == '\0')
	{
		return;
	}
	/* Exact, non-moving heartbeats already used by both real executors.
	 * Malformed or unrelated input cannot keep the hold alive. */
	if (strcmp(line, "PING") == 0 || strcmp(line, "STATUS") == 0 ||
		strcmp(line, "IOSTATUS") == 0)
		MotionProtocol_RefreshLease();
	/* Auxiliary stop/status must remain available during stepper motion. */
	if (Aux_HandleLine(line)) return;
	/* 运动进行中只允许 PING/STATUS/STOP */
	if ((motion_pending != 0u) &&
		(MotionProtocol_CommandIs(line, "PING") == 0u) &&
		(MotionProtocol_CommandIs(line, "STATUS") == 0u) &&
		(MotionProtocol_CommandIs(line, "STOP") == 0u))
	{
		Usart_SendString("ERR,BUSY\n");
		return;
	}
	if (MotionProtocol_CommandIs(line, "PING") != 0u)
	{
		MotionProtocol_CommandPing(line);
	}
	else if (MotionProtocol_CommandIs(line, "STATUS") != 0u)
	{
		MotionProtocol_CommandStatus(line);
	}
	else if (MotionProtocol_CommandIs(line, "STOP") != 0u)
	{
		MotionProtocol_CommandStop(line);
	}
	else if (MotionProtocol_CommandIs(line, "MOTION") != 0u)
	{
		MotionProtocol_CommandMotion(line);
	}
	else if (MotionProtocol_CommandIs(line, "ZERO") != 0u)
	{
		MotionProtocol_CommandZero(line);
	}
	else if (MotionProtocol_CommandIs(line, "GOTO") != 0u)
	{
		MotionProtocol_CommandGoto(line);
	}
	else if (MotionProtocol_CommandIs(line, "MOVE") != 0u)
	{
		MotionProtocol_CommandLegacyMove(line);
	}
	else if (MotionProtocol_CommandIs(line, "Z") != 0u)
	{
		MotionProtocol_CommandLegacyZ(line);
	}
	else if (MotionProtocol_CommandIs(line, "ROTATE") != 0u)
	{
		MotionProtocol_CommandLegacyRotate(line);
	}
	else if ((MotionProtocol_CommandIs(line, "PLAN") != 0u) ||
			 (MotionProtocol_CommandIs(line, "PIECE") != 0u) ||
			 (MotionProtocol_CommandIs(line, "ENDPIECE") != 0u) ||
			 (MotionProtocol_CommandIs(line, "ENDPLAN") != 0u))
	{
		MotionProtocol_HandleLegacyPlan(line);
	}
	else
	{
		Usart_SendString("ERR,UNKNOWN\n");
	}
}

/**
  * @brief  轮询运动完成：当无运动进行中且有待完成任务时，更新坐标并回包
  *         对旧版 GOTO 的顺序 X/Y 运动，会在 X 完成后自动启动 Y
  */
static void MotionProtocol_PollMove(void)
{
	if (MotionProtocol_CheckLease() != 0u) return;
	if ((motion_pending == 0u) || (A4988_AnyBusy() != 0u))
	{
		return;
	}
	/* 旧版 GOTO：X 轴完成后启动 Y 轴 */
	if (pending_kind == PENDING_LEGACY_GOTO_X)
	{
		motion_position_x100[A4988_AXIS_X] = legacy_xy_target[0];
		motion_pulse_position[A4988_AXIS_X] = motion_pending_pulse_position[A4988_AXIS_X];
		if (motion_pending_pulse_position[A4988_AXIS_Y] != motion_pulse_position[A4988_AXIS_Y])
		{
			int64_t delta = motion_pending_pulse_position[A4988_AXIS_Y] - motion_pulse_position[A4988_AXIS_Y];
			A4988_Status status = A4988_Move(A4988_AXIS_Y,
				(A4988_Direction)((delta > 0) ? motion_axis_config[A4988_AXIS_Y].positive_direction :
				(motion_axis_config[A4988_AXIS_Y].positive_direction ^ 1u)),
				(uint32_t)((delta > 0) ? delta : -delta));
			if (status != A4988_OK)
			{
				motion_pending = 0u; pending_kind = 0u;
				Usart_SendString("ERR,MOTION\n");
				return;
			}
			pending_kind = PENDING_LEGACY_GOTO_Y;
			return;
		}
		pending_kind = PENDING_LEGACY_GOTO_Y;
	}
	/* 旧版 GOTO Y 轴完成，回包 */
	if (pending_kind == PENDING_LEGACY_GOTO_Y)
	{
		motion_position_x100[A4988_AXIS_X] = legacy_xy_target[0];
		motion_position_x100[A4988_AXIS_Y] = legacy_xy_target[1];
		motion_pulse_position[A4988_AXIS_Y] = motion_pending_pulse_position[A4988_AXIS_Y];
		motion_pending = 0u; pending_kind = 0u;
		printf("ACK,GOTO,%ld,%ld\n", (long)legacy_xy_target[0], (long)legacy_xy_target[1]);
		return;
	}
	/* 通用完成处理：更新坐标与脉冲位置，按类型回包 */
	memcpy(motion_position_x100, pending_target, sizeof(motion_position_x100));
	memcpy(motion_pulse_position, motion_pending_pulse_position,
		   sizeof(motion_pulse_position));
	motion_pending = 0u;
	switch (pending_kind)
	{
	case PENDING_NATIVE:
		MotionProtocol_ReplyPosition("ACK,GOTO", motion_position_x100);
		break;
	case PENDING_LEGACY_MOVE:
		MotionProtocol_ReplyLegacyMove();
		break;
	case PENDING_LEGACY_Z:
		printf("ACK,Z,%s\n", pending_z_text);
		break;
	case PENDING_LEGACY_ROTATE:
		printf("ACK,ROTATE,%ld\n", (long)pending_delta);
		break;
	default:
		break;
	}
	pending_kind = 0u;
}

/**
  * @brief  协议初始化：清零全部状态与坐标，释放电磁铁
  */
void MotionProtocol_Init(void)
{
	MotionProtocol_Disarm();
	motion_line_length = 0u;
	motion_discard_line = 0u;
	motion_enabled = 0u;
	motion_zeroed = 0u;
	motion_pending = 0u;
	pending_kind = 0u;
	legacy_plan_active = 0u;
	legacy_piece_active = 0u;
	legacy_plan_total = 0u;
	legacy_piece_done = 0u;
	MotionProtocol_ClearPosition();
	Aux_Stop();
}

/**
  * @brief  协议主循环处理函数，需在主循环中持续调用
  *         1. 轮询运动完成
  *         2. 读取串口字节，组装指令行
  *         3. 遇到换行时解析并执行指令
  */
void MotionProtocol_Process(void)
{
	uint8_t byte;

	if (MotionProtocol_CheckLease() != 0u) return;
	MotionProtocol_PollMove();
	/* 检查接收溢出 */
	if (Usart_TakeRxOverflow() != 0u)
	{
		motion_discard_line = 1u;
	}
	/* 逐字节读取串口 */
	while (Usart_ReadByte(&byte) != 0u)
	{
		if (MotionProtocol_CheckLease() != 0u) return;
		if (Usart_TakeRxOverflow() != 0u)
		{
			motion_discard_line = 1u;
		}
		/* 忽略回车 */
		if (byte == '\r')
		{
			continue;
		}
		/* 换行：行结束 */
		if (byte == '\n')
		{
			if (motion_discard_line != 0u)
			{
				Usart_SendString("ERR,OVERFLOW\n");
			}
			else
			{
				motion_line[motion_line_length] = '\0';
				MotionProtocol_HandleLine(motion_line);
			}
			motion_line_length = 0u;
			motion_discard_line = 0u;
			continue;
		}
		/* 当前行需丢弃时跳过 */
		if (motion_discard_line != 0u)
		{
			continue;
		}
		/* 行超长则标记丢弃 */
		if (motion_line_length >= MOTION_LINE_MAX)
		{
			motion_discard_line = 1u;
			continue;
		}
		motion_line[motion_line_length] = (char)byte;
		motion_line_length++;
	}
}
