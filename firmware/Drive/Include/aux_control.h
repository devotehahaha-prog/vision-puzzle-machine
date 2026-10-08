#ifndef AUX_CONTROL_H
#define AUX_CONTROL_H
#include <stdint.h>
void Aux_Stop(void);
void Aux_Tick1ms(void);
/* Returns 1 for a consumed auxiliary command, including malformed requests. */
uint8_t Aux_HandleLine(const char *line);
#endif
