#pragma once


#define reg_load(var, lane_id, width)\
    __shfl_sync(0xffffffff, var, lane_id, width)
