#pragma once

__constant__ uint4* d_dag;
__constant__ uint4* d_light_cache;
__constant__ uint4 d_header[2];
__constant__ uint64_t d_boundary;
__constant__ uint32_t d_dag_number_item;
__constant__ uint32_t d_light_number_item;
