#pragma once

// DAG
constexpr uint32_t DAG_SIZE{ 8388606u };

// Common
constexpr uint32_t LANES{ 16u };
constexpr uint32_t REGS{ 32u };
constexpr uint32_t COUNT_DAG{ 64u };
constexpr uint32_t LANE_ID_MAX{ 15u };
constexpr uint32_t STATE_LEN{ 25u };
constexpr uint32_t MODULE_CACHE{ 4096u };

// Load dag
constexpr uint32_t THREAD_COUNT{ 256u };
constexpr uint32_t HEADER_ITEM_BY_THREAD{ 16u };
constexpr uint32_t THREAD_BY_BLOCK{ 32u };
