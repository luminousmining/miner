#pragma once

#if defined (TOOL_TRACE_MEMORY)

#include <unordered_map>
#include <string>


namespace common
{
    namespace tool
    {
        class ToolMemoryTrace
        {
        public:
            struct AllocationInfo
            {
                std::string functionName;
                uint32_t    line{ 0u };
                size_t      size{ 0u };
            };

            static ToolMemoryTrace& instance();

            void show() const;
            void registerAllocation(void* ptr,
                                    size_t const size,
                                    char const* functionName,
                                    uint32_t const line);
            void registerDesallocation(void* ptr,
                                       char const* functionName,
                                       uint32_t const line);

        private:
            ToolMemoryTrace() = default;
            ToolMemoryTrace(ToolMemoryTrace const&) = delete;
            void operator=(ToolMemoryTrace const&) = delete;

            std::unordered_map<void*, AllocationInfo> allocations;
            size_t totalAllocated{ 0u };
            size_t totalFreed{ 0u };
            size_t currentAllocated{ 0u };
        };
    }
}



#define TOOL_MEMORY_ALLOC(ptr, size)\
    common::tool::ToolMemoryTrace::instance().registerAllocation\
    (\
        ptr,\
        size,\
        __FUNCTION__,\
        __LINE__\
    );

#define TOOL_MEMORY_FREE(ptr)\
    common::tool::ToolMemoryTrace::instance().registerDesallocation\
    (\
        ptr,\
        __FUNCTION__,\
        __LINE__\
    );

#else

#define TOOL_MEMORY_ALLOC(ptr, size) {}
#define TOOL_MEMORY_FREE(ptr) {}

#endif
