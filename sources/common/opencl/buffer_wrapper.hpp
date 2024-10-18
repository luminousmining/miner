#pragma once

#if defined(AMD_ENABLE)

#include <CL/opencl.hpp>

#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>


namespace common
{
    namespace opencl
    {
        struct BufferData
        {
            cl::Buffer*  buffer{ nullptr };
            cl_mem_flags flags{};
            size_t       size{ 0u };
        };


        template<typename TBuffer>
        struct Buffer : private BufferData
        {
            explicit Buffer(cl_mem_flags _flags,
                            size_t const _size = sizeof(TBuffer))
            : BufferData()
            {
                flags = _flags;
                size = _size;
            }

            inline
            void setSize(size_t const _size)
            {
                size = _size;
            }

            inline
            void setCapacity(size_t const capacity)
            {
                size = capacity * sizeof(TBuffer);
            }

            inline
            cl::Buffer* getBuffer()
            {
                return buffer;
            }

            inline
            bool alloc(cl::Context& clContext)
            {
                if (size == 0u)
                {
                    logErr() << "Cannot alloc! The size is 0!";
                    return false;
                }

                free();

                OPENCL_CATCH(
                    buffer = NEW(cl::Buffer(clContext, flags, size)));
                IS_NULL(buffer);

                return true;
            }

            inline
            void free()
            {
                if (nullptr != buffer)
                {
                    *buffer = nullptr;
                    delete buffer;
                    buffer = nullptr;
                }
            }

            inline
            bool write(TBuffer* src,
                       size_t const _size,
                       cl::CommandQueue* const clQueue)
            {
                OPENCL_ER(
                    clQueue->enqueueWriteBuffer(
                        *buffer,
                        CL_TRUE,
                        0,
                        _size,
                        src));
                return true;
            }
        };
    }
}

#endif
