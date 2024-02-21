#pragma once

#include <CL/opencl.hpp>

#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>


namespace common
{
    namespace opencl
    {
        struct BufferMappedData
        {
            cl::Buffer*  buffer{ nullptr };
            void*        mapped{ nullptr };
            cl_mem_flags flags{};
            size_t       size{ 0u };
        };

        template<typename TBuffer>
        struct BufferMapped : private BufferMappedData
        {
            explicit BufferMapped(cl_mem_flags _flags,
                                  size_t const _size = sizeof(TBuffer))
                : BufferMappedData()
            {
                flags = _flags;
                size = _size;
            }

            inline
            cl::Buffer* getBuffer()
            {
                return buffer;
            }

            inline
            bool alloc(cl::CommandQueue* const clQueue,
                       cl::Context& clContext)
            {
                if (size == 0u)
                {
                    logErr() << "Cannot alloc! The size is 0!";
                    return false;
                }

                SAFE_DELETE(buffer);
                OPENCL_CATCH(
                    buffer = new(std::nothrow) cl::Buffer(
                        clContext, flags, size));
                IS_NULL(buffer);

                OPENCL_CATCH(
                    mapped = clQueue->enqueueMapBuffer(
                        *buffer,
                        CL_TRUE,
                        flags,
                        0,
                        size));
                IS_NULL(mapped);

                if (false == resetBufferHost(clQueue))
                {
                    return false;
                }

                return true;
            }

            inline
            void free()
            {
                SAFE_DELETE(buffer);
            }

            inline
            bool getBufferHost(cl::CommandQueue* const clQueue,
                               TBuffer* const dst)
            {
                IS_NULL(clQueue->get());
                IS_NULL(buffer);
                IS_NULL(buffer->get());
                IS_NULL(mapped);

                IS_NULL(memcpy(dst, mapped, size));
                OPENCL_ER(
                    ::clEnqueueUnmapMemObject(
                        clQueue->get(),
                        buffer->get(),
                        mapped,
                        0,
                        nullptr,
                        nullptr));

                return true;
            }

            inline
            bool setBufferDevice(cl::CommandQueue* const clQueue,
                                 TBuffer const* const value)
            {
                IS_NULL(clQueue->get());
                IS_NULL(buffer);
                IS_NULL(buffer->get());
                IS_NULL(mapped);

                IS_NULL(memcpy(mapped, value, size));
                OPENCL_ER(
                    ::clEnqueueUnmapMemObject(
                        clQueue->get(),
                        buffer->get(),
                        mapped,
                        0,
                        nullptr,
                        nullptr));
                return true;
            }

            inline
            bool resetBufferHost(cl::CommandQueue* const clQueue)
            {
                TBuffer defaultValue{};

                IS_NULL(buffer);
                IS_NULL(buffer->get());
                IS_NULL(clQueue->get());
                IS_NULL(mapped);

                IS_NULL(memcpy(mapped, &defaultValue, size));
                OPENCL_ER(
                    ::clEnqueueUnmapMemObject(
                        clQueue->get(),
                        buffer->get(),
                        mapped,
                        0,
                        nullptr,
                        nullptr));

                return true;
            }
        };
    }
}
