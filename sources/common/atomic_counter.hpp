#pragma once

#include <boost/atomic/atomic.hpp>

#include <common/log/log.hpp>


namespace common
{
    template<typename T>
    struct AtomicCounter
    {
        boost::atomic<T>    current{};
        T                   last{};
        boost::memory_order memoryOrder{ boost::memory_order::seq_cst };

        explicit AtomicCounter(T const value) : current(value), last(value)
        {
        }

        // Set once before the counter is shared across threads (memoryOrder itself is not atomic).
        void setMemoryOrder(boost::memory_order const order)
        {
            memoryOrder = order;
        }

        T get() const
        {
            return current.load(memoryOrder);
        }

        bool isEqual() const
        {
            return last == get();
        }

        void store(T const value)
        {
            current.store(value, memoryOrder);
        }

        T add(T const value)
        {
            return current.fetch_add(value, memoryOrder);
        }

        T sub(T const value)
        {
            return current.fetch_sub(value, memoryOrder);
        }

        void update(T const value)
        {
            last = value;
        }
    };
}
