#pragma once

#include <boost/atomic/atomic.hpp>

#include <common/log/log.hpp>


namespace common
{
    template<typename T>
    struct AtomicCounter
    {
        boost::atomic<T> current{};
        T                last{};

        AtomicCounter(T const value)
        {
            current = value;
            last = value;
        }

        T get() const
        {
            return current.load(boost::memory_order::seq_cst);
        }

        bool isEqual() const
        {
            return last == get();
        }

        void add(T const value)
        {
            current.fetch_add(value, boost::memory_order::seq_cst);
        }

        void sub(T const value)
        {
            current.fetch_sub(value, boost::memory_order::seq_cst);
        }

        void update(T const value)
        {
            last = value;
        }
    };
}
