#include <utility>

#include <common/custom.hpp>
#include <network/write_pump.hpp>


network::WritePump::WritePump(Transmit transmit) : transmit{ std::move(transmit) }
{
}


void network::WritePump::enqueue(Payload payload)
{
    Payload next{};
    {
        UNIQUE_LOCK(mutex);
        queue.push(std::move(payload));
        if (true == writeInFlight)
        {
            return;
        }
        writeInFlight = true;
        next = queue.front();
    }
    transmit(next);
}


void network::WritePump::onComplete(bool const success)
{
    Payload next{};
    {
        UNIQUE_LOCK(mutex);
        if (false == queue.empty())
        {
            queue.pop();
        }
        if (false == success)
        {
            std::queue<Payload> empty{};
            queue.swap(empty);
        }
        if (true == queue.empty())
        {
            writeInFlight = false;
            return;
        }
        next = queue.front();
    }
    transmit(next);
}
