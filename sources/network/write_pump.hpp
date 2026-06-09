#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>


namespace network
{
    // Serializes outbound writes for a single socket: at most one payload is
    // "in flight" (handed to transmit, not yet acknowledged via onComplete) and
    // payloads are sent in FIFO order. transmit is invoked OUTSIDE the internal
    // mutex, so a transmit that calls onComplete synchronously cannot deadlock.
    class WritePump
    {
      public:
        using Payload = std::shared_ptr<std::string const>;
        using Transmit = std::function<void(Payload const&)>;

        explicit WritePump(Transmit transmit);

        // Queue a payload. If nothing is in flight, hands the head to transmit.
        void enqueue(Payload payload);

        // Owner reports the in-flight write finished.
        //   success: drop the completed head; send next if any, else go idle.
        //   failure: drop all queued payloads (stale; connection going down) and
        //            go idle.
        void onComplete(bool success);

      private:
        std::mutex          mutex{};
        std::queue<Payload> queue{};
        bool                writeInFlight{ false };
        Transmit            transmit{};
    };
}
