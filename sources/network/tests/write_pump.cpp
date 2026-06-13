#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <network/write_pump.hpp>


namespace
{
    network::WritePump::Payload makePayload(std::string const& s)
    {
        return std::make_shared<std::string const>(s);
    }


    // Records each transmitted payload; never auto-completes.
    struct Recorder
    {
        std::vector<std::string> sent{};

        network::WritePump::Transmit transmit()
        {
            return [this](network::WritePump::Payload const& p)
            {
                sent.push_back(*p);
            };
        }
    };
}


TEST(WritePump, FirstEnqueueTransmitsImmediately)
{
    Recorder           rec{};
    network::WritePump pump{ rec.transmit() };

    pump.enqueue(makePayload("a"));

    ASSERT_EQ(1u, rec.sent.size());
    EXPECT_EQ("a", rec.sent[0]);
}


TEST(WritePump, SecondEnqueueWaitsForCompletion)
{
    Recorder           rec{};
    network::WritePump pump{ rec.transmit() };

    pump.enqueue(makePayload("a"));
    pump.enqueue(makePayload("b"));

    // Only "a" is in flight; "b" must wait.
    ASSERT_EQ(1u, rec.sent.size());
    EXPECT_EQ("a", rec.sent[0]);

    pump.onComplete(true);

    ASSERT_EQ(2u, rec.sent.size());
    EXPECT_EQ("b", rec.sent[1]);
}


TEST(WritePump, DrainsInFifoOrder)
{
    Recorder           rec{};
    network::WritePump pump{ rec.transmit() };

    pump.enqueue(makePayload("a"));
    pump.enqueue(makePayload("b"));
    pump.enqueue(makePayload("c"));
    pump.onComplete(true); // a done -> b
    pump.onComplete(true); // b done -> c
    pump.onComplete(true); // c done -> idle

    std::vector<std::string> const expected{ "a", "b", "c" };
    EXPECT_EQ(expected, rec.sent);
}


TEST(WritePump, GoesIdleThenRestartsOnNextEnqueue)
{
    Recorder           rec{};
    network::WritePump pump{ rec.transmit() };

    pump.enqueue(makePayload("a"));
    pump.onComplete(true); // queue empty -> idle

    pump.enqueue(makePayload("b")); // must transmit immediately again

    std::vector<std::string> const expected{ "a", "b" };
    EXPECT_EQ(expected, rec.sent);
}


TEST(WritePump, FailureClearsQueuedPayloads)
{
    Recorder           rec{};
    network::WritePump pump{ rec.transmit() };

    pump.enqueue(makePayload("a"));
    pump.enqueue(makePayload("b"));
    pump.enqueue(makePayload("c"));

    pump.onComplete(false); // a failed -> drop b, c; idle

    ASSERT_EQ(1u, rec.sent.size()); // only "a" was ever transmitted
    EXPECT_EQ("a", rec.sent[0]);

    pump.enqueue(makePayload("d")); // fresh start
    ASSERT_EQ(2u, rec.sent.size());
    EXPECT_EQ("d", rec.sent[1]);
}


TEST(WritePump, ConcurrentEnqueueAndCompleteNeverStalls)
{
    // Stress enqueue from many threads while a single consumer completes once
    // per transmit. The pump must never stall (every enqueue eventually
    // transmits exactly once).
    std::atomic<int>   transmitted{ 0 };
    network::WritePump pump{ [&transmitted](network::WritePump::Payload const&)
                             {
                                 transmitted.fetch_add(1, std::memory_order_relaxed);
                             } };

    int const producers{ 8 };
    int const perThread{ 1000 };

    std::vector<std::thread> threads{};
    for (int t{ 0 }; t < producers; ++t)
    {
        threads.emplace_back(
            [&pump]
            {
                for (int i{ 0 }; i < perThread; ++i)
                {
                    pump.enqueue(std::make_shared<std::string const>("x"));
                }
            });
    }

    // Consumer: keep completing until every enqueue has been transmitted.
    int const total{ producers * perThread };
    int       completed{ 0 };
    while (completed < total)
    {
        if (transmitted.load(std::memory_order_relaxed) > completed)
        {
            pump.onComplete(true);
            ++completed;
        }
        else
        {
            std::this_thread::yield();
        }
    }

    for (auto& th : threads)
    {
        th.join();
    }

    EXPECT_EQ(total, transmitted.load());
}
