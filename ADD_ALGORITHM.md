# Adding a New Algorithm

This guide describes every step required to integrate a new mining algorithm into LuminousMiner. The fictional algorithm **MyNewPOW** is used as the running example throughout.

> **Recommended reference**: KawPOW is the simplest existing implementation to read — it inherits from ProgPOW without adding any logic of its own. For a non-DAG algorithm, see BLAKE3 (NVIDIA only) or AUTOLYKOS_V2.

---

## Files Overview

| File | Action |
|---|---|
| `sources/algo/algo_type.hpp` | Add enum value |
| `sources/algo/algo_type.cpp` | Add toString / toEnum |
| `sources/algo/mynewpow/mynewpow.hpp` | **Create** – algorithm constants |
| `sources/algo/mynewpow/mynewpow.cpp` | **Create** – CPU implementation (kernel generator) |
| `sources/stratum/mynewpow.hpp` | **Create** – stratum header |
| `sources/stratum/mynewpow.cpp` | **Create** – stratum implementation |
| `sources/stratum/stratums.hpp` | Add include |
| `sources/stratum/stratums.cpp` | Add case in `NewStratum()` |
| `sources/resolver/amd/mynewpow.hpp` | **Create** – OpenCL resolver |
| `sources/resolver/amd/mynewpow.cpp` | **Create** – OpenCL implementation |
| `sources/resolver/amd/tests/mynewpow.cpp` | **Create** – AMD unit tests |
| `sources/resolver/nvidia/mynewpow.hpp` | **Create** – CUDA resolver |
| `sources/resolver/nvidia/mynewpow.cpp` | **Create** – CUDA implementation |
| `sources/resolver/nvidia/tests/mynewpow.cpp` | **Create** – NVIDIA unit tests |
| `sources/device/device.cpp` | Add case in `setAlgorithm()` |

The `CMakeLists.txt` files use glob patterns — no build system changes are needed as long as the files follow the naming conventions above.

---

## Step 1 – Algorithm Enum

### `sources/algo/algo_type.hpp`

Add the new value **before** `UNKNOWN`:

```cpp
enum class ALGORITHM : uint8_t
{
    SHA256,
    ETHASH,
    // ... existing algorithms ...
    BLAKE3,
    MYNEWPOW,   // <-- add here
    UNKNOWN,
    MAX_SIZE = algo::ALGORITHM::UNKNOWN
};
```

### `sources/algo/algo_type.cpp`

Add a case in **each** of the two functions:

```cpp
// In toString()
case algo::ALGORITHM::MYNEWPOW:
{
    return "mynewpow";
}

// In toEnum()
else if (algo == "mynewpow")
{
    return algo::ALGORITHM::MYNEWPOW;
}
```

The string returned by `toString()` is what the user passes via `--algo mynewpow`.

---

## Step 2 – Algorithm Constants

Create the directory `sources/algo/mynewpow/` with the following files.

### `sources/algo/mynewpow/mynewpow.hpp`

This file holds only the constants consumed by the resolver and the stratum.

```cpp
#pragma once

#include <cstdint>


namespace algo
{
    namespace mynewpow
    {
        // DAG constants (if Ethash-based algorithm)
        constexpr uint32_t DAG_ITEM_PARENTS{ 512u };

        // ProgPOW constants (if derived from ProgPOW)
        constexpr uint32_t MAX_PERIOD{ 3u };
        constexpr uint32_t COUNT_CACHE{ 11u };
        constexpr uint32_t COUNT_MATH{ 18u };
    }
}
```

For a non-DAG algorithm, adapt the constants to the algorithm's requirements (block size, number of rounds, etc.).

### `sources/algo/mynewpow/mynewpow.cpp`

For a ProgPOW derivative, this file implements the kernel generator functions (math/merge sequence). See `sources/algo/progpow/kawpow.cpp` for a complete example. For a standalone algorithm, this file can remain empty or contain CPU helper functions.

---

## Step 3 – Stratum

The stratum handles pool communication: subscribe, authorize, job reception, and share submission.

### Case 1: deriving from an existing protocol (most common)

If the stratum protocol is identical to ProgPOW/Ethash (same `mining.notify` and `mining.submit` format), inherit and configure the algorithm-specific parameters:

**`sources/stratum/mynewpow.hpp`**
```cpp
#pragma once

#include <stratum/progpow.hpp>


namespace stratum
{
    struct StratumMyNewPOW : public stratum::StratumProgPOW
    {
      public:
        StratumMyNewPOW();
        ~StratumMyNewPOW() = default;
    };
}
```

**`sources/stratum/mynewpow.cpp`**
```cpp
#include <algo/progpow/mynewpow.hpp>
#include <stratum/mynewpow.hpp>


stratum::StratumMyNewPOW::StratumMyNewPOW() : stratum::StratumProgPOW()
{
    maxPeriod      = algo::mynewpow::MAX_PERIOD;
    maxEpochLength = algo::progpow::EPOCH_LENGTH;
}
```

### Case 2: custom protocol

If the pool uses a different protocol, inherit directly from `stratum::Stratum` and implement the virtual methods:

**`sources/stratum/mynewpow.hpp`**
```cpp
#pragma once

#include <stratum/stratum.hpp>


namespace stratum
{
    struct StratumMyNewPOW : public stratum::Stratum
    {
      public:
        void onResponse(boost::json::object const& root) final;
        void onMiningNotify(boost::json::object const& root) final;
        void onMiningSetDifficulty(boost::json::object const& root) final;
        void miningSubmit(uint32_t const deviceId, boost::json::array const& params) final;
    };
}
```

**`sources/stratum/mynewpow.cpp`** – skeleton to fill in:
```cpp
#include <stratum/mynewpow.hpp>


void stratum::StratumMyNewPOW::onMiningNotify(boost::json::object const& root)
{
    // Extract jobId, headerHash, seedHash, cleanJob from root["params"]
    // Call updateJob() to notify devices
}

void stratum::StratumMyNewPOW::onMiningSetDifficulty(boost::json::object const& root)
{
    // Parse difficulty and update jobInfo.boundary
}

void stratum::StratumMyNewPOW::onResponse(boost::json::object const& root)
{
    // Handle pool responses by id (subscribe=1, authorize=2, submit>=1000)
}

void stratum::StratumMyNewPOW::miningSubmit(
    uint32_t const            deviceId,
    boost::json::array const& params)
{
    // Build and send the mining.submit JSON message
}
```

See `sources/stratum/progpow.cpp` and `sources/stratum/autolykos_v2.cpp` for complete implementation examples.

### Registering the stratum

**`sources/stratum/stratums.hpp`** – add the include:
```cpp
#include <stratum/mynewpow.hpp>
```

**`sources/stratum/stratums.cpp`** – add the case in `NewStratum()` before the `UNKNOWN` case:
```cpp
case algo::ALGORITHM::MYNEWPOW:
{
    stratum = NEW(stratum::StratumMyNewPOW);
    break;
}
```

---

## Step 4 – AMD Resolver (OpenCL)

The resolver runs the GPU kernel and submits found shares.

### Case 1: ProgPOW derivative (simplest)

**`sources/resolver/amd/mynewpow.hpp`**
```cpp
#pragma once

#if defined(AMD_ENABLE)

#include <resolver/amd/progpow.hpp>


namespace resolver
{
    class ResolverAmdMyNewPOW : public resolver::ResolverAmdProgPOW
    {
      public:
        ResolverAmdMyNewPOW();
        ~ResolverAmdMyNewPOW() = default;
    };
}

#endif
```

**`sources/resolver/amd/mynewpow.cpp`**
```cpp
#include <algo/ethash/ethash.hpp>
#include <algo/progpow/mynewpow.hpp>
#include <resolver/amd/mynewpow.hpp>


resolver::ResolverAmdMyNewPOW::ResolverAmdMyNewPOW() : resolver::ResolverAmdProgPOW()
{
    ///////////////////////////////////////////////////////////////////////////
    algorithm = algo::ALGORITHM::MYNEWPOW;

    ///////////////////////////////////////////////////////////////////////////
    // Ethash DAG
    maxEpoch            = algo::ethash::MAX_EPOCH_NUMBER;
    dagCountItemsGrowth = algo::ethash::DAG_COUNT_ITEMS_GROWTH;
    dagCountItemsInit   = algo::ethash::DAG_COUNT_ITEMS_INIT;

    ///////////////////////////////////////////////////////////////////////////
    // MyNewPOW specifics
    progpowVersion = algo::progpow::VERSION::KAWPOW; // replace if using a custom version
    dagItemParents = algo::mynewpow::DAG_ITEM_PARENTS;
    countCache     = algo::mynewpow::COUNT_CACHE;
    countMath      = algo::mynewpow::COUNT_MATH;
}
```

### Case 2: non-DAG or custom structure

Inherit from `resolver::ResolverAmd` and implement the 5 pure virtual methods:

```cpp
#pragma once

#if defined(AMD_ENABLE)

#include <resolver/amd/amd.hpp>


namespace resolver
{
    class ResolverAmdMyNewPOW : public resolver::ResolverAmd
    {
      public:
        ResolverAmdMyNewPOW();
        ~ResolverAmdMyNewPOW();

        bool updateMemory(stratum::StratumJobInfo const& jobInfo) final;
        bool updateConstants(stratum::StratumJobInfo const& jobInfo) final;
        bool executeSync(stratum::StratumJobInfo const& jobInfo) final;
        bool executeAsync(stratum::StratumJobInfo const& jobInfo) final;
        void submit(stratum::Stratum* const stratum) final;
        void submit(stratum::StratumSmartMining* const stratum) final;
    };
}

#endif
```

See `sources/resolver/amd/autolykos_v2.hpp` / `.cpp` for a complete non-ProgPOW example.

---

## Step 5 – NVIDIA Resolver (CUDA)

Same structure as the AMD resolver, under `sources/resolver/nvidia/`.

**`sources/resolver/nvidia/mynewpow.hpp`**
```cpp
#pragma once

#if defined(CUDA_ENABLE)

#include <resolver/nvidia/progpow.hpp>


namespace resolver
{
    class ResolverNvidiaMyNewPOW : public resolver::ResolverNvidiaProgPOW
    {
      public:
        ResolverNvidiaMyNewPOW();
        ~ResolverNvidiaMyNewPOW() = default;
    };
}

#endif
```

**`sources/resolver/nvidia/mynewpow.cpp`**
```cpp
#include <algo/ethash/ethash.hpp>
#include <algo/progpow/mynewpow.hpp>
#include <resolver/nvidia/mynewpow.hpp>


resolver::ResolverNvidiaMyNewPOW::ResolverNvidiaMyNewPOW() : resolver::ResolverNvidiaProgPOW()
{
    algorithm      = algo::ALGORITHM::MYNEWPOW;
    maxEpoch       = algo::ethash::MAX_EPOCH_NUMBER;
    progpowVersion = algo::progpow::VERSION::KAWPOW;
    dagItemParents = algo::mynewpow::DAG_ITEM_PARENTS;
    countCache     = algo::mynewpow::COUNT_CACHE;
    countMath      = algo::mynewpow::COUNT_MATH;
}
```

---

## Step 6 – Register in `device/device.cpp`

### Add the includes at the top of the file

```cpp
#include <resolver/amd/mynewpow.hpp>
#include <resolver/nvidia/mynewpow.hpp>
```

### Add the case in `setAlgorithm()`

Copy the block from a similar algorithm (e.g. KAWPOW) and adapt:

```cpp
case algo::ALGORITHM::MYNEWPOW:
{
    switch (deviceType)
    {
#if defined(CUDA_ENABLE)
        case device::DEVICE_TYPE::NVIDIA:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverNvidiaMyNewPOW);
            break;
        }
#endif
#if defined(AMD_ENABLE)
        case device::DEVICE_TYPE::AMD:
        {
            SAFE_DELETE(resolver);
            resolver = NEW(resolver::ResolverAmdMyNewPOW);
            break;
        }
#endif
        case device::DEVICE_TYPE::UNKNOWN:
        {
            break;
        }
    }
    break;
}
```

> If the algorithm is not supported on one platform (e.g. BLAKE3 is NVIDIA only), leave the AMD case empty with just `break;`. The device will be killed with `RESOLVER_NULLPTR` if no resolver is instantiated.

---

## Step 7 – Unit Tests

Tests verify that the resolver finds (or does not find) a known nonce for a given job.

### `sources/resolver/amd/tests/mynewpow.cpp`

```cpp
#include <CL/opencl.hpp>
#include <gtest/gtest.h>

#include <algo/hash.hpp>
#include <algo/hash_utils.hpp>
#include <algo/progpow/mynewpow.hpp>
#include <common/log/log.hpp>
#include <common/mocker/stratum.hpp>
#include <resolver/amd/mynewpow.hpp>
#include <resolver/tests/amd.hpp>


struct ResolverMyNewPOWAmdTest : public testing::Test
{
    stratum::StratumJobInfo        jobInfo{};
    resolver::tests::Properties    properties{};
    resolver::ResolverAmdMyNewPOW  resolver{};
    common::mocker::MockerStratum  stratum{};

    ResolverMyNewPOWAmdTest()
    {
        common::setLogLevel(common::TYPELOG::__DEBUG);
    }

    ~ResolverMyNewPOWAmdTest()
    {
        properties.clDevice  = nullptr;
        properties.clContext = nullptr;
        properties.clQueue   = nullptr;
    }

    void initializeDevice(uint32_t const index)
    {
        if (false == resolver::tests::initializeOpenCL(properties, index))
        {
            logErr() << "fail init opencl";
        }
        resolver.setDevice(&properties.clDevice);
        resolver.setQueue(&properties.clQueue);
        resolver.setContext(&properties.clContext);
    }

    void initializeJob(uint64_t const nonce)
    {
        jobInfo.nonce       = nonce;
        jobInfo.blockNumber = 965398ull;
        jobInfo.headerHash  = algo::toHash256("<header_hash_hex>");
        jobInfo.seedHash    = algo::toHash256("<seed_hash_hex>");
        jobInfo.boundary    = algo::toHash256("<boundary_hex>");
        jobInfo.boundaryU64 = algo::toUINT64(jobInfo.boundary);
        jobInfo.epoch       = algo::ethash::ContextGenerator::instance()
                                .findEpoch(jobInfo.seedHash, algo::progpow::EPOCH_LENGTH);
        jobInfo.period      = jobInfo.blockNumber / algo::mynewpow::MAX_PERIOD;
    }
};


TEST_F(ResolverMyNewPOWAmdTest, findNonce)
{
    initializeDevice(0u);
    initializeJob(/* known starting nonce */ 0x0000000000000000ull);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    ASSERT_FALSE(stratum.paramSubmit.empty());

    std::string const nonceStr{ stratum.paramSubmit[1].as_string().c_str() };
    using namespace std::string_literals;
    EXPECT_EQ("0x<expected_nonce>"s, nonceStr);
}


TEST_F(ResolverMyNewPOWAmdTest, notFindNonce)
{
    initializeDevice(0u);
    initializeJob(/* nonce with no solution in the search window */ 0x0000000000000001ull);

    ASSERT_TRUE(resolver.updateMemory(jobInfo));
    ASSERT_TRUE(resolver.updateConstants(jobInfo));
    ASSERT_TRUE(resolver.executeSync(jobInfo));
    resolver.submit(&stratum);

    EXPECT_TRUE(stratum.paramSubmit.empty());
}


TEST_F(ResolverMyNewPOWAmdTest, allDeviceFindNonce)
{
    uint32_t const countDevice{ resolver::tests::getDeviceCount() };
    for (uint32_t index{ 0u }; index < countDevice; ++index)
    {
        initializeDevice(index);
        initializeJob(0x0000000000000000ull);

        ASSERT_TRUE(resolver.updateMemory(jobInfo));
        ASSERT_TRUE(resolver.updateConstants(jobInfo));
        ASSERT_TRUE(resolver.executeSync(jobInfo));
        resolver.submit(&stratum);

        ASSERT_FALSE(stratum.paramSubmit.empty());
    }
}
```

### `sources/resolver/nvidia/tests/mynewpow.cpp`

Same structure — replace `ResolverAmdMyNewPOW` with `ResolverNvidiaMyNewPOW` and use `resolver::tests::initializeCUDA()` instead of `initializeOpenCL()`. See `sources/resolver/nvidia/tests/kawpow.cpp` for the exact template.

---

## Step 8 – Build Verification

```sh
# Reconfigure to pick up new files (glob re-evaluation)
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build the miner
cmake --build build --target miner -j$(nproc)

# Build and run unit tests
cmake --build build --target unit_test -j$(nproc)
./bin/unit_test --gtest_filter=*MyNewPOW*

# Check formatting
cmake --build build --target format-check
```

---

## Final Checklist

- [ ] `algo_type.hpp` – enum value added before `UNKNOWN`
- [ ] `algo_type.cpp` – case in `toString()` and branch in `toEnum()`
- [ ] `sources/algo/mynewpow/mynewpow.hpp` – constants defined
- [ ] `sources/stratum/mynewpow.hpp` / `.cpp` – stratum implemented
- [ ] `sources/stratum/stratums.hpp` – include added
- [ ] `sources/stratum/stratums.cpp` – case in `NewStratum()`
- [ ] `sources/resolver/amd/mynewpow.hpp` / `.cpp` – OpenCL resolver
- [ ] `sources/resolver/nvidia/mynewpow.hpp` / `.cpp` – CUDA resolver
- [ ] `sources/device/device.cpp` – includes + case in `setAlgorithm()`
- [ ] AMD and NVIDIA tests created with a known test vector
- [ ] Clean build with no warnings
- [ ] `format-check` passes with no diff
