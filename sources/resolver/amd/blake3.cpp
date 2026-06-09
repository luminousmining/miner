#include <iomanip>
#include <sstream>

#include <CL/opencl.hpp>

#include <algo/blake3/blake3.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>
#include <common/error/opencl_error.hpp>
#include <common/log/log.hpp>
#include <resolver/amd/blake3.hpp>


resolver::ResolverAmdBlake3::ResolverAmdBlake3() : resolver::ResolverAmd()
{
    algorithm = algo::ALGORITHM::BLAKE3;
}


resolver::ResolverAmdBlake3::~ResolverAmdBlake3()
{
    parameters.headerCache.free();
    parameters.targetCache.free();
    parameters.resultCache.free();
}


bool resolver::ResolverAmdBlake3::updateMemory([[maybe_unused]] stratum::StratumJobInfo const& jobInfo)
{
    if (nullptr == clContext) [[unlikely]]
    {
        return false;
    }
    if (nullptr == clQueue[0] || nullptr == clQueue[1]) [[unlikely]]
    {
        return false;
    }

    parameters.headerCache.free();
    parameters.targetCache.free();
    parameters.resultCache.free();

    if (false == parameters.headerCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.targetCache.alloc(clQueue[currentIndexStream], *clContext)
        || false == parameters.resultCache.alloc(clQueue[currentIndexStream], *clContext))
    {
        return false;
    }

    if (false == buildSearch())
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdBlake3::updateConstants(stratum::StratumJobInfo const& jobInfo)
{
    if (false == parameters.headerCache.setBufferDevice(clQueue[currentIndexStream], &jobInfo.headerBlob))
    {
        return false;
    }
    if (false == parameters.targetCache.setBufferDevice(clQueue[currentIndexStream], &jobInfo.targetBlob))
    {
        return false;
    }

    overrideOccupancy(128u, 8192u);

    return true;
}


bool resolver::ResolverAmdBlake3::buildSearch()
{
    kernelGenerator.clear();
    kernelGenerator.setKernelName("search");
    kernelGenerator.addDefine("MAX_RESULT", algo::blake3::MAX_RESULT);

    if (false == kernelGenerator.appendFile("kernel/blake3/blake3.cl"))
    {
        return false;
    }
    if (false == kernelGenerator.build(clDevice, clContext))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdBlake3::executeSync(stratum::StratumJobInfo const& jobInfo)
{
    auto& clKernel{ kernelGenerator.clKernel };
    OPENCL_ER(clKernel.setArg(0u, *(parameters.headerCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(1u, *(parameters.targetCache.getBuffer())));
    OPENCL_ER(clKernel.setArg(2u, jobInfo.nonce));
    OPENCL_ER(clKernel.setArg(3u, jobInfo.fromGroup));
    OPENCL_ER(clKernel.setArg(4u, jobInfo.toGroup));
    OPENCL_ER(clKernel.setArg(5u, *(parameters.resultCache.getBuffer())));

    size_t const globalSize{ static_cast<size_t>(getBlocks()) * static_cast<size_t>(getThreads()) };
    size_t const localSize{ static_cast<size_t>(getThreads()) };
    OPENCL_ER(clQueue[currentIndexStream]
                  ->enqueueNDRangeKernel(clKernel, cl::NullRange, cl::NDRange(globalSize), cl::NDRange(localSize)));
    OPENCL_ER(clQueue[currentIndexStream]->finish());

    if (false == getResultCache(jobInfo.jobIDStr, jobInfo.fromGroup, jobInfo.toGroup, jobInfo.extraNonceSize))
    {
        return false;
    }

    return true;
}


bool resolver::ResolverAmdBlake3::executeAsync(stratum::StratumJobInfo const& jobInfo)
{
    // Correctness-first: single-queue. Double-buffering is a deferred follow-up.
    return executeSync(jobInfo);
}


bool resolver::ResolverAmdBlake3::getResultCache(
    std::string const& _jobId,
    uint32_t const     fromGroup,
    uint32_t const     toGroup,
    uint32_t const     extraNonceSize)
{
    algo::blake3::Result data{};

    if (false == parameters.resultCache.getBufferHost(clQueue[currentIndexStream], &data))
    {
        return false;
    }

    if (true == data.found)
    {
        uint32_t const count{ common::max_limit(data.count, algo::blake3::MAX_RESULT) };

        resultShare.found = true;
        resultShare.fromGroup = fromGroup;
        resultShare.toGroup = toGroup;
        resultShare.count = count;
        resultShare.extraNonceSize = extraNonceSize;
        resultShare.jobId.assign(_jobId);

        for (uint32_t i{ 0u }; i < count; ++i)
        {
            resultShare.nonces[i] = data.nonces[i];
        }

        if (false == parameters.resultCache.resetBufferHost(clQueue[currentIndexStream]))
        {
            return false;
        }
    }

    return true;
}


void resolver::ResolverAmdBlake3::submit(stratum::Stratum* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                // Fixed 16-hex-char (8-byte) field first — std::hex alone drops leading
                // zeros, which would byte-misalign the 24-byte nonce the pool recomputes.
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                std::string nonceStr{ nonceHexa.str() };
                while (nonceStr.size() < 48)
                {
                    nonceStr += "0";
                }

                boost::json::object params{};
                params["jobId"] = resultShare.jobId;
                params["fromGroup"] = resultShare.fromGroup;
                params["toGroup"] = resultShare.toGroup;
                params["nonce"] = nonceStr;

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}


void resolver::ResolverAmdBlake3::submit(stratum::StratumSmartMining* const stratum)
{
    if (true == resultShare.found)
    {
        if (false == isStale(resultShare.jobId))
        {
            for (uint32_t i{ 0u }; i < resultShare.count; ++i)
            {
                std::stringstream nonceHexa;
                nonceHexa << std::setw(16) << std::setfill('0') << std::hex << resultShare.nonces[i];

                boost::json::object params{};
                params["jobId"] = resultShare.jobId;
                params["fromGroup"] = resultShare.fromGroup;
                params["toGroup"] = resultShare.toGroup;
                params["nonce"] = nonceHexa.str().substr(resultShare.extraNonceSize);

                stratum->miningSubmit(deviceId, params);

                resultShare.nonces[i] = 0ull;
            }
        }
    }

    resultShare.count = 0u;
    resultShare.found = false;
}
