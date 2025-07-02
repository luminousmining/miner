#if defined(CUDA_ENABLE)

#if defined(__linux__)
    #include <experimental/filesystem>
    namespace __fs = std::experimental::filesystem;
#else
    #include <filesystem>
    namespace __fs = std::filesystem;
#endif
#include <fstream>

#include <algo/crypto/kiss99.hpp>
#include <algo/progpow/evrprogpow.hpp>
#include <algo/progpow/firopow.hpp>
#include <algo/progpow/kawpow.hpp>
#include <algo/progpow/meowpow.hpp>
#include <algo/progpow/progpow_quai.hpp>
#include <algo/progpow/progpow.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>


void algo::progpow::nvidia::writeSequenceMergeEntries(
    std::stringstream& ss,
    uint32_t const i,
    uint32_t const x,
    uint32_t const sel)
{
    ////////////////////////////////////////////////////////////////////////////
    ss << "\t" << "// iter[" << i << "] merge_entries " << sel % 4 << "\n";

    ////////////////////////////////////////////////////////////////////////////
    std::string l { "mix[" + std::to_string(x) + "]" };
    std::string ret { "mix[" + std::to_string(x) + "] = " };

    ////////////////////////////////////////////////////////////////////////////
    std::string r;
    switch (i)
    {
        case 0: r = "entries->x"; break;
        case 1: r = "entries->y"; break;
        case 2: r = "entries->z"; break;
        case 3: r = "entries->w"; break;
    }

    ////////////////////////////////////////////////////////////////////////////
    switch (sel % 4u)
    {
        case 0u: ss << "\t" << ret << "(" << l << " * 33) + " << r << ";\n"; break;
        case 1u: ss << "\t" << ret << "(" << l << " ^ " << r << ") * 33" << ";\n"; break;
        case 2u: ss << "\t" << ret << "rol_u32(" << l << ", " << (((sel >> 16) % 31) + 1) << ") ^ " << r << ";\n"; break;
        case 3u: ss << "\t" << ret << "ror_u32(" << l << ", " << (((sel >> 16) % 31) + 1) << ") ^ " << r << ";\n"; break;
    }
}


void algo::progpow::nvidia::writeSequenceMathMerge(
    std::stringstream& ss,
    uint32_t const i,
    uint32_t const dst,
    uint32_t const src1,
    uint32_t const src2,
    uint32_t const sel_math,
    uint32_t const sel_merge)
{
    ////////////////////////////////////////////////////////////////////////////
    ss << "\t" << "// iter[" << i << "] sel_math " << sel_math % 11 << "\n";

    ////////////////////////////////////////////////////////////////////////////
    std::string l { "mix[" + std::to_string(src1) + "]" };
    std::string r { "mix[" + std::to_string(src2) + "]" };
    std::string ret { "data = " };

    ////////////////////////////////////////////////////////////////////////////
    switch (sel_math % 11)
    {
        case 0u:  ss << "\t" << ret << "__clz(" << l << ") + __clz(" << r << ")" << ";\n"; break;
        case 1u: ss << "\t" << ret << "__popc(" << l << ") + __popc(" << r << ")" << ";\n"; break;
        case 2u:  ss << "\t" << ret << l << " + " << r << ";\n"; break;
        case 3u:  ss << "\t" << ret << l << " * " << r << ";\n"; break;
        case 4u:  ss << "\t" << ret << "__umulhi(" << l << ", " << r << ")" << ";\n"; break;
        case 5u:  ss << "\t" << ret << "min(" << l << ", " << r << ")" << ";\n"; break;
        case 6u:  ss << "\t" << ret << "rol_u32(" << l << ", " << r << ")" << ";\n"; break;
        case 7u:  ss << "\t" << ret << "ror_u32(" << l << ", " << r << ")" << ";\n"; break;
        case 8u:  ss << "\t" << ret << l << " & " << r << ";\n"; break;
        case 9u:  ss << "\t" << ret << l << " | " << r << ";\n"; break;
        case 10u:  ss << "\t" << ret << l << " ^ " << r << ";\n"; break;
    }
    l = "mix[" + std::to_string(dst) + "]";
    r = "data";
    ret = "mix[" + std::to_string(dst) + "] = ";

    ////////////////////////////////////////////////////////////////////////////
    ss << "\t" << "// iter[" << i << "] sel_merge " << sel_merge % 4 << "\n";
    switch (sel_merge % 4u)
    {
        case 0u: ss << "\t" << ret << "(" << l << " * 33) + " << r << ";\n"; break;
        case 1u: ss << "\t" << ret << "(" << l << " ^ " << r << ") * 33" << ";\n"; break;
        case 2u: ss << "\t" << ret << "rol_u32(" << l << ", " << (((sel_merge >> 16) % 31) + 1) << ") ^ " << r << ";\n"; break;
        case 3u: ss << "\t" << ret << "ror_u32(" << l << ", " << (((sel_merge >> 16) % 31) + 1) << ") ^ " << r << ";\n"; break;
    }
}


void algo::progpow::nvidia::writeSequenceMergeCache(
    std::stringstream& ss,
    uint32_t const i,
    uint32_t const src,
    uint32_t const dst,
    uint32_t const sel)
{
    ////////////////////////////////////////////////////////////////////////////
    ss << "\t" << "// iter[" << i << "] merge " << sel % 4 << "\n";
    ss << "\t" << "dag_offset = mix[" << src << "] & " << (algo::progpow::MODULE_CACHE - 1) << "u" << ";\n";

    ////////////////////////////////////////////////////////////////////////////
    std::string const l { "mix[" + std::to_string(dst) + "]" };
    std::string const r { "header_dag[dag_offset]" };
    std::string const ret { "mix[" + std::to_string(dst) + "] = " };

    ////////////////////////////////////////////////////////////////////////////
    switch (sel % 4u)
    {
        case 0u: ss << "\t" << ret << "(" << l << " * 33) + " << r << ";\n"; break;
        case 1u: ss << "\t" << ret << "(" << l << " ^ " << r << ") * 33" << ";\n"; break;
        case 2u: ss << "\t" << ret << "rol_u32(" << l << ", " << ((sel >> 16) % 31) + 1 << ") ^ " << r << ";\n"; break;
        case 3u: ss << "\t" << ret << "ror_u32(" << l << ", " << ((sel >> 16) % 31) + 1 << ") ^ " << r << ";\n"; break;
    }
}

#include <algo/algo_type.hpp>

void algo::progpow::writeMathRandomKernelCuda(
    algo::progpow::VERSION const progpowVersion,
    uint32_t const deviceId,
    uint64_t const period,
    uint32_t const countCache,
    uint32_t const countMath,
    uint32_t const regs,
    uint32_t const moduleSource)
{
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    __fs::path pathSequenceMath { "kernel" };
    pathSequenceMath /= "progpow";
    pathSequenceMath /=
        "sequence_math_random"s
        + "_"s + std::to_string(deviceId)
        + "_"s + std::to_string(period)
        + ".cuh"s;
    __fs::create_directories(pathSequenceMath.parent_path());
    std::ofstream ofs{ pathSequenceMath };

    ////////////////////////////////////////////////////////////////////////////
    int32_t* dst{ NEW_ARRAY(int32_t, regs) };
    int32_t* src{ NEW_ARRAY(int32_t, regs) };
    algo::Kiss99Properties round{ algo::progpow::initializeRound(period, dst, src, regs) };

    ////////////////////////////////////////////////////////////////////////////
    std::stringstream ss;
    ss << "// period " << std::to_string(period) << "\n"
    << "// COUNT_CACHE = " << countCache << "\n"
    << "// COUNT_MATH = " << countMath << "\n"
    << "__device__ __forceinline__" << "\n"
    << "void sequence_dynamic("<< "\n"
    << "\t" << "uint32_t const* __restrict__ const header_dag," << "\n"
    << "\t" << "uint32_t* __restrict__ const mix,"<< "\n"
    << "\t" << "uint4 const* __restrict__ entries" << ")" << "\n"
    << "{" << "\n"
    << "\t" << "uint32_t dag_offset;" << "\n"
    << "\t" << "uint32_t data;" << "\n";

    ////////////////////////////////////////////////////////////////////////////
    uint32_t dstCnt{ 0u };
    uint32_t srcCnt{ 0u };
    uint32_t const max{ countCache > countMath ? countCache : countMath };
    for (auto i{ 0u }; i < max; ++i)
    {
        ////////////////////////////////////////////////////////////////////////////
        if (i < countCache)
        {
            int32_t  const srcValue{ src[srcCnt % regs] };
            int32_t  const dstValue{ dst[dstCnt % regs] };
            uint32_t const sel{ algo::kiss99(round) };
            ++srcCnt;
            ++dstCnt;

            switch (progpowVersion)
            {
                case algo::progpow::VERSION::V_0_9_2:      /* algo::progpow::VERSION::V_0_9_4 */
                case algo::progpow::VERSION::V_0_9_3:      /* algo::progpow::VERSION::V_0_9_4 */;
                case algo::progpow::VERSION::V_0_9_4:      algo::progpow::nvidia::writeSequenceMergeCache(ss, i, srcValue, dstValue, sel); break;
                case algo::progpow::VERSION::KAWPOW:       algo::kawpow::nvidia::writeSequenceMergeCache(ss, i, srcValue, dstValue, sel); break;
                case algo::progpow::VERSION::MEOWPOW:      algo::meowpow::nvidia::writeSequenceMergeCache(ss, i, srcValue, dstValue, sel); break;
                case algo::progpow::VERSION::FIROPOW:      algo::firopow::nvidia::writeSequenceMergeCache(ss, i, srcValue, dstValue, sel); break;
                case algo::progpow::VERSION::EVRPROGPOW:   algo::evrprogpow::nvidia::writeSequenceMergeCache(ss, i, srcValue, dstValue, sel); break;
                case algo::progpow::VERSION::PROGPOWQUAI:  algo::progpow_quai::nvidia::writeSequenceMergeCache(ss, i, srcValue, dstValue, sel); break;
            }
        }

        ////////////////////////////////////////////////////////////////////////////
        if (i < countMath)
        {
            uint32_t const srcRnd{ algo::kiss99(round) % moduleSource };
            uint32_t const src1{ srcRnd % regs };
            uint32_t       src2{ srcRnd / regs };
            uint32_t const sel1{ algo::kiss99(round) };
            uint32_t const sel2{ algo::kiss99(round) };
            int32_t const  dstValue{ dst[dstCnt % regs] };

            if (src2 >= src1)
            {
                ++src2;
            }
            ++dstCnt;
            switch (progpowVersion)
            {
                case algo::progpow::VERSION::V_0_9_2:      /* algo::progpow::VERSION::V_0_9_4*/
                case algo::progpow::VERSION::V_0_9_3:      /* algo::progpow::VERSION::V_0_9_4*/
                case algo::progpow::VERSION::V_0_9_4:      algo::progpow::nvidia::writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2); break;
                case algo::progpow::VERSION::KAWPOW:       algo::kawpow::nvidia::writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2); break;
                case algo::progpow::VERSION::MEOWPOW:      algo::meowpow::nvidia::writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2); break;
                case algo::progpow::VERSION::FIROPOW:      algo::firopow::nvidia::writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2); break;
                case algo::progpow::VERSION::EVRPROGPOW:   algo::evrprogpow::nvidia::writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2); break;
                case algo::progpow::VERSION::PROGPOWQUAI:  algo::progpow_quai::nvidia::writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2); break;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t x{ 0u };
    for (uint32_t i{ 0u }; i < algo::progpow::DAG_LOADS; ++i)
    {
        if (i != 0u)
        {
            x = dst[dstCnt % regs];
            ++dstCnt;
        }
        uint32_t const sel{ algo::kiss99(round) };
        algo::progpow::nvidia::writeSequenceMergeEntries(ss, i, x, sel);
    }

    ss << "}" << "\n";

    ofs << ss.str();

    SAFE_DELETE_ARRAY(src);
    SAFE_DELETE_ARRAY(dst);
}

#endif
