#if defined(__linux__)
    #include <experimental/filesystem>
    namespace __fs = std::experimental::filesystem;
#else
    #include <filesystem>
    namespace __fs = std::filesystem;
#endif
#include <fstream>

#include <algo/crypto/kiss99.hpp>
#include <algo/progpow/progpow.hpp>
#include <common/cast.hpp>
#include <common/custom.hpp>


static
void writeSequenceMergeEntries(
    std::stringstream& ss,
    uint32_t const i,
    uint32_t const x,
    uint32_t const sel)
{
    ////////////////////////////////////////////////////////////////////////////
    ss << "\t" << "// iter[" << i << "] merge_entries " << sel % 4 << "\n";

    ////////////////////////////////////////////////////////////////////////////
    std::string l { "hash[" + std::to_string(x) + "]" };
    std::string ret { "hash[" + std::to_string(x) + "] = " };

    ////////////////////////////////////////////////////////////////////////////
    std::string r;
    switch (i)
    {
        case 0: r = "entries.x"; break;
        case 1: r = "entries.y"; break;
        case 2: r = "entries.z"; break;
        case 3: r = "entries.w"; break;
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


static
void writeSequenceMathMerge(
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
    std::string l { "hash[" + std::to_string(src1) + "]" };
    std::string r { "hash[" + std::to_string(src2) + "]" };
    std::string ret { "data = " };

    ////////////////////////////////////////////////////////////////////////////
    switch (sel_math % 11)
    {
        case 0u:  ss << "\t" << ret << l << " + " << r << ";\n"; break;
        case 1u:  ss << "\t" << ret << l << " * " << r << ";\n"; break;
        case 2u:  ss << "\t" << ret << "mul_hi(" << l << ", " << r << ")" << ";\n"; break;
        case 3u:  ss << "\t" << ret << "min(" << l << ", " << r << ")" << ";\n"; break;
        case 4u:  ss << "\t" << ret << "rol_u32(" << l << ", " << r << ")" << ";\n"; break;
        case 5u:  ss << "\t" << ret << "ror_u32(" << l << ", " << r << ")" << ";\n"; break;
        case 6u:  ss << "\t" << ret << l << " & " << r << ";\n"; break;
        case 7u:  ss << "\t" << ret << l << " | " << r << ";\n"; break;
        case 8u:  ss << "\t" << ret << l << " ^ " << r << ";\n"; break;
        case 9u:  ss << "\t" << ret << "clz(" << l << ") + clz(" << r << ")" << ";\n"; break;
        case 10u: ss << "\t" << ret << "popcount(" << l << ") + popcount(" << r << ")" << ";\n"; break;
    }
    l = "hash[" + std::to_string(dst) + "]";
    r = "data";
    ret = "hash[" + std::to_string(dst) + "] = ";

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


static
void writeSequenceMergeCache(
    std::stringstream& ss,
    uint32_t const i,
    uint32_t const src,
    uint32_t const dst,
    uint32_t const sel)
{
    ////////////////////////////////////////////////////////////////////////////
    ss << "\t" << "// iter[" << i << "] merge " << sel % 4 << "\n";
    ss << "\t" << "dag_offset = hash[" << src << "] & " << (algo::progpow::MODULE_CACHE - 1) << "u" << ";\n";

    ////////////////////////////////////////////////////////////////////////////
    std::string const l { "hash[" + std::to_string(dst) + "]" };
    std::string const r { "header_dag[dag_offset]" };
    std::string const ret { "hash[" + std::to_string(dst) + "] = " };

    ////////////////////////////////////////////////////////////////////////////
    switch (sel % 4u)
    {
        case 0u: ss << "\t" << ret << "(" << l << " * 33) + " << r << ";\n"; break;
        case 1u: ss << "\t" << ret << "(" << l << " ^ " << r << ") * 33" << ";\n"; break;
        case 2u: ss << "\t" << ret << "rol_u32(" << l << ", " << ((sel >> 16) % 31) + 1 << ") ^ " << r << ";\n"; break;
        case 3u: ss << "\t" << ret << "ror_u32(" << l << ", " << ((sel >> 16) % 31) + 1 << ") ^ " << r << ";\n";  break;
    }
}


void algo::progpow::writeMathRandomKernelOpenCL(
    uint32_t const deviceId,
    uint64_t const period,
    uint32_t const countCache,
    uint32_t const countMath)
{
    using namespace std::string_literals;

    ////////////////////////////////////////////////////////////////////////////
    __fs::path pathSequenceMath { "kernel" };
    pathSequenceMath /= "progpow";
    pathSequenceMath /=
        "sequence_math_random"s
        + "_"s + std::to_string(deviceId)
        + "_"s + std::to_string(period)
        + ".cl"s;
    __fs::create_directories(pathSequenceMath.parent_path());
    std::ofstream ofs{ pathSequenceMath };

    ////////////////////////////////////////////////////////////////////////////
    int32_t dst[algo::progpow::REGS]{};
    int32_t src[algo::progpow::REGS]{};
    algo::Kiss99Properties round { algo::progpow::initializeRound(period, dst, src) };

    ////////////////////////////////////////////////////////////////////////////
    std::stringstream ss;
    ss << "// period " << std::to_string(period) << "\n"
        << "inline" << "\n"
        << "void sequence_dynamic(" << "\n"
        << "\t" << "__local uint const* const restrict header_dag, " << "\n"
        << "\t" <<"uint* const restrict hash, " << "\n"
        << "\t" <<"uint4 entries)" << "\n"
        << "{" << "\n"
        << "\t" << "uint dag_offset;" << "\n"
        << "\t" << "uint data;" << "\n"
        ;

    ////////////////////////////////////////////////////////////////////////////
    uint32_t dstCnt{ 0u };
    uint32_t srcCnt{ 0u };
    uint32_t const max{ countCache > countMath ? countCache : countMath };
    for (uint32_t i{ 0u }; i < max; ++i)
    {

        ////////////////////////////////////////////////////////////////////////////
        if (i < countCache)
        {
            int32_t  const srcValue{ src[srcCnt % algo::progpow::REGS] };
            int32_t  const dstValue{ dst[dstCnt % algo::progpow::REGS] };
            uint32_t const sel{ algo::kiss99(round) };
            ++srcCnt;
            ++dstCnt;

            writeSequenceMergeCache(ss, i, srcValue, dstValue, sel);
        }

        ////////////////////////////////////////////////////////////////////////////
        if (i < countMath)
        {
            uint32_t const srcRnd{ algo::kiss99(round) % algo::progpow::MODULE_SOURCE };
            uint32_t const src1{ srcRnd % algo::progpow::REGS };
            uint32_t       src2{ srcRnd / algo::progpow::REGS };
            uint32_t const sel1{ algo::kiss99(round) };
            uint32_t const sel2{ algo::kiss99(round) };
            int32_t  const dstValue{ dst[dstCnt % algo::progpow::REGS] };

            if (src2 >= src1)
            {
                ++src2;
            }
            ++dstCnt;

            writeSequenceMathMerge(ss, i, castU32(dstValue), src1, src2, sel1, sel2);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    uint32_t x{ 0u };
    for (uint32_t i{ 0u }; i < algo::progpow::DAG_LOADS; ++i)
    {
        if (i != 0u)
        {
            x = dst[dstCnt % algo::progpow::REGS];
            ++dstCnt;
        }
        uint32_t const sel{ algo::kiss99(round) };
        writeSequenceMergeEntries(ss, i, x, sel);
    }

    ss << "}" << "\n";

    ofs << ss.str();
}
