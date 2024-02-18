#include <algo/progpow/firopow.hpp>
#include <algo/progpow/progpow.hpp>


void algo::firopow::nvidia::writeSequenceMathMerge(
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
        case 0u:  ss << "\t" << ret << l << " + " << r << ";\n"; break;
        case 1u:  ss << "\t" << ret << l << " * " << r << ";\n"; break;
        case 2u:  ss << "\t" << ret << "__umulhi(" << l << ", " << r << ")" << ";\n"; break;
        case 3u:  ss << "\t" << ret << "min(" << l << ", " << r << ")" << ";\n"; break;
        case 4u:  ss << "\t" << ret << "rol_u32(" << l << ", " << r << ")" << ";\n"; break;
        case 5u:  ss << "\t" << ret << "ror_u32(" << l << ", " << r << ")" << ";\n"; break;
        case 6u:  ss << "\t" << ret << l << " & " << r << ";\n"; break;
        case 7u:  ss << "\t" << ret << l << " | " << r << ";\n"; break;
        case 8u:  ss << "\t" << ret << l << " ^ " << r << ";\n"; break;
        case 9u:  ss << "\t" << ret << "__clz(" << l << ") + __clz(" << r << ")" << ";\n"; break;
        case 10u: ss << "\t" << ret << "__popc(" << l << ") + __popc(" << r << ")" << ";\n"; break;
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


void algo::firopow::nvidia::writeSequenceMergeCache(
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


void algo::firopow::amd::writeSequenceMathMerge(
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


void algo::firopow::amd::writeSequenceMergeCache(
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
        case 3u: ss << "\t" << ret << "ror_u32(" << l << ", " << ((sel >> 16) % 31) + 1 << ") ^ " << r << ";\n"; break;
    }
}
