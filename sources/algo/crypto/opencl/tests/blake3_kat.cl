// Test-only wrapper kernel for the shared BLAKE3 primitive. The host harness
// concatenates sources/algo/crypto/opencl/blake3.cl in front of this file before
// building, so blake3_hash_chunk is in scope (no OpenCL #include needed).
__kernel void blake3_kat(__global uchar const* in, uint const len, uint const outlen, __global uchar* out)
{
    uchar buf[1024];
    for (uint i = 0u; i < len; ++i)
    {
        buf[i] = in[i];
    }

    uchar res[64];
    blake3_hash_chunk(buf, len, outlen, res);

    for (uint i = 0u; i < outlen; ++i)
    {
        out[i] = res[i];
    }
}
