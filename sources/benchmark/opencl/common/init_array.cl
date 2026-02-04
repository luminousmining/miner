__kernel
void init_array(
    __global uint* const dest,
    uint const size)
{
    for (uint i = 0u; i < size; ++i)
    {
        dest[i] = i;
    }
}
