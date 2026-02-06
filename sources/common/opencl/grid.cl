#ifndef COMMON_OPENCL_GRID
#define COMMON_OPENCL_GRID

#ifndef OCL_DIM
    #error "OCL_DIM must be defined to 1, 2 or 3 before including common_opencl_grid.cl"
#endif


inline
uint get_thread_id_1d(void)
{
    return get_global_id(0);
}


inline
uint get_thread_id_2d(void)
{
    uint const gx = get_global_id(0);
    uint const gy = get_global_id(1);
    uint const sx = get_global_size(0);

    return gx + gy * sx;
}


inline
uint get_thread_id_3d()
{
    uint const gx = get_global_id(0);
    uint const gy = get_global_id(1);
    uint const gz = get_global_id(2);

    uint const sx = get_global_size(0);
    uint const sy = get_global_size(1);

    return gx + gy * sx + gz * sx * sy;
}


inline
uint get_thread_id(void)
{
#if OCL_DIM == 1
    return get_thread_id_1d;
#elif OCL_DIM == 2
    return get_thread_id_2d();
#elif OCL_DIM == 3
    return get_thread_id_3d();
#else
    #error "Unsupported OCL_DIM value"
#endif
}


#endif // COMMON_OPENCL_GRID
