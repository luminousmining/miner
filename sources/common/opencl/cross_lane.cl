///////////////////////////////////////////////////////////////////////////////
// WAVEFRONT 32
///////////////////////////////////////////////////////////////////////////////
#if WAVEFRONT == 32
    inline
    uint reg_load(
        uint const var,
        uint const lane_id,
        uint const width)
    {
#if defined CROSSLANE_BY_LDS
        uint const gap = get_thread_id_2d() / width;
        uint const value = sub_group_shuffle(var, gap + lane_id);

        return value;
#else
        uint const local_id = get_sub_group_local_id();
        uint const group_id = local_id / width;

        uint const val_group_0 = sub_group_broadcast(var, lane_id);
        uint const val_group_1 = sub_group_broadcast(var, width + lane_id);

        return (group_id == 0u) ? val_group_0 : val_group_1;
#endif
    }
#endif // WAVEFRONT == 32


///////////////////////////////////////////////////////////////////////////////
// WAVEFRONT 64
///////////////////////////////////////////////////////////////////////////////
#if WAVEFRONT == 64
    inline
    uint reg_load(
        uint const var,
        uint const lane_target,
        uint const width)
    {
#if defined CROSSLANE_BY_LDS
        uint const gap = get_thread_id_2d() / width;
        uint const value = sub_group_shuffle(var, gap + lane_id);

        return value;
#else
        uint const local_id = get_sub_group_local_id();
        uint const group_id = local_id / width;

        uint const val_0 = sub_group_broadcast(var, lane_target);
        uint const val_1 = sub_group_broadcast(var, width + lane_target);
        uint const val_2 = sub_group_broadcast(var, 2u * width + lane_target);
        uint const val_3 = sub_group_broadcast(var, 3u * width + lane_target);

        uint result;
        switch (group_id)
        {
            case 0u: result = val_0; break;
            case 1u: result = val_1; break;
            case 2u: result = val_2; break;
            default: result = val_3; break;
        }

        return result;
#endif
    }
#endif // WAVEFRONT == 64
