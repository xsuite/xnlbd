#ifndef XNLBD_RANDOMNORMALKICK_H
#define XNLBD_RANDOMNORMALKICK_H

/*gpufun*/
void RandomNormalKick_track_local_particle(RandomNormalKickData el, LocalParticle *part0)
{
    //start_per_particle_block (part0->part)
        double r;
        if (RandomNormalKickData_get_x_flag(el))
        {
            r = RandomNormal_generate(part);
            LocalParticle_add_to_x(part,
                                r * RandomNormalKickData_get_x_module(el));
        }

        if (RandomNormalKickData_get_px_flag(el))
        {
            r = RandomNormal_generate(part);
            LocalParticle_add_to_px(part,
                                    r * RandomNormalKickData_get_px_module(el));
        }

        if (RandomNormalKickData_get_y_flag(el))
        {
            r = RandomNormal_generate(part);
            LocalParticle_add_to_y(part,
                                r * RandomNormalKickData_get_y_module(el));
        }

        if (RandomNormalKickData_get_py_flag(el))
        {
            r = RandomNormal_generate(part);
            LocalParticle_add_to_py(part,
                                    r * RandomNormalKickData_get_py_module(el));
        }

        if (RandomNormalKickData_get_zeta_flag(el))
        {
            r = RandomNormal_generate(part);
            LocalParticle_add_to_zeta(part,
                                    r * RandomNormalKickData_get_zeta_module(el));
        }

        if (RandomNormalKickData_get_ptau_flag(el))
        {
            r = RandomNormal_generate(part);
            LocalParticle_add_to_ptau(part,
                                    r * RandomNormalKickData_get_ptau_module(el));
        }
    //end_per_particle_block
}

#endif /* XNLBD_RANDOMNORMALKICK_H */
