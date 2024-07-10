#ifndef XCHAOS_GET_DISP_AND_DIR_H
#define XCHAOS_GET_DISP_AND_DIR_H

/*gpukern*/
void get_disp_and_dir(ParticlesData part_a, ParticlesData part_b, GhostParticleManagerData manager, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        // get argsort idx_a and idx_b from the ghost manager
        // here is the logic behind it:
        // (storage_a + storage_b[argsort(idx_b)[idx_a]])[argsort(idx_a)]
        int64_t idx_a = GhostParticleManagerData_get_idx_a(manager, ii);
        int64_t argsort_a = GhostParticleManagerData_get_argsort_a(manager, ii);
        int64_t idx_b = GhostParticleManagerData_get_argsort_b(manager, idx_a);

        // if either of the particles is invalid, skip it
        if (ParticlesData_get_state(part_a, ii) <= 1 || ParticlesData_get_state(part_b, idx_b) <= 1){}
        else{
            double displacement_x = ParticlesData_get_x(part_b, idx_b) - ParticlesData_get_x(part_a, ii);
            double displacement_px = ParticlesData_get_px(part_b, idx_b) - ParticlesData_get_px(part_a, ii);
            double displacement_y = ParticlesData_get_y(part_b, idx_b) - ParticlesData_get_y(part_a, ii);
            double displacement_py = ParticlesData_get_py(part_b, idx_b) - ParticlesData_get_py(part_a, ii);
            double displacement_zeta = ParticlesData_get_zeta(part_b, idx_b) - ParticlesData_get_zeta(part_a, ii);
            double displacement_ptau = ParticlesData_get_ptau(part_b, idx_b) - ParticlesData_get_ptau(part_a, ii);

            double module = sqrt(displacement_x * displacement_x + displacement_px * displacement_px + displacement_y * displacement_y + displacement_py * displacement_py + displacement_zeta * displacement_zeta + displacement_ptau * displacement_ptau);

            // set the info back to the ghost manager
            GhostParticleManagerData_set_module(manager, argsort_a, module);

            GhostParticleManagerData_set_displacement_x(manager, argsort_a, displacement_x / module);
            GhostParticleManagerData_set_displacement_px(manager, argsort_a, displacement_px / module);
            GhostParticleManagerData_set_displacement_y(manager, argsort_a, displacement_y / module);
            GhostParticleManagerData_set_displacement_py(manager, argsort_a, displacement_py / module);
            GhostParticleManagerData_set_displacement_zeta(manager, argsort_a, displacement_zeta / module);
            GhostParticleManagerData_set_displacement_ptau(manager, argsort_a, displacement_ptau / module);
        }
    } // end_vectorize
}

#endif /* XCHAOS_GET_DISP_AND_DIR_H */