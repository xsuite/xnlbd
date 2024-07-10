#ifndef XCHAOS_GET_DISP_AND_DIR_NORMED_H
#define XCHAOS_GET_DISP_AND_DIR_NORMED_H

/*gpukern*/
void get_disp_and_dir_normed(NormedParticlesData part_a, NormedParticlesData part_b, GhostParticleManagerData manager, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        // get argsort idx_a and idx_b from the ghost manager
        // here is the logic behind it:
        // (storage_a + storage_b[argsort(idx_b)[idx_a]])[argsort(idx_a)]
        int64_t idx_a = GhostParticleManagerData_get_idx_a(manager, ii);
        int64_t argsort_a = GhostParticleManagerData_get_argsort_a(manager, ii);
        int64_t idx_b = GhostParticleManagerData_get_argsort_b(manager, idx_a);

        // if either of the particles is invalid, skip it
        if (NormedParticlesData_get_state(part_a, ii) <= 1 || NormedParticlesData_get_state(part_b, idx_b) <= 1){}
        else{
            double displacement_x_norm = NormedParticlesData_get_x_norm(part_b, idx_b) - NormedParticlesData_get_x_norm(part_a, ii);
            double displacement_px_norm = NormedParticlesData_get_px_norm(part_b, idx_b) - NormedParticlesData_get_px_norm(part_a, ii);
            double displacement_y_norm = NormedParticlesData_get_y_norm(part_b, idx_b) - NormedParticlesData_get_y_norm(part_a, ii);
            double displacement_py_norm = NormedParticlesData_get_py_norm(part_b, idx_b) - NormedParticlesData_get_py_norm(part_a, ii);
            double displacement_zeta_norm = NormedParticlesData_get_zeta_norm(part_b, idx_b) - NormedParticlesData_get_zeta_norm(part_a, ii);
            double displacement_pzeta_norm = NormedParticlesData_get_pzeta_norm(part_b, idx_b) - NormedParticlesData_get_pzeta_norm(part_a, ii);

            double module = sqrt(displacement_x_norm * displacement_x_norm + displacement_px_norm * displacement_px_norm + displacement_y_norm * displacement_y_norm + displacement_py_norm * displacement_py_norm + displacement_zeta_norm * displacement_zeta_norm + displacement_pzeta_norm * displacement_pzeta_norm);

            // set the info back to the ghost manager
            GhostParticleManagerData_set_module(manager, argsort_a, module);

            GhostParticleManagerData_set_displacement_x_norm(manager, argsort_a, displacement_x_norm / module);
            GhostParticleManagerData_set_displacement_px_norm(manager, argsort_a, displacement_px_norm / module);
            GhostParticleManagerData_set_displacement_y_norm(manager, argsort_a, displacement_y_norm / module);
            GhostParticleManagerData_set_displacement_py_norm(manager, argsort_a, displacement_py_norm / module);
            GhostParticleManagerData_set_displacement_zeta_norm(manager, argsort_a, displacement_zeta_norm / module);
            GhostParticleManagerData_set_displacement_pzeta_norm(manager, argsort_a, displacement_pzeta_norm / module);
        }
    } // end_vectorize
}

#endif /* XCHAOS_GET_DISP_AND_DIR_NORMED_H */