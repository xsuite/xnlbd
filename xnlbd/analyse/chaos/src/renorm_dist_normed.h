#ifndef XNLBD_RENORM_DIST_NORMED_H
#define XNLBD_RENORM_DIST_NORMED_H

/*gpukern*/
void renorm_dist_normed(NormedParticlesData ref_part, NormedParticlesData part, GhostParticleManagerData manager, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        int64_t idx_a = GhostParticleManagerData_get_argsort_a(manager, ii);
        int64_t idx_b = GhostParticleManagerData_get_argsort_b(manager, ii);
        double module = GhostParticleManagerData_get_target_module(manager);

        // if the state of the particle is not valid, skip it
        if (NormedParticlesData_get_state(part, idx_b) <= 0){}
        else{
            NormedParticlesData_set_x_norm(part, idx_b, NormedParticlesData_get_x_norm(ref_part, idx_a) + GhostParticleManagerData_get_displacement_x_norm(manager, ii) * module);
            NormedParticlesData_set_px_norm(part, idx_b, NormedParticlesData_get_px_norm(ref_part, idx_a) + GhostParticleManagerData_get_displacement_px_norm(manager, ii) * module);
            NormedParticlesData_set_y_norm(part, idx_b, NormedParticlesData_get_y_norm(ref_part, idx_a) + GhostParticleManagerData_get_displacement_y_norm(manager, ii) * module);
            NormedParticlesData_set_py_norm(part, idx_b, NormedParticlesData_get_py_norm(ref_part, idx_a) + GhostParticleManagerData_get_displacement_py_norm(manager, ii) * module);
            NormedParticlesData_set_zeta_norm(part, idx_b, NormedParticlesData_get_zeta_norm(ref_part, idx_a) + GhostParticleManagerData_get_displacement_zeta_norm(manager, ii) * module);
            NormedParticlesData_set_pzeta_norm(part, idx_b, NormedParticlesData_get_pzeta_norm(ref_part, idx_a) + GhostParticleManagerData_get_displacement_pzeta_norm(manager, ii) * module);
        }
    } // end_vectorize
}

#endif /* XNLBD_RENORM_DIST_H */