#ifndef XNLBD_RENORM_DIST_H
#define XNLBD_RENORM_DIST_H

/*gpukern*/
void renorm_dist(ParticlesData ref_part, ParticlesData part, GhostParticleManagerData manager, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        int64_t idx_a = GhostParticleManagerData_get_argsort_a(manager, ii);
        int64_t idx_b = GhostParticleManagerData_get_argsort_b(manager, ii);
        double module = GhostParticleManagerData_get_target_module(manager);

        // if the state of the particle is not valid, skip it
        if (ParticlesData_get_state(part, idx_b) <= 0){}
        else{
            ParticlesData_set_x(part, idx_b, ParticlesData_get_x(ref_part, idx_a) + GhostParticleManagerData_get_displacement_x(manager, ii) * module);
            ParticlesData_set_px(part, idx_b, ParticlesData_get_px(ref_part, idx_a) + GhostParticleManagerData_get_displacement_px(manager, ii) * module);
            ParticlesData_set_y(part, idx_b, ParticlesData_get_y(ref_part, idx_a) + GhostParticleManagerData_get_displacement_y(manager, ii) * module);
            ParticlesData_set_py(part, idx_b, ParticlesData_get_py(ref_part, idx_a) + GhostParticleManagerData_get_displacement_py(manager, ii) * module);
            ParticlesData_set_zeta(part, idx_b, ParticlesData_get_zeta(ref_part, idx_a) + GhostParticleManagerData_get_displacement_zeta(manager, ii) * module);
            ParticlesData_set_ptau(part, idx_b, ParticlesData_get_ptau(ref_part, idx_a) + GhostParticleManagerData_get_displacement_ptau(manager, ii) * module);
}
    } // end_vectorize
}

#endif /* XNLBD_RENORM_DIST_H */