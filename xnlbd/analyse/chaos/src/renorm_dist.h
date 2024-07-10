#ifndef XNLBD_RENORM_DIST_H
#define XNLBD_RENORM_DIST_H

/*gpukern*/
void renorm_dist(ParticlesData part, GhostParticleManagerData manager, const int64_t nelem)
{
    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        int64_t idx_b = GhostParticleManagerData_get_argsort_b(manager, ii);
        double module = GhostParticleManagerData_get_module(manager, ii);

        // if the state of the particle is not valid, skip it
        if (ParticlesData_get_state(part, idx_b) <= 1){}
        else{
            ParticlesData_set_x(part, idx_b, ParticlesData_get_x(part, idx_b) + GhostParticleManagerData_get_displacement_x(manager, ii) * module);
            ParticlesData_set_px(part, idx_b, ParticlesData_get_px(part, idx_b) + GhostParticleManagerData_get_displacement_px(manager, ii) * module);
            ParticlesData_set_y(part, idx_b, ParticlesData_get_y(part, idx_b) + GhostParticleManagerData_get_displacement_y(manager, ii) * module);
            ParticlesData_set_py(part, idx_b, ParticlesData_get_py(part, idx_b) + GhostParticleManagerData_get_displacement_py(manager, ii) * module);
            ParticlesData_set_zeta(part, idx_b, ParticlesData_get_zeta(part, idx_b) + GhostParticleManagerData_get_displacement_zeta(manager, ii) * module);
            ParticlesData_set_ptau(part, idx_b, ParticlesData_get_ptau(part, idx_b) + GhostParticleManagerData_get_displacement_ptau(manager, ii) * module);
        }
    } // end_vectorize
}

#endif /* XNLBD_RENORM_DIST_H */