#ifndef XNLBD_PHYS_TO_NORM_H
#define XNLBD_PHYS_TO_NORM_H

/*gpukern*/
void phys_to_norm(ParticlesData part, NormedParticlesData norm_part, const int64_t nelem)
{
    const double gemitt_x = NormedParticlesData_get_twiss_data(norm_part, 0) / ParticlesData_get_beta0(part, 0) / ParticlesData_get_gamma0(part, 0);
    const double gemitt_y = NormedParticlesData_get_twiss_data(norm_part, 1) / ParticlesData_get_beta0(part, 0) / ParticlesData_get_gamma0(part, 0);
    // if twiss_data[8] is not nan, evaluate gemitt_z, otherwise it is 1
    const double gemitt_z = isnan(NormedParticlesData_get_twiss_data(norm_part, 8)) ? 1.0 : NormedParticlesData_get_twiss_data(norm_part, 8) / ParticlesData_get_beta0(part, 0) / ParticlesData_get_gamma0(part, 0);

    const double *w_inv = NormedParticlesData_getp1_w_inv(norm_part, 0);

    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        double x_norm = ParticlesData_get_x(part, ii) - NormedParticlesData_get_twiss_data(norm_part, 2);
        double px_norm = ParticlesData_get_px(part, ii) - NormedParticlesData_get_twiss_data(norm_part, 3);
        double y_norm = ParticlesData_get_y(part, ii) - NormedParticlesData_get_twiss_data(norm_part, 4);
        double py_norm = ParticlesData_get_py(part, ii) - NormedParticlesData_get_twiss_data(norm_part, 5);
        double zeta_norm = ParticlesData_get_zeta(part, ii) - NormedParticlesData_get_twiss_data(norm_part, 6);
        double pzeta_norm = (ParticlesData_get_ptau(part, ii) - NormedParticlesData_get_twiss_data(norm_part, 7)) / ParticlesData_get_beta0(part, ii);

        // need to be careful by setting up a matrix dot operation
        // by hand with the flatten w_inv matrix

        NormedParticlesData_set_x_norm(norm_part, ii,
                                       (w_inv[0] * x_norm + w_inv[1] * px_norm + w_inv[2] * y_norm + w_inv[3] * py_norm + w_inv[4] * zeta_norm + w_inv[5] * pzeta_norm) / sqrt(gemitt_x));
        NormedParticlesData_set_px_norm(norm_part, ii,
                                        (w_inv[6] * x_norm + w_inv[7] * px_norm + w_inv[8] * y_norm + w_inv[9] * py_norm + w_inv[10] * zeta_norm + w_inv[11] * pzeta_norm) / sqrt(gemitt_x));
        NormedParticlesData_set_y_norm(norm_part, ii,
                                       (w_inv[12] * x_norm + w_inv[13] * px_norm + w_inv[14] * y_norm + w_inv[15] * py_norm + w_inv[16] * zeta_norm + w_inv[17] * pzeta_norm) / sqrt(gemitt_y));
        NormedParticlesData_set_py_norm(norm_part, ii,
                                        (w_inv[18] * x_norm + w_inv[19] * px_norm + w_inv[20] * y_norm + w_inv[21] * py_norm + w_inv[22] * zeta_norm + w_inv[23] * pzeta_norm) / sqrt(gemitt_y));
        NormedParticlesData_set_zeta_norm(norm_part, ii,
                                          (w_inv[24] * x_norm + w_inv[25] * px_norm + w_inv[26] * y_norm + w_inv[27] * py_norm + w_inv[28] * zeta_norm + w_inv[29] * pzeta_norm) / sqrt(gemitt_z));
        NormedParticlesData_set_pzeta_norm(norm_part, ii,
                                           (w_inv[30] * x_norm + w_inv[31] * px_norm + w_inv[32] * y_norm + w_inv[33] * py_norm + w_inv[34] * zeta_norm + w_inv[35] * pzeta_norm) / sqrt(gemitt_z));

        // also copy the particle_id and state
        NormedParticlesData_set_particle_id(
            norm_part, ii, ParticlesData_get_particle_id(part, ii));
        NormedParticlesData_set_state(
            norm_part, ii, ParticlesData_get_state(part, ii));

    } // end_vectorize
}

#endif /* XNLBD_PHYS_TO_NORM_H */