#ifndef XCHAOS_NORM_TO_PHYS_H
#define XCHAOS_NORM_TO_PHYS_H

/*gpukern*/
void norm_to_phys(ParticlesData part, NormedParticlesData norm_part, const int64_t nelem)
{
    const double gemitt_x = NormedParticlesData_get_twiss_data(norm_part, 0) / ParticlesData_get_beta0(part, 0) / ParticlesData_get_gamma0(part, 0);
    const double gemitt_y = NormedParticlesData_get_twiss_data(norm_part, 1) / ParticlesData_get_beta0(part, 0) / ParticlesData_get_gamma0(part, 0);
    // if twiss_data[8] is not nan, evaluate gemitt_z, otherwise it is 1
    const double gemitt_z = isnan(NormedParticlesData_get_twiss_data(norm_part, 8)) ? 1.0 : NormedParticlesData_get_twiss_data(norm_part, 8) / ParticlesData_get_beta0(part, 0) / ParticlesData_get_gamma0(part, 0);

    const double* w = NormedParticlesData_getp1_w(norm_part, 0);

    for (int ii = 0; ii < nelem; ii++){ // vectorize_over ii nelem
        double x_norm = NormedParticlesData_get_x_norm(norm_part, ii) * sqrt(gemitt_x);
        double px_norm = NormedParticlesData_get_px_norm(norm_part, ii) * sqrt(gemitt_x);
        double y_norm = NormedParticlesData_get_y_norm(norm_part, ii) * sqrt(gemitt_y);
        double py_norm = NormedParticlesData_get_py_norm(norm_part, ii) * sqrt(gemitt_y);
        double zeta_norm = NormedParticlesData_get_zeta_norm(norm_part, ii) * sqrt(gemitt_z);
        double pzeta_norm = NormedParticlesData_get_pzeta_norm(norm_part, ii) * sqrt(gemitt_z);

        // need to be careful by setting up a matrix dot operation
        // by hand with the flatten w matrix

        ParticlesData_set_x(part, ii, 
            (w[0] * x_norm + w[1] * px_norm + w[2] * y_norm + w[3] * py_norm + w[4] * zeta_norm + w[5] * pzeta_norm) + NormedParticlesData_get_twiss_data(norm_part, 2)
        );
        ParticlesData_set_px(part, ii, 
            (w[6] * x_norm + w[7] * px_norm + w[8] * y_norm + w[9] * py_norm + w[10] * zeta_norm + w[11] * pzeta_norm) + NormedParticlesData_get_twiss_data(norm_part, 3)
        );
        ParticlesData_set_y(part, ii, 
            (w[12] * x_norm + w[13] * px_norm + w[14] * y_norm + w[15] * py_norm + w[16] * zeta_norm + w[17] * pzeta_norm) + NormedParticlesData_get_twiss_data(norm_part, 4)
        );
        ParticlesData_set_py(part, ii, 
            (w[18] * x_norm + w[19] * px_norm + w[20] * y_norm + w[21] * py_norm + w[22] * zeta_norm + w[23] * pzeta_norm) + NormedParticlesData_get_twiss_data(norm_part, 5)
        );
        ParticlesData_set_zeta(part, ii, 
            (w[24] * x_norm + w[25] * px_norm + w[26] * y_norm + w[27] * py_norm + w[28] * zeta_norm + w[29] * pzeta_norm) + NormedParticlesData_get_twiss_data(norm_part, 6)
        );
        ParticlesData_set_ptau(part, ii, 
            (w[30] * x_norm + w[31] * px_norm + w[32] * y_norm + w[33] * py_norm + w[34] * zeta_norm + w[35] * pzeta_norm) * ParticlesData_get_beta0(part, ii) + NormedParticlesData_get_twiss_data(norm_part, 7)
        );

    } // end_vectorize
}

#endif /* XCHAOS_NORM_TO_PHYS_H */