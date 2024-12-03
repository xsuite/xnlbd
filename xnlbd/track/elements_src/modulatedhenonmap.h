#ifndef XNLBD_MODULATEDHENONMAP_H
#define XNLBD_MODULATEDHENONMAP_H

/*gpufun*/
void ModulatedHenonmap_track_local_particle(ModulatedHenonmapData el, LocalParticle* part0){
    int const n_turns = ModulatedHenonmapData_get_n_turns(el);
    int const n_par_multipoles = ModulatedHenonmapData_get_n_par_multipoles(el);

    int const n_fx_coeffs = ModulatedHenonmapData_get_n_fx_coeffs(el);
    int const n_fy_coeffs = ModulatedHenonmapData_get_n_fy_coeffs(el);

    double const alpha_x = ModulatedHenonmapData_get_twiss_params(el, 0);
    double const beta_x = ModulatedHenonmapData_get_twiss_params(el, 1);
    double const alpha_y = ModulatedHenonmapData_get_twiss_params(el, 2);
    double const beta_y = ModulatedHenonmapData_get_twiss_params(el, 3);
    double const sqrt_beta_x = sqrt(beta_x);
    double const sqrt_beta_y = sqrt(beta_y);
    double const domegax = ModulatedHenonmapData_get_domegax(el);
    double const domegay = ModulatedHenonmapData_get_domegay(el);
    double const dx = ModulatedHenonmapData_get_dx(el);
    double const ddx = ModulatedHenonmapData_get_ddx(el);

    int const norm = ModulatedHenonmapData_get_norm(el);
    
    
    //start_per_particle_block (part0->part)

        double x = LocalParticle_get_x(part);
        double px = LocalParticle_get_px(part);
        double y = LocalParticle_get_y(part);
        double py = LocalParticle_get_py(part);
        double delta = LocalParticle_get_delta(part);
        int at_turn = LocalParticle_get_at_turn(part);
        int at_turn_multipole = LocalParticle_get_at_turn(part);

        if(at_turn >= n_turns)
        {
            at_turn = at_turn % n_turns;
        }
        if(at_turn < 0)
        {
            do
            {
                at_turn += n_turns;
            }while(at_turn < 0);
        }

        if (at_turn_multipole >= n_par_multipoles)
        {
            at_turn_multipole = at_turn_multipole % n_par_multipoles;
        }
        if (at_turn_multipole < 0)
        {
            do
            {
                at_turn_multipole += n_par_multipoles;
            } while (at_turn_multipole < 0);
        }

        double const sin_omega_x = ModulatedHenonmapData_get_sin_omega_x(el, at_turn);
        double const cos_omega_x = ModulatedHenonmapData_get_cos_omega_x(el, at_turn);
        double const sin_omega_y = ModulatedHenonmapData_get_sin_omega_y(el, at_turn);
        double const cos_omega_y = ModulatedHenonmapData_get_cos_omega_y(el, at_turn);

        double x_hat, px_hat, y_hat, py_hat, x_hat_f, px_hat_f;
        if (norm)
        {
            x_hat = x;
            px_hat = px;
            y_hat = y;
            py_hat = py;
        }
        else
        {
            x_hat = x / sqrt_beta_x;
            px_hat = alpha_x * x / sqrt_beta_x + px * sqrt_beta_x;
            y_hat = y / sqrt_beta_y;
            py_hat = alpha_y * y / sqrt_beta_y + py * sqrt_beta_y;
        }
        x_hat_f = dx * delta / sqrt_beta_x;
        px_hat_f = alpha_x * dx * delta / sqrt_beta_x + ddx * delta * sqrt_beta_x;

        double const multipole_scale = 1.0 / (1.0 + delta);
    
        double curr_cos_omega_x, curr_sin_omega_x, curr_cos_omega_y, curr_sin_omega_y;
        if (domegax == 0)
        {
            curr_cos_omega_x = cos_omega_x;
            curr_sin_omega_x = sin_omega_x;
        }
        else
        {
            double const cos_domega_x = cos(domegax * delta);
            double const sin_domega_x = sin(domegax * delta);
            curr_cos_omega_x = cos_omega_x * cos_domega_x - sin_omega_x * sin_domega_x;
            curr_sin_omega_x = sin_omega_x * cos_domega_x + cos_omega_x * sin_domega_x;
        }
        if (domegay == 0)
        {
            curr_cos_omega_y = cos_omega_y;
            curr_sin_omega_y = sin_omega_y;
        }
        else
        {
            double const cos_domega_y = cos(domegay * delta);
            double const sin_domega_y = sin(domegay * delta);
            curr_cos_omega_y = cos_omega_y * cos_domega_y - sin_omega_y * sin_domega_y;
            curr_sin_omega_y = sin_omega_y * cos_domega_y + cos_omega_y * sin_domega_y;
        }

        #ifdef XSUITE_BACKTRACK
        x_hat -= x_hat_f;
        px_hat -= px_hat_f;
        double const x_hat_new = curr_cos_omega_x * x_hat - curr_sin_omega_x * px_hat;
        double const px_hat_new = curr_sin_omega_x * x_hat + curr_cos_omega_x * px_hat;
        double const y_hat_new = curr_cos_omega_y * y_hat - curr_sin_omega_y * py_hat;
        double const py_hat_new = curr_sin_omega_y * y_hat + curr_cos_omega_y * py_hat;
        x_hat = x_hat_new + x_hat_f;
        px_hat = px_hat_new + px_hat_f;
        y_hat = y_hat_new;
        py_hat = py_hat_new;
        #endif

        double fx = 0;
        for (int i = 0; i < n_fx_coeffs; i++)
        {
            double prod = ModulatedHenonmapData_get_fx_coeffs(el, n_fx_coeffs * at_turn_multipole + i) * multipole_scale;
            int x_power = ModulatedHenonmapData_get_fx_x_exps(el, i);
            int y_power = ModulatedHenonmapData_get_fx_y_exps(el, i);
            for (int j = 0; j < x_power; j++)
            {
                prod *= (sqrt_beta_x * x_hat);
            }
            for (int j = 0; j < y_power; j++)
            {
                prod *= (sqrt_beta_y * y_hat);
            }
            fx += prod;
        }
        double fy = 0;
        for (int i = 0; i < n_fy_coeffs; i++)
        {
            double prod = ModulatedHenonmapData_get_fy_coeffs(el, n_fy_coeffs * at_turn_multipole + i) * multipole_scale;
            int x_power = ModulatedHenonmapData_get_fy_x_exps(el, i);
            int y_power = ModulatedHenonmapData_get_fy_y_exps(el, i);
            for (int j = 0; j < x_power; j++)
            {
                prod *= (sqrt_beta_x * x_hat);
            }
            for (int j = 0; j < y_power; j++)
            {
                prod *= (sqrt_beta_y * y_hat);
            }
            fy += prod;
        }
        fx *= sqrt_beta_x;
        fy *= sqrt_beta_y;

        #ifdef XSUITE_BACKTRACK
        px_hat -= fx;
        py_hat -= fy;
        #else
        double const x_hat_new = (
            curr_cos_omega_x * (x_hat - x_hat_f) + 
            curr_sin_omega_x * (px_hat - px_hat_f + fx)
            ) + x_hat_f;
        double const px_hat_new = (
            -curr_sin_omega_x * (x_hat - x_hat_f) + 
            curr_cos_omega_x * (px_hat - px_hat_f + fx)
            ) + px_hat_f;
        double const y_hat_new = (
            curr_cos_omega_y * y_hat + 
            curr_sin_omega_y * (py_hat + fy));
        double const py_hat_new = (
            -curr_sin_omega_y * y_hat + 
            curr_cos_omega_y * (py_hat + fy));
        x_hat = x_hat_new;
        px_hat = px_hat_new;
        y_hat = y_hat_new;
        py_hat = py_hat_new;
        #endif

        if (norm)
        {
            x = x_hat;
            px = px_hat;
            y = y_hat;
            py = py_hat;
        }
        else
        {
            x = sqrt_beta_x * x_hat;
            px = -alpha_x * x_hat / sqrt_beta_x + px_hat / sqrt_beta_x;
            y = sqrt_beta_y * y_hat;
            py = -alpha_y * y_hat / sqrt_beta_y + py_hat / sqrt_beta_y;
        }

        LocalParticle_set_x(part, x);
        LocalParticle_set_px(part, px);
        LocalParticle_set_y(part, y);
        LocalParticle_set_py(part, py);

    //end_per_particle_block

}

#endif /* XNLBD_MODULATEDHENONMAP_H */