import numpy as np
import xpart as xp  # type: ignore
import xtrack as xt  # type: ignore
from scipy.special import factorial  # type: ignore
from xobjects.test_helpers import for_all_test_contexts  # type: ignore

from xnlbd.track import Henonmap, ModulatedHenonmap


@for_all_test_contexts
def test_modulatedhenonmap(test_context):
    n_turns = 100

    omega_x = 2 * np.pi * np.random.uniform(0.3310, 0.3320, n_turns)
    omega_y = 2 * np.pi * np.random.uniform(0.578, 0.582, n_turns)
    sin_omega_x = np.sin(omega_x)
    cos_omega_x = np.cos(omega_x)
    sin_omega_y = np.sin(omega_y)
    cos_omega_y = np.cos(omega_y)

    all_multipole_coeffs = []
    all_K3 = np.random.uniform(-20, -5, n_turns)
    for i in range(len(all_K3)):
        all_multipole_coeffs.append(
            [0.0721945, all_K3[i] * 3.0 * 0.0721945**2 * 104.0 / 2.0]
        )

    all_fx_coeffs = []
    all_fx_x_exps = []
    all_fx_y_exps = []
    all_fy_coeffs = []
    all_fy_x_exps = []
    all_fy_y_exps = []
    for i in range(len(all_K3)):
        multipole_coeffs = [0.0721945, all_K3[i] * 3.0 * 0.0721945**2 * 104.0 / 2.0]
        fx_coeffs = []
        fx_x_exps = []
        fx_y_exps = []
        fy_coeffs = []
        fy_x_exps = []
        fy_y_exps = []
        for n in range(2, len(multipole_coeffs) + 2):
            for k in range(0, n + 1):
                if (k % 4) == 0:
                    fx_coeffs.append(
                        multipole_coeffs[n - 2] / factorial(k) / factorial(n - k)
                    )
                    fx_x_exps.append(n - k)
                    fx_y_exps.append(k)
                elif (k % 4) == 2:
                    fx_coeffs.append(
                        -1 * multipole_coeffs[n - 2] / factorial(k) / factorial(n - k)
                    )
                    fx_x_exps.append(n - k)
                    fx_y_exps.append(k)
                elif (k % 4) == 1:
                    fy_coeffs.append(
                        -1 * multipole_coeffs[n - 2] / factorial(k) / factorial(n - k)
                    )
                    fy_x_exps.append(n - k)
                    fy_y_exps.append(k)
                else:
                    fy_coeffs.append(
                        multipole_coeffs[n - 2] / factorial(k) / factorial(n - k)
                    )
                    fy_x_exps.append(n - k)
                    fy_y_exps.append(k)
        all_fx_coeffs.append(fx_coeffs)
        all_fx_x_exps.append(fx_x_exps)
        all_fx_y_exps.append(fx_y_exps)
        all_fy_coeffs.append(fy_coeffs)
        all_fy_x_exps.append(fy_x_exps)
        all_fy_y_exps.append(fy_y_exps)

    line1 = xt.Line(
        elements=[
            Henonmap(
                omega_x=omega_x[0],
                omega_y=omega_y[0],
                n_turns=1,
                twiss_params=[0.0, 104.0, 0.0, 20.0],
                dqx=0.05,
                dqy=0,
                dx=0.1,
                ddx=0,
                multipole_coeffs=all_multipole_coeffs[0],
                norm=False,
            ),
            xt.Drift(length=0.0),
        ],
        element_names=["henon", "drift"],
    )
    line1.build_tracker(_context=test_context)

    line2 = xt.Line(
        elements=[
            ModulatedHenonmap(
                omega_x=omega_x,
                omega_y=omega_y,
                twiss_params=[0.0, 104.0, 0.0, 20.0],
                dqx=0.05,
                dqy=0,
                dx=0.1,
                ddx=0,
                multipole_coeffs=all_multipole_coeffs,
                norm=False,
            ),
            xt.Drift(length=0.0),
        ],
        element_names=["henon", "drift"],
    )
    line2.build_tracker(_context=test_context)

    N = 100
    x = np.linspace(0.0, 0.03, N)
    px = np.zeros(N)
    y = np.zeros(N)
    py = np.zeros(N)

    p1 = xp.Particles(x=x, px=px, y=y, py=py, p0c=4e11, _context=test_context)
    p1._init_random_number_generator()
    p2 = xp.Particles(x=x, px=px, y=y, py=py, p0c=4e11, _context=test_context)
    p2._init_random_number_generator()

    for n in range(0, n_turns, 1):
        line1["henon"].sin_omega_x = sin_omega_x[n]
        line1["henon"].cos_omega_x = cos_omega_x[n]
        line1["henon"].sin_omega_y = sin_omega_y[n]
        line1["henon"].cos_omega_y = cos_omega_y[n]
        line1["henon"].fx_coeffs = test_context.nparray_to_context_array(
            np.asarray(all_fx_coeffs[n])
        )
        line1["henon"].fx_x_exps = test_context.nparray_to_context_array(
            np.asarray(all_fx_x_exps[n])
        )
        line1["henon"].fx_y_exps = test_context.nparray_to_context_array(
            np.asarray(all_fx_y_exps[n])
        )
        line1["henon"].fy_coeffs = test_context.nparray_to_context_array(
            np.asarray(all_fy_coeffs[n])
        )
        line1["henon"].fy_x_exps = test_context.nparray_to_context_array(
            np.asarray(all_fy_x_exps[n])
        )
        line1["henon"].fy_y_exps = test_context.nparray_to_context_array(
            np.asarray(all_fy_y_exps[n])
        )
        line1.track(p1)

    line2.track(p2, num_turns=n_turns)

    x1 = test_context.nparray_from_context_array(p1.x)[
        np.argsort(test_context.nparray_from_context_array(p1.particle_id))
    ]
    px1 = test_context.nparray_from_context_array(p1.px)[
        np.argsort(test_context.nparray_from_context_array(p1.particle_id))
    ]
    y1 = test_context.nparray_from_context_array(p1.y)[
        np.argsort(test_context.nparray_from_context_array(p1.particle_id))
    ]
    py1 = test_context.nparray_from_context_array(p1.py)[
        np.argsort(test_context.nparray_from_context_array(p1.particle_id))
    ]

    x2 = test_context.nparray_from_context_array(p2.x)[
        np.argsort(test_context.nparray_from_context_array(p2.particle_id))
    ]
    px2 = test_context.nparray_from_context_array(p2.px)[
        np.argsort(test_context.nparray_from_context_array(p2.particle_id))
    ]
    y2 = test_context.nparray_from_context_array(p2.y)[
        np.argsort(test_context.nparray_from_context_array(p2.particle_id))
    ]
    py2 = test_context.nparray_from_context_array(p2.py)[
        np.argsort(test_context.nparray_from_context_array(p2.particle_id))
    ]

    assert np.all(
        [
            np.isclose(x1, x2, atol=1e-15, rtol=1e-10),
            np.isclose(px1, px2, atol=1e-15, rtol=1e-10),
            np.isclose(y1, y2, atol=1e-15, rtol=1e-10),
            np.isclose(py1, py2, atol=1e-15, rtol=1e-10),
        ]
    )
