"""Work-energy functions checked against independent quadrature and dynamics."""

import numpy as np
import pytest
from scipy.integrate import quad

from exobalancelab.paper import AUTHORS_PROFILES, exo_torque
from exobalancelab.work_energy import potential, zero_torque_velocity


@pytest.mark.parametrize("mode", ["none", "gc"])
def test_potential_matches_integrated_physical_work(mode):
    p = AUTHORS_PROFILES["YF"]
    for q in [-0.4, -0.15, -0.01, 0.01, 0.15, 0.4]:
        integral = quad(
            lambda x: -(p.stiffness_nm * np.sin(x) + exo_torque(x, 0, p, mode)),
            0,
            q,
            epsabs=1e-9,
            points=[
                x
                for x in [-np.arcsin(50 / p.stiffness_nm), np.arcsin(50 / p.stiffness_nm)]
                if min(0, q) < x < max(0, q)
            ],
        )[0]
        assert potential(q, p, mode) == pytest.approx(integral, abs=1e-9)
        speed = zero_torque_velocity(q, p, mode)
        assert 0.5 * p.inertia_kg_m2 * speed**2 + potential(q, p, mode) == pytest.approx(0, abs=1e-10)
