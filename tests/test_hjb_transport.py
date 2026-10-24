"""Independent checks of the transport reference and numerical operators."""

import os

import numpy as np
import pytest
from scipy.optimize import linprog

from examples.benchmark_transport import constrained_bounds, triple_bounds

jax_only = pytest.mark.skipif(
    os.environ.get("EXOBALANCE_JAX_TEST") != "1", reason="Separate JAX environment is opt-in"
)


def test_constrained_reference_matches_independent_control_optimization():
    duration, count, cap = 0.5, 400, 0.3
    dt = duration / count
    weights = (duration - (np.arange(count) + 0.5) * dt) * dt
    cumulative = dt * np.tril(np.ones((count, count)))
    constraints = np.concatenate([cumulative, -cumulative, np.full((1, count), dt), np.full((1, count), -dt)])
    for velocity in [-0.3, -0.23, 0, 0.17, 0.3]:
        limits = np.r_[
            np.full(count, cap - velocity), np.full(count, cap + velocity), 0.01 - velocity, 0.01 + velocity
        ]
        low = linprog(weights, A_ub=constraints, b_ub=limits, bounds=(-1, 1), method="highs")
        high = linprog(-weights, A_ub=constraints, b_ub=limits, bounds=(-1, 1), method="highs")
        assert low.success and high.success
        lower, upper, feasible = constrained_bounds(velocity)
        assert feasible
        np.testing.assert_allclose(
            [lower, upper],
            [-0.005 - velocity * duration + high.fun, 0.005 - velocity * duration - low.fun],
            atol=dt**2 / 2,
            rtol=0,
        )
    assert not constrained_bounds(0.30001)[2]


def test_three_state_projection_matches_independent_control_optimization():
    duration, count = 0.5, 800
    times = np.linspace(0, duration, count + 1)
    # Backward linear dynamics, with terminal speed/acceleration as independent
    # optimization variables. Piecewise constant jerk is bounded in [-1,1].
    cost = np.r_[-duration, duration**2 / 2, -np.diff(times**3) / 6]
    velocity_constraint = np.r_[1, -duration, np.diff(times**2) / 2][None, :]
    bounds = [(-0.01, 0.01), (-0.02, 0.02)] + [(-1, 1)] * count
    for velocity in [-0.14, -0.12, -0.03, 0, 0.1, 0.13, 0.144]:
        minimum = linprog(cost, A_eq=velocity_constraint, b_eq=[velocity], bounds=bounds, method="highs")
        maximum = linprog(-cost, A_eq=velocity_constraint, b_eq=[velocity], bounds=bounds, method="highs")
        assert minimum.success and maximum.success
        lower, upper, feasible = triple_bounds(velocity)
        assert feasible
        np.testing.assert_allclose(
            [lower, upper],
            [-0.005 + minimum.fun, 0.005 - maximum.fun],
            atol=duration * (duration / count) ** 2 / 4,
            rtol=0,
        )
    assert not triple_bounds(0.146)[2]


@jax_only
def test_large_control_intervals_include_interior_minima():
    import jax.numpy as jnp

    from exobalancelab.hjb_transport import interval_minimum

    random = np.random.default_rng(20261023)
    values = random.uniform(-1, 1, (3, 5, 33))
    lower = random.uniform(-10, 32, 33)
    upper = lower + random.uniform(0, 40, 33)
    expected = np.full(values.shape, np.inf)
    for k, (lo, hi) in enumerate(zip(lower, upper)):
        indices = np.arange(33)[(np.arange(33) >= lo) & (np.arange(33) <= hi)]
        if len(indices):
            expected[..., k] = values[..., indices].min(axis=-1)
    actual = interval_minimum(jnp.array(values), jnp.array(lower), jnp.array(upper))
    np.testing.assert_array_equal(actual, expected)


@jax_only
def test_enclosure_extension_is_continuous_positive_and_has_no_constant_jump():
    import jax.numpy as jnp

    from exobalancelab.hjb_transport import interpolate

    # The zero set is inside the old enclosure. Its exterior must connect
    # continuously to the positive endpoint, without an O(1) value jump.
    values = jnp.array([[[0.004, 0.002, -0.001, 0.002, 0.004]]])
    query = jnp.array([[[-1e-8, -0.5, 0, 4, 4 + 1e-8, 4.5]]])
    actual = np.asarray(interpolate(values, query, 2, True, True)).ravel()
    np.testing.assert_allclose(
        actual, [0.004 + 0.002e-8, 0.005, 0.004, 0.004, 0.004 + 0.002e-8, 0.005], atol=2e-15
    )
    assert np.all(actual > 0)


@jax_only
@pytest.mark.parametrize(
    "name,mode,mt,mrtd", [("YF", "none", 0.5, 1), ("YF", "gc", 1, 0.5), ("OM", "gc", 1, 1)]
)
def test_transport_physics_matches_independent_paper_implementation(name, mode, mt, mrtd):
    from dataclasses import asdict

    from exobalancelab.hjb_transport import paper_problem, physical_fields
    from exobalancelab.paper import (
        AUTHORS_PROFILES,
        acceleration,
        exo_torque,
        ground_reaction,
        scale_strength,
    )

    p = scale_strength(AUTHORS_PROFILES[name], mt, mrtd)
    data = asdict(p)
    data.pop("name")
    problem = paper_problem(
        dict(profile=name, effective_profile=data, exo=mode, method="none"),
        np.array([0.5, -3, -150]),
        np.array([2.5, 3, 100]),
    )
    random = np.random.default_rng(623)
    theta = random.uniform(0.6, 2.4, 300)
    speed = random.uniform(-2.5, 2.5, 300)
    torque = random.uniform(p.torque_min_nm - 10, p.torque_max_nm + 10, 300)
    actual, obstacle = physical_fields(problem, [theta, speed, torque])
    expected, feasible = [], []
    for q, v, t in zip(theta - np.pi / 2, speed, torque):
        device = exo_torque(q, v, p, mode)
        expected.append(acceleration(q, v, t + device, p))
        reaction = ground_reaction(q, v, t + device, p)
        feasible.append(
            reaction["normal_n"] >= 0
            and reaction["cop_margin_m"] >= 0
            and reaction["friction_margin_n"] >= 0
            and p.torque_min_nm <= t <= p.torque_max_nm
        )
    np.testing.assert_allclose(actual[0], speed, atol=1e-13, rtol=0)
    np.testing.assert_allclose(actual[1], expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(np.asarray(obstacle) <= 0, feasible)


@jax_only
@pytest.mark.parametrize("kind", ["triple_integrator", "paper_none", "paper_gc"])
def test_slab_execution_matches_full_volume_operators_at_partial_slabs(kind):
    import jax
    import jax.numpy as jnp

    from exobalancelab.hjb_transport import (
        SPLIT_STAGES,
        Problem,
        angle,
        angle_axis,
        control,
        control_axis,
        initial_value,
        integrate,
        velocity,
        velocity_axis,
    )

    problem = Problem(
        *map(jnp.asarray, [[-0.1, -0.3, -0.75], [0.1, 0.3, 0.75], [np.inf], [-1.0, 1.0]]),
        kind="triple_integrator",
    )
    center = np.zeros(3)
    if kind.startswith("paper"):
        from dataclasses import asdict

        from exobalancelab.hjb_transport import paper_problem
        from exobalancelab.paper import AUTHORS_PROFILES

        profile = asdict(AUTHORS_PROFILES["OM"])
        profile.pop("name")
        problem = paper_problem(
            dict(profile="OM", effective_profile=profile, exo=kind.split("_")[1], method="none"),
            np.array([0.6, -3, -150]),
            np.array([2.5, 3, 100]),
        )
        center[0] = np.pi / 2
    shape = (35, 11, 33)
    widths = np.array([0.005, 0.01, 0.02])
    axes = tuple(jnp.linspace(c - 2 * w, c + 2 * w, n) for c, w, n in zip(center, widths, shape))
    value = 0.005 * (
        jnp.maximum(
            jnp.maximum(
                abs(axes[0][:, None, None] - center[0]) / widths[0], abs(axes[1][None, :, None]) / widths[1]
            ),
            abs(axes[2][None, None, :]) / widths[2],
        )
        - 1
    )
    expected = initial_value(axes, problem, [(center, widths, 0.005)])
    np.testing.assert_allclose(expected, value, atol=1e-14, rtol=0)
    expected_axes = axes
    grid_steps = {control: control_axis, angle: angle_axis, velocity: velocity_axis}
    # Compare the same operator boundaries with one full-volume call. Fusing
    # all five stages on CPU can perturb ENO stencil ties through roundoff.
    for operator, fraction in SPLIT_STAGES:
        destination = jax.jit(grid_steps[operator])(expected_axes, problem, 0.005 * fraction)
        expected, expected_axes = jax.jit(operator)(
            expected, expected_axes, problem, 0.005 * fraction, destination
        )
    actual_axes, actual = integrate(problem, [(center, widths, 0.005)], shape, 0.005, 0.005)
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)
    np.testing.assert_array_equal(actual <= 0, np.asarray(expected) <= 0)
    for a, b in zip(actual_axes, expected_axes):
        np.testing.assert_array_equal(a, b)


@jax_only
def test_history_observer_does_not_change_transport_and_records_every_step():
    from exobalancelab.hjb_transport import analytic_problem, integrate

    boxes = [(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.005, 0.01]), 0.005)]
    captured = []

    def snapshot(time, axes, value):
        captured.append((time, [np.array(a) for a in axes], np.array(value)))

    args = (analytic_problem(), boxes, (3, 13, 17), 0.03, 0.01)
    a, expected = integrate(*args)
    b, actual = integrate(*args, snapshot=snapshot)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose([c[0] for c in captured], [0, 0.01, 0.02, 0.03], atol=1e-15)
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    np.testing.assert_array_equal(captured[-1][2], actual)
    assert not np.array_equal(captured[0][1][2], captured[-1][1][2])


@jax_only
def test_device_feedback_codes_match_native_derivative_signs():
    from examples.export_transport_policy import device_switching_codes
    from exobalancelab.transport_policy import switching_codes

    values = np.random.default_rng(240805418).normal(size=(35, 13, 17))
    values[:, 0, :] = 0
    values[:, 1, :] = np.arange(17)
    values[:, 2, :] = -np.arange(17)
    np.testing.assert_array_equal(device_switching_codes(values), switching_codes(values))
