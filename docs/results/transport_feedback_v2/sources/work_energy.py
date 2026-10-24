"""SI work-energy diagnostic for the paper's unconstrained, undamped pendulum.

Uses E=I*v²/2+U(q). Ground contact and torque-rate constraints are deliberately
absent from this diagnostic, so its curves are not a feasibility certificate.
"""

import numpy as np

from exobalancelab.paper import Profile


def potential(q, profile: Profile, assistance: str):
    """Potential relative to upright; -dU/dq equals gravity plus device torque."""
    q = np.asarray(q)
    stiffness = profile.stiffness_nm
    if assistance == "none":
        return stiffness * (np.cos(q) - 1)
    if assistance != "gc":
        raise ValueError("Conservative work-energy comparison supports none or gc")
    saturation = np.arcsin(50 / stiffness)
    return np.where(
        abs(q) <= saturation, 0, stiffness * (np.cos(q) - np.cos(saturation)) + 50 * (abs(q) - saturation)
    )


def zero_torque_velocity(q, profile: Profile, assistance: str):
    """Inward velocity with the same energy as the zero-torque standing set."""
    q = np.asarray(q)
    return -np.sign(q) * np.sqrt(
        np.maximum(0, -2 * potential(q, profile, assistance) / profile.inertia_kg_m2)
    )


def stopping_envelope(q, profile: Profile, assistance: str, heel: float, toe: float):
    """Relaxed stopping speeds under constant maximum opposing human torque.

    Final positions are prescribed standing positions. Rate is unbounded, so a
    final switch to the equilibrium torque is allowed. Return NaN where that
    final position is behind the proposed direction of motion or work is positive.
    """
    q = np.asarray(q)
    upper_work = potential(q, profile, assistance) - potential(toe, profile, assistance)
    upper_work += profile.torque_min_nm * (toe - q)
    lower_work = potential(q, profile, assistance) - potential(heel, profile, assistance)
    lower_work += profile.torque_max_nm * (heel - q)
    upper = np.where(
        (q <= toe) & (upper_work <= 0),
        np.sqrt(np.maximum(0, -2 * upper_work / profile.inertia_kg_m2)),
        np.nan,
    )
    lower = np.where(
        (q >= heel) & (lower_work <= 0),
        -np.sqrt(np.maximum(0, -2 * lower_work / profile.inertia_kg_m2)),
        np.nan,
    )
    return lower, upper
