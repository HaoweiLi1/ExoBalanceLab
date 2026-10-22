"""Raz et al. (2024), Eq. (3), Table I and foot constraints in SI units.

q = theta - pi/2; q>0 leans forward (world x>0), positive torque is dorsiflexion.
The paper's torque and angular velocity signs are unchanged by this angle offset.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Profile:
    """Table I anthropometry and bilateral MT/MRTD limits; negative PF, positive DF."""

    name: str
    mass_kg: float
    height_m: float
    torque_min_nm: float
    torque_max_nm: float
    rate_min_nm_s: float
    rate_max_nm_s: float
    damping_nm_s: float = 0.001
    friction: float = 1.0

    @property
    def body_mass_kg(self) -> float:
        return self.mass_kg * (1 - 2 * 0.0145)

    @property
    def length_m(self) -> float:
        return 0.575 * self.height_m

    @property
    def inertia_kg_m2(self) -> float:
        return self.body_mass_kg * self.length_m**2

    @property
    def stiffness_nm(self) -> float:
        return self.body_mass_kg * 9.81 * self.length_m

    @property
    def foot_length_m(self) -> float:
        return 0.152 * self.height_m


PROFILES = {
    "YF": Profile("YF", 59.4, 1.65, -130, 28, -608, 219),
    "OF": Profile("OF", 60.0, 1.59, -88, 22, -389, 148),
    "YM": Profile("YM", 72.9, 1.77, -181, 43, -957, 309),
    "OM": Profile("OM", 74.5, 1.74, -137, 37, -681, 232),
    "wOF": Profile("wOF", 60.0, 1.59, -78, 21, -303, 130),
}


def exo_torque(q: float, velocity: float, profile: Profile, mode: str) -> float:
    """Bilateral GC or restoring PD torque, clipped to +/-50 Nm (25 Nm/ankle).

    PD follows the authors' dynamics.m: -Kp*q - Kd*velocity. The printed Eq. (16)
    has the opposite signs; see docs/PAPER.md. GC uses exact trigonometry.
    """
    kp = profile.stiffness_nm
    if mode == "none":
        return 0.0
    if mode == "gc":
        raw = -kp * np.sin(q)
    elif mode == "pd":
        kd = 0.3 * np.sqrt(profile.inertia_kg_m2 * kp)
        raw = -kp * q - kd * velocity
    else:
        raise ValueError(f"Unknown exoskeleton mode: {mode}")
    return float(np.clip(raw, -50, 50))


def acceleration(q: float, velocity: float, torque: float, profile: Profile) -> float:
    """Eq. (3) angular acceleration, rad/s^2, given TOTAL human + device torque."""
    return (
        profile.stiffness_nm * np.sin(q) - profile.damping_nm_s * velocity + torque
    ) / profile.inertia_kg_m2


def ground_reaction(q: float, velocity: float, torque: float, profile: Profile) -> dict[str, float]:
    """Flat, stationary foot inverse dynamics: C3-C5 and CoP in world metres.

    These are required reactions, not measured MuJoCo contact forces. If a constraint
    fails, the fixed-foot assumption has failed even if the pendulum remains upright.
    """
    qdd = acceleration(q, velocity, torque, profile)
    mass, length = profile.body_mass_kg, profile.length_m
    fx = mass * length * (np.cos(q) * qdd - np.sin(q) * velocity**2)
    fz = profile.mass_kg * 9.81 - mass * length * (np.sin(q) * qdd + np.cos(q) * velocity**2)
    foot_mass = profile.mass_kg - mass
    foot_com = (0.5 - 0.19) * profile.foot_length_m
    ankle_height = 0.039 * profile.height_m
    moment = -torque + profile.damping_nm_s * velocity + foot_mass * 9.81 * foot_com - ankle_height * fx
    cop = moment / fz if fz > 0 else float("nan")
    heel, toe = -0.19 * profile.foot_length_m, 0.81 * profile.foot_length_m
    return {
        "normal_n": float(fz),
        "friction_n": float(fx),
        "cop_m": float(cop),
        "cop_margin_m": float(min(cop - heel, toe - cop)) if fz > 0 else -1.0,
        "friction_margin_n": float(profile.friction * fz - abs(fx)),
    }


def limit_human(raw: float, previous: float, dt: float, profile: Profile) -> float:
    """Apply C1 and C2 once, preserving the actually applied torque as the state."""
    lower = max(profile.torque_min_nm, previous + profile.rate_min_nm_s * dt)
    upper = min(profile.torque_max_nm, previous + profile.rate_max_nm_s * dt)
    return float(np.clip(raw, lower, upper))
