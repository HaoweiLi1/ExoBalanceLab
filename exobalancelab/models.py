"""Procedural MJCF models in metres/kg: paper pendulum and articulated sagittal human.

World x is anterior, y left, z up. Positive whole-body lean rotates about positive y toward the toes.
The articulated model has independent bilateral hip/knee/ankle pitch joints and a
three-DOF planar floating pelvis. No weld or support is applied to either foot.
"""

import xml.etree.ElementTree as ET

import mujoco

from exobalancelab.paper import Profile


def _element(parent: ET.Element, tag: str, **attrs) -> ET.Element:
    return ET.SubElement(parent, tag, {key: str(value) for key, value in attrs.items()})


def _scene(dt: float, gravity_m_s2: float) -> tuple[ET.Element, ET.Element]:
    root = ET.Element("mujoco", model="ExoBalanceLab")
    _element(root, "compiler", angle="radian", autolimits="true")
    _element(
        root,
        "option",
        timestep=dt,
        gravity=f"0 0 {-gravity_m_s2}",
        integrator="implicitfast",
        solver="Newton",
        iterations="60",
        tolerance="1e-10",
    )
    default = _element(root, "default")
    _element(
        default,
        "geom",
        rgba="0.72 0.80 0.86 1",
        friction="1 .005 .0001",
        solref=".006 1",
        solimp=".95 .99 .001",
        condim="3",
    )
    _element(default, "joint", damping=".3", armature=".005")
    visual = _element(root, "visual")
    _element(visual, "global", offwidth="1600", offheight="900")
    _element(visual, "headlight", ambient=".4 .4 .4", diffuse=".6 .6 .6")
    _element(visual, "rgba", haze=".08 .11 .16 1")
    asset = _element(root, "asset")
    _element(
        asset,
        "texture",
        name="floor",
        type="2d",
        builtin="checker",
        width="512",
        height="512",
        rgb1=".12 .16 .20",
        rgb2=".16 .21 .26",
    )
    _element(asset, "material", name="floor", texture="floor", texrepeat="2 2", reflectance=".12")
    world = _element(root, "worldbody")
    _element(world, "light", pos="1 -3 4", dir="-.2 .6 -1", diffuse=".8 .8 .8")
    _element(world, "light", pos="-2 2 3", dir=".5 -.5 -1", diffuse=".4 .4 .4")
    _element(world, "geom", name="ground", type="plane", size="5 5 .1", material="floor")
    return root, world


def paper_xml(profile: Profile, dt: float = 0.001, contact: bool = False) -> str:
    """Create the point-mass model; contact=True releases the foot in x/z/pitch.

    Fixed-foot COM inertia is 1e-8 kg m^2, bounded in dynamics tests. Contact mode
    uses the physical inertia of the visible 0.1 m sphere for stable body impacts.
    Visual geometry has zero mass; it never changes the analytical plant.
    """
    root, world = _scene(dt, profile.gravity_m_s2)
    length, foot = profile.length_m, profile.foot_length_m
    height, offset = 0.039 * profile.height_m, 0.31 * foot
    base = _element(world, "body", name="feet", pos=f"0 0 {height}")
    if contact:
        _element(base, "joint", name="base_x", type="slide", axis="1 0 0", damping="0", armature="0")
        _element(base, "joint", name="base_z", type="slide", axis="0 0 1", damping="0", armature="0")
        _element(base, "joint", name="base_pitch", axis="0 1 0", damping="0", armature="0")
    foot_mass = profile.mass_kg - profile.body_mass_kg
    _element(
        base,
        "inertial",
        pos=f"{offset} 0 {-height / 2}",
        mass=foot_mass,
        diaginertia=f"{foot_mass * 0.02} {foot_mass * foot**2 / 12} {foot_mass * 0.02}",
    )
    for side in [-1, 1]:
        _element(
            base,
            "geom",
            name=f"foot_{side}",
            type="box",
            size=f"{foot / 2} .055 {height / 2}",
            pos=f"{offset} {side * 0.09} {-height / 2}",
            mass="0",
            rgba=".22 .31 .39 1",
        )
    body = _element(base, "body", name="human")
    _element(body, "joint", name="ankle", axis="0 1 0", damping=profile.damping_nm_s, armature="0")
    # Contact mode gives the visible 0.1 m COM sphere its physical inertia so
    # ground impacts do not excite an almost massless rotational mode.
    com_inertia = 0.4 * profile.body_mass_kg * 0.1**2 if contact else 1e-8
    _element(
        body,
        "inertial",
        pos=f"0 0 {length}",
        mass=profile.body_mass_kg,
        diaginertia=f"{com_inertia} {com_inertia} {com_inertia}",
    )
    _element(
        body,
        "geom",
        type="capsule",
        fromto=f"0 0 0 0 0 {length}",
        size=".035",
        mass="0",
        contype="1" if contact else "0",
        conaffinity="1" if contact else "0",
    )
    _element(
        body,
        "geom",
        type="sphere",
        pos=f"0 0 {length}",
        size=".10",
        mass="0",
        contype="1" if contact else "0",
        conaffinity="1" if contact else "0",
        rgba=".31 .68 .78 1",
    )
    _element(body, "site", name="com", pos=f"0 0 {length}", size=".012", rgba="1 .75 .15 1")
    _element(
        body,
        "geom",
        type="cylinder",
        size=".045 .075",
        quat=".7071068 .7071068 0 0",
        pos="0 0 .03",
        rgba=".94 .44 .18 1",
        mass="0",
        contype="0",
        conaffinity="0",
    )
    actuator = _element(root, "actuator")
    _element(
        actuator,
        "motor",
        name="human_ankle",
        joint="ankle",
        gear="1",
        ctrlrange=f"{profile.torque_min_nm} {profile.torque_max_nm}",
    )
    _element(actuator, "motor", name="exo_ankle", joint="ankle", gear="1", ctrlrange="-50 50")
    return ET.tostring(root, encoding="unicode")


def human_xml(profile: Profile, dt: float = 0.001, device_mass_kg: float = 2.0) -> str:
    """Bilateral 9-DOF contact model with distributed segment/device mass.

    Segment fractions: paired feet .029, shanks .093, thighs .200, pelvis .142,
    trunk .355, head .081 and paired arms .100 of total human mass. This is an
    engineering anthropometric approximation, not a subject-specific musculoskeletal model.
    Device mass is total bilateral mass, 70% on shanks and 30% on feet.
    """
    if device_mass_kg < 0:
        raise ValueError("Device mass must be nonnegative")
    root, world = _scene(dt, profile.gravity_m_s2)
    height, mass = profile.height_m, profile.mass_kg
    shank, thigh, trunk = 0.246 * height, 0.245 * height, 0.288 * height
    ankle_h = 0.039 * height
    pelvis = _element(world, "body", name="pelvis", pos=f"0 0 {shank + thigh + ankle_h}")
    _element(pelvis, "joint", name="root_x", type="slide", axis="1 0 0", damping="0", armature="0")
    _element(pelvis, "joint", name="root_z", type="slide", axis="0 0 1", damping="0", armature="0")
    _element(pelvis, "joint", name="root_pitch", axis="0 1 0", damping="0", armature="0")
    _element(pelvis, "geom", name="pelvis_geom", type="ellipsoid", size=".09 .14 .10", mass=0.142 * mass)
    chest = _element(pelvis, "body", name="trunk")
    _element(
        chest,
        "geom",
        type="capsule",
        fromto=f"0 0 .07 0 0 {trunk}",
        size=".115",
        mass=0.355 * mass,
        rgba=".36 .60 .71 1",
    )
    _element(
        chest,
        "geom",
        type="sphere",
        pos=f"0 0 {trunk + 0.14}",
        size=".105",
        mass=0.081 * mass,
        rgba=".83 .72 .60 1",
    )
    for side, y in [("left", 0.115), ("right", -0.115)]:
        _element(
            chest,
            "geom",
            type="capsule",
            fromto=f"0 {y * 1.65} {trunk - 0.02} .01 {y * 2} .04",
            size=".037",
            mass=0.05 * mass,
            rgba=".83 .72 .60 1",
        )
        leg = _element(pelvis, "body", name=f"{side}_thigh", pos=f"0 {y} 0")
        _element(leg, "joint", name=f"{side}_hip", axis="0 1 0", range="-1.2 .8")
        _element(
            leg, "geom", type="capsule", fromto=f"0 0 -.04 0 0 {-thigh + 0.04}", size=".063", mass=0.10 * mass
        )
        shin = _element(leg, "body", name=f"{side}_shank", pos=f"0 0 {-thigh}")
        _element(shin, "joint", name=f"{side}_knee", axis="0 1 0", range="-.1 2.2")
        _element(
            shin,
            "geom",
            type="capsule",
            fromto=f"0 0 -.025 0 0 {-shank + 0.04}",
            size=".043",
            mass=0.0465 * mass,
        )
        _element(
            shin,
            "geom",
            name=f"{side}_exo_cuff",
            type="box",
            pos=f"-.04 0 {-shank * 0.7}",
            size=".025 .064 .06",
            mass=0.35 * device_mass_kg,
            rgba=".95 .43 .16 1",
            contype="0",
            conaffinity="0",
        )
        _element(
            shin,
            "geom",
            name=f"{side}_exo_strut",
            type="capsule",
            fromto=f"-.045 0 {-shank * 0.7} -.045 0 {-shank}",
            size=".009",
            mass="0",
            rgba=".70 .77 .82 1",
            contype="0",
            conaffinity="0",
        )
        foot = _element(shin, "body", name=f"{side}_foot", pos=f"0 0 {-shank}")
        _element(foot, "joint", name=f"{side}_ankle", axis="0 -1 0", range="-.65 .65")
        lf = profile.foot_length_m
        _element(
            foot,
            "geom",
            name=f"{side}_sole",
            type="box",
            pos=f"{0.31 * lf} 0 {-ankle_h / 2}",
            size=f"{lf / 2} .058 {ankle_h / 2}",
            mass=0.0145 * mass,
            rgba=".19 .27 .33 1",
        )
        _element(
            foot,
            "geom",
            name=f"{side}_exo_motor",
            type="cylinder",
            size=".038 .065",
            quat=".7071068 .7071068 0 0",
            mass=0.15 * device_mass_kg,
            rgba=".95 .43 .16 1",
            contype="0",
            conaffinity="0",
        )
    actuator = _element(root, "actuator")
    for side in ["left", "right"]:
        for joint in ["hip", "knee", "ankle"]:
            limits = (
                f"{profile.torque_min_nm / 2} {profile.torque_max_nm / 2}" if joint == "ankle" else "-100 100"
            )
            _element(
                actuator,
                "motor",
                name=f"{side}_{joint}_human",
                joint=f"{side}_{joint}",
                gear="1",
                ctrlrange=limits,
            )
    for side in ["left", "right"]:
        _element(
            actuator, "motor", name=f"{side}_ankle_exo", joint=f"{side}_ankle", gear="1", ctrlrange="-25 25"
        )
    return ET.tostring(root, encoding="unicode")


def compile_model(xml: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Compile a standalone XML and compute its initial kinematics."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data
