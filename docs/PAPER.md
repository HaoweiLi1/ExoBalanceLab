# Paper correspondence and reproduction status

Reference: Daphna Raz, Varun Joshi, Brian R. Umberger and Necmiye Ozay,
[*Ankle Exoskeletons May Hinder Standing Balance in Simple Models of Older and Younger Adults*](https://arxiv.org/abs/2408.05418),
arXiv v3, October 2, 2024. Authors' code:
[AnkleExoBRS at 517ab41](https://github.com/Raquelometer/AnkleExoBRS/tree/517ab41b7767c5ab0e856a8885b118bd739a5011).

## What is implemented

| Paper component | Implementation | Evidence / scope |
|---|---|---|
| Lifted state `(theta, theta_dot, tau_human)`, Eq. (3) | `paper.py`, `models.paper_xml` | Independent acceleration and integration checks |
| C1 human maximum torque | `limit_human` and actuator bounds | Raw/applied action tests |
| C2 maximum torque rate | State-aware sampled torque limiting | Directional rate and saturation tests |
| C3 normal force, C4 CoP, C5 friction | Analytical required reactions; measured contact extension | Static equilibrium/CoP checks and logged margins |
| Table I, five profiles | `PROFILES` | Explicit bilateral MT/MRTD and anthropometry |
| GC, Eq. (15) | `exo_torque(..., "gc")` | Exact trigonometry, hard ±50 Nm saturation |
| PD, Eq. (16) | Restoring signs from authors' `dynamics.m` | Opposite signs to the printed equation; explicit audit below |
| Eq. (11), value-gradient torque-rate feedback | `reachability.HJBPolicy` and MATLAB exporter | Synthetic grid/interface tests; original HJB policy pending |
| Published HJB regions, Figs. 4–6 / Table II | Not yet reproduced | Requires working MATLAB runtime, matching source/dependencies and convergence study |
| More realistic embodiment | Independent articulated model | A new engineering experiment; outside the paper's model assumptions |

The recovery grid is a **fixed-controller, zero-initial-human-torque slice**. The paper's
stabilizable region projects over all admissible initial human torques and optimizes
control through a Hamilton–Jacobi value function. Counts or areas from these two objects
are not interchangeable.

## Coordinates and mechanics

World x points forward toward the toe. Set `q = theta - pi/2`, with positive q forward.
Thus body CoM is `(l*sin(q), 0, h_ankle+l*cos(q))`; theta is the paper angle, not a polar
angle measured counterclockwise from world +x. Positive torque is dorsiflexion (DF),
negative torque is plantarflexion (PF). These conventions are checked through body
kinematics and static pressure-center tests, not just symbol names.

The point-mass equation is

```text
I = m*l²
qdd = (m*g*l*sin(q) - b*qdot + tau_human + tau_exo) / I
```

MuJoCo needs a positive body inertia: the COM inertia is 1e-8 kg m². All visible geometry
in the paper model has zero mass. The resulting analytical inertia discrepancy is
quantified in the numerical report. The fixed foot does not produce measured contact
forces; required reactions are evaluated from inverse dynamics:

```text
Fx = m*l*(cos(q)*qdd - sin(q)*qdot²)
Fz = (m+mf)*g - m*l*(sin(q)*qdd + cos(q)*qdot²)
CoP = (-tau_total + b*qdot + mf*g*c - h_ankle*Fx) / Fz
```

The foot extends from `-0.19*lf` to `0.81*lf`. C3–C5 are evaluated only as the flat-foot
hypothesis permits; a violation invalidates recovery even when the constrained pendulum
remains upright. `contact` releases foot x/z/pitch, permitting lift, slip and rotation. Its 0.1 m COM
sphere has finite solid-sphere rotational inertia, so body-ground impact remains physical;
this adds 0.4*m*0.1² kg m² to ankle inertia. Only `paper` is the exact point-mass match.

## Parameter provenance

| Profile | Mass kg | Height m | PF / DF MT Nm | PF / DF MRTD Nm/s |
|---|---:|---:|---:|---:|
| YF | 59.4 | 1.65 | -130 / 28 | -608 / 219 |
| OF | 60.0 | 1.59 | -88 / 22 | -389 / 148 |
| YM | 72.9 | 1.77 | -181 / 43 | -957 / 309 |
| OM | 74.5 | 1.74 | -137 / 37 | -681 / 232 |
| wOF | 60.0 | 1.59 | -78 / 21 | -303 / 130 |

Mass and height follow the printed Table I. Segment scaling follows the authors'
`proportionallyEstimatedParams.m`: `l=.575*height`, `lf=.152*height`, paired feet mass
`.029*total_mass`, ankle position `.19*lf`, ankle height `.039*height`, foot COM `.31*lf`.
The reduced-body mass excludes feet. Default g=9.81 m/s²; damping b=.001 Nm s/rad follows
the authors' dynamics class, replacing the much larger historical simulator damping.
These are bilateral human/device limits; individual human-model ankles receive half.

## Source discrepancies to resolve before claiming exact reproduction

Source inspection at the pinned authors' commit found:

1. **Printed PD signs:** Eq. (16) displays positive proportional/derivative feedback while
   `Dynamics/LiftedInvPendExo/dynamics.m` uses `Kp*(pi/2-theta) + Kd*(0-theta_dot)`.
   ExoBalanceLab uses the restoring implementation, with positive gain magnitudes.
2. **Height and gravity:** the MAT files use OF height 1.58 m and YF 1.653 m, versus Table I's
   1.59 and 1.65 m. The source uses g=9.8, while this simulator defaults to 9.81.
3. **Smoothed GC:** `saturatedCosVal.m` is approximately **minus cosine** near upright, despite
   its name. The dynamics add it while the constraint functions use another sign. This
   differs from Eq. (15)'s gravity-compensating torque. Our GC implements Eq. (15) directly.
   Exporting an upstream GC value function does not by itself resolve this plant mismatch.
4. **Missing friction function:** both solver entry points call
   `generateLiftedInvPendFrictionConstraint`, absent from the inspected repository.
5. **Incomplete stored target configuration:** stored `targetSets` have widths but some
   entry scripts access equilibrium fields absent from the saved structs. The provided
   parameter files cover no-exo and GC; PD target files are not included.
6. **Target-set dependencies:** the target generators use YALMIP (`sdpvar`, `sdpsettings`),
   SeDuMi and `lcon2vert`; all must be available in addition to helperOC/ToolboxLS.

These findings are implementation differences, not a claim that the paper's mathematical
results are wrong. An exact reproduction needs a reconciled parameter/controller version,
all target and friction functions, solver resolution/time settings and a comparison against
the reported figures. Record any patch separately from the authors' unchanged source.

## Acceptance for the remaining HJB reproduction

1. Complete MATLAB runtime sign-in and verify the required numerical/toolbox dependencies.
2. Verify the [pinned helperOC/ToolboxLS, YALMIP and SeDuMi sources](../matlab/README.md#prepared-hjb-dependencies)
   in MATLAB; acquire and pin the geometry utility. Preserve vendor licenses.
3. Resolve the source differences above with versioned patches and independent unit checks.
4. Recompute target sets and all five population profiles for no-exo/GC/PD, at documented
   grid resolutions and horizons; check numerical convergence.
5. Compare forward/backward projected areas to the paper's Table II, reporting discrepancies.
6. Export matching policies; select states inside/outside and near boundaries before viewing
   results; compare their MuJoCo trajectories and C1–C5 validity against the solver prediction.

[The results page](RESULTS.md) lists what has actually been run. Synthetic policy tests,
a point-mass acceleration match or attractive animation do not satisfy these six steps.
