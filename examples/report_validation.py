"""Rebuild numerical acceptance, strength, timing and paired-model reports.

Accepts either logs/validation_v1 or its curated docs/results/validation_v1 copy.
Full HJB histories and dense simulation trajectories are not needed for reporting.
"""

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

from exobalancelab.paper import AUTHORS_PROFILES, exo_torque, ground_reaction, scale_strength
from exobalancelab.work_energy import potential, stopping_envelope, zero_torque_velocity


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def table(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)


def area(theta, velocity, field, direction):
    mask = velocity >= 0 if direction == "forward" else velocity <= 0
    return float(
        np.trapezoid(np.trapezoid((field[:, mask] <= 0).astype(float), velocity[mask], axis=1), theta)
    )


def overlap(a, b, direction):
    axes = [np.linspace(max(a[i][0], b[i][0]), min(a[i][-1], b[i][-1]), 601) for i in [0, 1]]
    axes[1] = axes[1][axes[1] >= 0 if direction == "forward" else axes[1] <= 0]
    x, y = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel()])
    masks = [RegularGridInterpolator(item[:2], item[2])(points) <= 0 for item in [a, b]]
    return float(np.count_nonzero(masks[0] & masks[1]) / max(1, np.count_nonzero(masks[0] | masks[1])))


def hjb_report(root, output, acceptance):
    manifest = json.loads((root / "hjb/manifest.json").read_text())
    rows, arrays, summaries = [], {}, {}
    for case in manifest["cases"]:
        path = root / "hjb/solved" / case["id"]
        summary = json.loads((path / "summary.json").read_text())
        if summary["stage"] != "solved":
            raise ValueError("Unsolved case: " + case["id"])
        for key in ["profile", "exo", "direction", "mt_scale", "mrtd_scale", "target_definition", "cfl"]:
            if summary[key] != case[key]:
                raise ValueError("Case contract mismatch: " + case["id"] + " / " + key)
        expected = scale_strength(AUTHORS_PROFILES[case["profile"]], case["mt_scale"], case["mrtd_scale"])
        for key in ["torque_min_nm", "torque_max_nm", "rate_min_nm_s", "rate_max_nm_s"]:
            if not np.isclose(summary["effective_profile"][key], getattr(expected, key), atol=1e-12, rtol=0):
                raise ValueError("Independent strength contract mismatch")
        data = loadmat(path / "projection.mat")
        axes = (data["theta_rad"].ravel(), data["velocity_rad_s"].ravel(), data["projected_value"])
        arrays[case["id"]] = axes
        summaries[case["id"]] = summary
        rows.append(
            dict(
                case,
                area_rad2_per_s=area(*axes, case["direction"]),
                wall_s=summary["wall_s"],
                projection_sha256=hashlib.sha256((path / "projection.mat").read_bytes()).hexdigest(),
            )
        )
    for row in rows:
        baseline = next(
            r
            for r in rows
            if all(r[k] == row[k] for k in ["profile", "direction", "resolution", "target_definition", "cfl"])
            and r["exo"] == "none"
            and r["mt_scale"] == r["mrtd_scale"] == 1
        )
        same_mode = next(
            r
            for r in rows
            if all(
                r[k] == row[k]
                for k in ["profile", "exo", "direction", "resolution", "target_definition", "cfl"]
            )
            and r["mt_scale"] == r["mrtd_scale"] == 1
        )
        row["change_from_nominal_unassisted_percent"] = 100 * (
            row["area_rad2_per_s"] / baseline["area_rad2_per_s"] - 1
        )
        row["change_from_same_mode_nominal_percent"] = 100 * (
            row["area_rad2_per_s"] / same_mode["area_rad2_per_s"] - 1
        )
        unassisted = next(
            r
            for r in rows
            if all(
                r[k] == row[k]
                for k in [
                    "profile",
                    "direction",
                    "resolution",
                    "target_definition",
                    "cfl",
                    "mt_scale",
                    "mrtd_scale",
                ]
            )
            and r["exo"] == "none"
        )
        row["assistance_effect_percent"] = 100 * (row["area_rad2_per_s"] / unassisted["area_rad2_per_s"] - 1)
    checks = []
    for fine in rows:
        if fine["resolution"] != 225 or fine["cfl"] != 0.8:
            continue
        coarse = next(
            r
            for r in rows
            if r["resolution"] == 151
            and all(
                r[k] == fine[k]
                for k in ["profile", "exo", "direction", "target_definition", "mt_scale", "mrtd_scale", "cfl"]
            )
        )
        change = 100 * abs(fine["area_rad2_per_s"] - coarse["area_rad2_per_s"]) / fine["area_rad2_per_s"]
        effect = abs(fine["assistance_effect_percent"] - coarse["assistance_effect_percent"])
        iou = overlap(arrays[coarse["id"]], arrays[fine["id"]], fine["direction"])
        checks.append(
            dict(
                coarse=coarse["id"],
                fine=fine["id"],
                area_error_percent=change,
                assistance_effect_change_pp=effect,
                projection_iou=iou,
                passed=bool(
                    change <= acceptance["spatial_area_relative_percent"]
                    and effect <= acceptance["spatial_assistance_effect_pp"]
                    and iou >= acceptance["common_grid_projection_iou"]
                ),
            )
        )
    temporal = []
    for fine in rows:
        if fine["cfl"] != 0.4:
            continue
        coarse = next(
            r
            for r in rows
            if r["cfl"] == 0.8
            and all(
                r[k] == fine[k]
                for k in [
                    "profile",
                    "exo",
                    "direction",
                    "target_definition",
                    "mt_scale",
                    "mrtd_scale",
                    "resolution",
                ]
            )
        )
        change = 100 * abs(fine["area_rad2_per_s"] - coarse["area_rad2_per_s"]) / fine["area_rad2_per_s"]
        iou = overlap(arrays[coarse["id"]], arrays[fine["id"]], fine["direction"])
        temporal.append(
            dict(
                coarse=coarse["id"],
                fine=fine["id"],
                area_error_percent=change,
                projection_iou=iou,
                passed=bool(
                    change <= acceptance["temporal_area_relative_percent"]
                    and iou >= acceptance["temporal_projection_iou"]
                ),
            )
        )
    targets = []
    for item in rows:
        s = summaries[item["id"]]
        p = scale_strength(AUTHORS_PROFILES[item["profile"]], item["mt_scale"], item["mrtd_scale"])
        for label, center in [("Ank", "ankleEq"), ("Toe", "toeEq")]:
            target = s["target"]
            widths = np.array(
                [target[f"theta{label}Range"], target[f"thetaDot{label}Range"], target[f"torque{label}Range"]]
            )
            reactions = []
            for signs in itertools.product([-1, 1], repeat=3):
                theta, v, torque = np.array(target[center]) + np.array(signs) * widths
                q = theta - np.pi / 2
                grf = ground_reaction(q, v, torque + exo_torque(q, v, p, item["exo"]), p)
                reactions.append([grf["normal_n"], grf["cop_margin_m"], grf["friction_margin_n"]])
            outward = np.array(target[center]) + [widths[0], widths[1] / 2, 0]
            q, v, torque = outward - [np.pi / 2, 0, 0]
            grf = ground_reaction(q, v, torque + exo_torque(q, v, p, item["exo"]), p)
            admissible = bool(
                min(grf["normal_n"], grf["cop_margin_m"], grf["friction_margin_n"]) >= 0
                and p.torque_min_nm <= torque <= p.torque_max_nm
            )
            if label == "Ank" and not admissible:
                raise ValueError("Ankle-goal noninvariance example is outside the admissible set")
            targets.append(
                dict(
                    case=item["id"],
                    target=label,
                    center_si=target[center],
                    widths_si=widths.tolist(),
                    minimum_corner_margins=np.min(reactions, axis=0).tolist(),
                    outward_face_state_si=outward.tolist(),
                    outward_angle_velocity_rad_s=float(v),
                    outward_face_admissible=admissible,
                    outward_face_ground_reaction=grf,
                    controlled_invariant_box=False if admissible else None,
                    reason=(
                        "At the upper angle face with positive velocity, theta_dot>0 "
                        "independently of the torque-rate input. This is a counterexample "
                        "for the clipped goal only when the face point is admissible."
                    ),
                )
            )
    table(output / "regions.csv", rows)
    fig5 = []
    # Article Section V-B reports these four forward-region examples. The MT
    # values are rounded prose figures; changed goals preclude an exact-replication claim.
    for parameter, mode, reported in [
        ("mt_scale", "none", -20.0),
        ("mt_scale", "gc", -10.0),
        ("mrtd_scale", "none", -2.6),
        ("mrtd_scale", "gc", -10.4),
    ]:
        row = next(
            r
            for r in rows
            if r["profile"] == "YF"
            and r["exo"] == mode
            and r["direction"] == "forward"
            and r["resolution"] == 151
            and r[parameter] == 0.7
            and r["cfl"] == 0.8
        )
        actual = row["change_from_nominal_unassisted_percent"]
        fig5.append(
            dict(
                parameter=parameter,
                scale=0.7,
                exo=mode,
                article_prose_change_percent=reported,
                measured_change_percent=actual,
                difference_pp=actual - reported,
                change_from_own_nominal_percent=row["change_from_same_mode_nominal_percent"],
                source="https://arxiv.org/html/2408.05418v3#S5.SS2",
                scope=(
                    "Source-entry goal construction; finite grid; direct quantitative agreement unaccepted."
                ),
            )
        )
    table(output / "fig5_comparison.csv", fig5)
    table(output / "spatial_acceptance.csv", checks)
    table(output / "temporal_acceptance.csv", temporal)
    write(output / "target_audit.json", targets)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, direction in zip(axes, ["forward", "backward"], strict=True):
        for mode, color in [("none", "#c87825"), ("gc", "#6b4e9b")]:
            for parameter, marker in [("mt_scale", "o"), ("mrtd_scale", "D")]:
                other = "mrtd_scale" if parameter == "mt_scale" else "mt_scale"
                data = sorted(
                    [
                        r
                        for r in rows
                        if r["profile"] == "YF"
                        and r["resolution"] == 151
                        and r["direction"] == direction
                        and r["exo"] == mode
                        and r[other] == 1
                        and r["cfl"] == 0.8
                    ],
                    key=lambda r: -r[parameter],
                )
                ax.plot(
                    [100 * (1 - r[parameter]) for r in data],
                    [r["change_from_nominal_unassisted_percent"] for r in data],
                    color=color,
                    marker=marker,
                    ls="-" if parameter == "mt_scale" else "--",
                    label=f"{mode.upper()} / {'MT' if parameter == 'mt_scale' else 'MRTD'}",
                )
                for coarse in data:
                    refined = next(
                        (
                            r
                            for r in rows
                            if r["profile"] == "YF"
                            and r["resolution"] == 225
                            and r["direction"] == direction
                            and r["exo"] == mode
                            and r["mt_scale"] == coarse["mt_scale"]
                            and r["mrtd_scale"] == coarse["mrtd_scale"]
                            and r["cfl"] == 0.8
                        ),
                        None,
                    )
                    if refined is not None:
                        y = coarse["change_from_nominal_unassisted_percent"]
                        delta = refined["change_from_nominal_unassisted_percent"] - y
                        ax.errorbar(
                            100 * (1 - coarse[parameter]),
                            y,
                            yerr=[[max(0, -delta)], [max(0, delta)]],
                            fmt="none",
                            color=color,
                            capsize=3,
                            linewidth=0.8,
                        )
        ax.set(title=direction.capitalize() + " velocity", xlabel="Independent capacity reduction (%)")
        ax.grid(alpha=0.2)
    axes[0].set_ylabel("Area change from nominal unassisted (%)")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "YF independent strength sweep · source-entry targets · 151³ requested grid\n"
        "Whiskers span 151³ and 225³ results: numerical sensitivity",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output / "strength.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    definitions = ["recomputed", "source_entry", "common_ankle"]
    for index, n in enumerate([151, 225]):
        effects = [
            next(
                r["assistance_effect_percent"]
                for r in rows
                if r["profile"] == "OM"
                and r["exo"] == "gc"
                and r["target_definition"] == target
                and r["resolution"] == n
            )
            for target in definitions
        ]
        bars = ax.bar(
            np.arange(3) + (index - 0.5) * 0.32,
            effects,
            width=0.32,
            color=["#a8c6d4", "#366b83"][index],
            label=f"Requested {n}³",
        )
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(3), ["Recomputed boxes", "Source-entry overrides", "Identical ankle goal"])
    ax.set(ylabel="GC area change relative to none (%)", title="OM forward domain · target sensitivity")
    ax.margins(y=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "targets.png", dpi=180)
    plt.close(fig)
    return dict(
        solves=len(rows),
        independent_strength_cases=sum(
            r["profile"] == "YF" and r["resolution"] == 151 and r["cfl"] == 0.8 for r in rows
        ),
        spatial_passed=sum(c["passed"] for c in checks),
        spatial_total=len(checks),
        spatial_max_area_error_percent=max(c["area_error_percent"] for c in checks),
        temporal=temporal,
        target_box_invariance_claim=False,
        article_fig5_comparison=fig5,
    )


def paired_report(root, output):
    results = json.loads((root / "paired/outcomes.json").read_text())
    specification = json.loads((root / "paired/protocol.json").read_text())
    expected_ids = [case["id"] for case in specification["cases"]]
    if [case["id"] for case in results] != expected_ids:
        raise ValueError("Paired results must retain every frozen case exactly once and in order")
    # Physics refinement must compare the same initialization and controller.
    for coarse in results:
        if coarse["config"]["timestep_s"] != 0.001 or coarse["status"] != "completed":
            continue
        fine = next(
            r
            for r in results
            if r["config"]["timestep_s"] == 0.0005
            and all(r["config"][k] == coarse["config"][k] for k in coarse["config"] if k != "timestep_s")
        )
        if fine["status"] == "completed":
            for key in ["initial_state", "feedback_gain", "feedback_reference"]:
                np.testing.assert_array_equal(coarse["metrics"][key], fine["metrics"][key])
    comparisons = []
    for pair in sorted({r["pair"] for r in results}):
        group = {r["config"]["exo"]: r for r in results if r["pair"] == pair}
        for mode in ["gc", "pd"]:
            base, assisted = group["none"], group[mode]
            valid = base["status"] == assisted["status"] == "completed"
            a, b = [x.get("metrics", {}).get("recovered", False) for x in [base, assisted]]
            outcome = (
                "rollout_error"
                if not valid
                else (
                    "both_recover"
                    if a and b
                    else "assistance_helps"
                    if b
                    else "assistance_harms"
                    if a
                    else "both_fail"
                )
            )
            config = assisted["config"]
            comparisons.append(
                dict(
                    pair=pair,
                    profile=config["profile"],
                    model=config["model"],
                    feedback=config["human_feedback"],
                    timestep_s=config["timestep_s"],
                    exo=mode,
                    lean_rad=config["lean_rad"],
                    velocity_rad_s=config["velocity_rad_s"],
                    push_n=config["push_n"],
                    outcome=outcome,
                    baseline_recovered=bool(a),
                    assisted_recovered=bool(b),
                )
            )
    groups = []
    for model, feedback, mode, dt in itertools.product(
        ["paper", "human"], ["fixed", "readapted"], ["gc", "pd"], [0.001, 0.0005]
    ):
        subset = [
            r
            for r in comparisons
            if (r["model"], r["feedback"], r["exo"], r["timestep_s"]) == (model, feedback, mode, dt)
        ]
        groups.append(
            dict(
                model=model,
                feedback=feedback,
                exo=mode,
                timestep_s=dt,
                pairs=len(subset),
                **{
                    key: sum(r["outcome"] == key for r in subset)
                    for key in [
                        "both_recover",
                        "assistance_helps",
                        "assistance_harms",
                        "both_fail",
                        "rollout_error",
                    ]
                },
            )
        )
    refinement = []
    for coarse in comparisons:
        if coarse["timestep_s"] != 0.001:
            continue
        fine = next(
            r
            for r in comparisons
            if r["timestep_s"] == 0.0005
            and all(
                r[k] == coarse[k]
                for k in ["profile", "model", "feedback", "exo", "lean_rad", "velocity_rad_s", "push_n"]
            )
        )
        refinement.append(
            dict(
                pair=coarse["pair"],
                exo=coarse["exo"],
                coarse=coarse["outcome"],
                fine=fine["outcome"],
                stable=coarse["outcome"] == fine["outcome"]
                and "rollout_error" not in [coarse["outcome"], fine["outcome"]],
            )
        )
    table(output / "paired_cases.csv", comparisons)
    table(output / "paired_summary.csv", groups)
    table(output / "paired_refinement.csv", refinement)
    correspondence = []
    for paper in comparisons:
        if paper["model"] != "paper" or paper["timestep_s"] != 0.0005:
            continue
        human = next(
            r
            for r in comparisons
            if r["model"] == "human"
            and r["timestep_s"] == 0.0005
            and all(
                r[k] == paper[k]
                for k in ["profile", "feedback", "exo", "lean_rad", "velocity_rad_s", "push_n"]
            )
        )
        correspondence.append(
            {
                **{k: paper[k] for k in ["profile", "feedback", "exo", "lean_rad", "velocity_rad_s"]},
                "paper_outcome": paper["outcome"],
                "human_outcome": human["outcome"],
                "same_outcome": paper["outcome"] == human["outcome"],
            }
        )
    table(output / "model_correspondence.csv", correspondence)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, model in zip(axes, ["paper", "human"], strict=True):
        subset = [r for r in groups if r["model"] == model and r["timestep_s"] == 0.0005]
        # Plot the shared lean/speed grid. Extra chest pushes remain in the tables.
        subset = [
            {
                **r,
                **{
                    key: sum(
                        c["outcome"] == key
                        and c["push_n"] == 0
                        and c["model"] == model
                        and c["timestep_s"] == 0.0005
                        and c["feedback"] == r["feedback"]
                        and c["exo"] == r["exo"]
                        for c in comparisons
                    )
                    for key in [
                        "both_recover",
                        "assistance_helps",
                        "assistance_harms",
                        "both_fail",
                        "rollout_error",
                    ]
                },
            }
            for r in subset
        ]
        bottom = np.zeros(len(subset))
        for key, color in [
            ("both_recover", "#5f8a83"),
            ("assistance_helps", "#43a6c6"),
            ("assistance_harms", "#d5764c"),
            ("both_fail", "#a9afb5"),
            ("rollout_error", "#482445"),
        ]:
            values = np.array([r[key] for r in subset])
            ax.bar(np.arange(len(subset)), values, bottom=bottom, label=key.replace("_", " "), color=color)
            bottom += values
        ax.set_xticks(np.arange(len(subset)), [r["feedback"] + "\n" + r["exo"].upper() for r in subset])
        ax.set_title(model.capitalize() + " · dt = 0.5 ms")
    axes[0].set_ylabel("Shared lean/speed conditions (two profiles)")
    axes[0].set_ylim(0, 18 * 1.12)
    axes[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1, 1))
    fig.tight_layout()
    fig.savefig(output / "paired.png", dpi=180)
    plt.close(fig)
    return dict(
        episodes=len(results),
        errors=sum(r["status"] != "completed" for r in results),
        groups=groups,
        refinement_stable=sum(r["stable"] for r in refinement),
        refinement_total=len(refinement),
        transferred_hjb_guarantee=False,
        fixed_initial_state_and_gains_across_refinement=True,
        cross_model_same_category=sum(r["same_outcome"] for r in correspondence),
        cross_model_total=len(correspondence),
    )


def contact_report(root, output):
    directory = root / "contact_followup"
    specification = json.loads((directory / "protocol.json").read_text())
    outcomes = json.loads((directory / "outcomes.json").read_text())
    if [r["config"] for r in outcomes] != specification["conditions"]:
        raise ValueError("Incomplete contact follow-up cohort")
    rows = json.loads((directory / "diagnosis.json").read_text())
    if [r["id"] for r in rows] != [r["id"] for r in outcomes if r["status"] == "completed"]:
        raise ValueError("Missing completed contact diagnosis")
    table(
        output / "contact_followup.csv",
        [
            dict(
                exo=r["config"]["exo"],
                timestep_s=r["config"]["timestep_s"],
                **{k: v for k, v in r.items() if k != "config"},
            )
            for r in rows
        ],
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, mode in zip(axes, ["gc", "pd"], strict=True):
        for row in rows:
            if row["config"]["exo"] != mode:
                continue
            with np.load(directory / "cases" / row["id"] / "contact_window.npz") as data:
                line = ax.plot(
                    data["time_s"] * 1000,
                    data["normal_n"],
                    label=f"{row['config']['timestep_s'] * 1e6:g} µs physics",
                    linewidth=0.9,
                )[0]
                zero = data["normal_n"] <= 0
                ax.scatter(data["time_s"][zero] * 1000, data["normal_n"][zero], color=line.get_color(), s=15)
        ax.set(title=mode.upper(), xlabel="Time (ms)", xlim=(90, 120))
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Total normal support (N)")
    fig.suptitle("Contact-sensitive OF condition · fixed 5 ms control · readapted feedback")
    fig.tight_layout()
    fig.savefig(output / "contact_followup.png", dpi=180)
    plt.close(fig)
    return dict(
        scope="Post-campaign diagnosis of every classification-unstable human condition.",
        episodes=len(outcomes),
        completed=len(rows),
        recovered=sum(r["recovered"] for r in rows),
        all_tested_steps_retain_zero_support=all(r["zero_support_samples"] > 0 for r in rows),
        original_classifications_retained=True,
    )


def timing_report(root, output, acceptance):
    results = json.loads((root / "timing/outcomes.json").read_text())
    specification = json.loads((root / "timing/protocol.json").read_text())
    setting = specification["parent_protocol"]["timing"]
    conditions = [
        (setting["physics_dt_s"], period, delay)
        for period, delay in itertools.product(setting["control_periods_s"], setting["command_delays_s"])
    ]
    conditions += [(setting["physics_reference_dt_s"], setting["physics_reference_control_period_s"], 0)]
    expected = [
        (group["id"], tuple(state["state"]), dt, period, delay)
        for group in specification["groups"]
        for dt, period, delay in conditions
        for state in group["states"]
    ]
    actual = [
        (
            r["policy_id"],
            tuple(r["state"]["state"]),
            r["config"]["timestep_s"],
            r["config"]["control_period_s"],
            r["config"]["command_delay_s"],
        )
        for r in results
    ]
    if actual != expected:
        raise ValueError("Timing results must retain every frozen case exactly once and in order")
    rows = []
    for r in results:
        c = r["config"]
        metrics = r.get("metrics", {})
        rows.append(
            dict(
                id=r["id"],
                policy=r["policy_id"],
                group=r["state"]["group"],
                predicted_inside=r["predicted_inside"],
                timestep_s=c["timestep_s"],
                control_period_s=c["control_period_s"],
                delay_s=c["command_delay_s"],
                status=r["status"],
                valid_target_reach=r.get("valid_target_reach", False),
                control_p99_us=metrics.get("control_p99_us"),
                deadline_miss_fraction=metrics.get("control_deadline_miss_fraction"),
                simulated_s_per_wall_s=metrics.get("simulated_s_per_wall_s"),
            )
        )
    table(output / "timing_cases.csv", rows)
    refinement = []
    limits = np.array(
        [acceptance["rollout_endpoint_abs"][key] for key in ["angle_rad", "velocity_rad_s", "torque_nm"]]
    )
    for coarse in results:
        c = coarse["config"]
        if c["timestep_s"] != 0.0000125 or c["control_period_s"] != 0.001 or c["command_delay_s"] != 0:
            continue
        fine = next(
            r
            for r in results
            if r["config"]["timestep_s"] == 0.00000625
            and r["policy_id"] == coarse["policy_id"]
            and r["state"] == coarse["state"]
        )
        error = None
        stable = False
        classifications_equal = None
        if fine["status"] == coarse["status"] == "completed":
            error = np.abs(np.array(fine["final_state"]) - coarse["final_state"])
            classifications_equal = all(
                fine[key] == coarse[key] for key in ["target_reached", "contact_valid", "valid_target_reach"]
            )
            stable = classifications_equal and np.all(error <= limits)
        refinement.append(
            dict(
                policy=coarse["policy_id"],
                group=coarse["state"]["group"],
                absolute_endpoint_error=None if error is None else error.tolist(),
                target_and_contact_classifications_equal=classifications_equal,
                passed=bool(stable),
                assessment="not_evaluable" if error is None else "pass" if stable else "fail",
                coarse_status=coarse["status"],
                fine_status=fine["status"],
            )
        )
    write(output / "timing_refinement.json", refinement)
    groups = []
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    for ax, mode in zip(axes, ["none", "gc", "pd"], strict=True):
        matrix = np.zeros((3, 4))
        for i, delay in enumerate([0, 0.01, 0.03]):
            for j, period in enumerate([0.001, 0.005, 0.01, 0.02]):
                group = [
                    r
                    for r in rows
                    if r["policy"].split("_")[1] == mode
                    and r["timestep_s"] == 0.0000125
                    and r["control_period_s"] == period
                    and r["delay_s"] == delay
                ]
                inside = [r for r in group if r["predicted_inside"]]
                success = sum(r["valid_target_reach"] for r in inside)
                matrix[i, j] = success / len(inside) if inside else 0
                groups.append(
                    dict(
                        mode=mode,
                        period_s=period,
                        delay_s=delay,
                        all_cases=len(group),
                        predicted_inside=len(inside),
                        inside_successes=success,
                        all_successes=sum(r["valid_target_reach"] for r in group),
                        coverage_or_rollout_errors=sum(r["status"] != "completed" for r in group),
                    )
                )
                ax.text(j, i, f"{success}/{len(inside)}", ha="center", va="center", color="black")
        ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues", alpha=0.6)
        ax.set(title=mode.upper(), xlabel="Control period (ms)")
        ax.set_xticks(range(4), [1, 5, 10, 20])
        ax.set_yticks(range(3), [0, 10, 30])
    axes[0].set_ylabel("Command delay (ms)")
    fig.suptitle("Valid target reaches / predicted-inside states · OF · fixed 12.5 µs physics")
    fig.tight_layout()
    fig.savefig(output / "timing.png", dpi=180)
    plt.close(fig)
    table(output / "timing_summary.csv", groups)
    return dict(
        episodes=len(results),
        valid_target_reaches=sum(r.get("valid_target_reach", False) for r in results),
        errors=sum(r["status"] != "completed" for r in results),
        physics_refinement_passed=sum(r["passed"] for r in refinement),
        physics_refinement_total=len(refinement),
        physics_refinement_evaluable=sum(r["assessment"] != "not_evaluable" for r in refinement),
        deadline_p99_evaluable=sum(r["control_p99_us"] is not None for r in rows),
        deadline_p99_passed=sum(
            r["control_p99_us"] is not None and r["control_p99_us"] <= r["control_period_s"] * 1e6
            for r in rows
        ),
        wallclock_workload=(
            "Concurrent experiment workload, not hard real-time scheduling or hardware latency."
        ),
        groups=groups,
    )


def energy_report(output):
    p = AUTHORS_PROFILES["YF"]
    rows = []
    for mode, q in itertools.product(
        ["none", "gc"], [-0.4, -0.25, -0.15, -0.08, -0.02, 0.02, 0.08, 0.15, 0.25, 0.4]
    ):
        v = float(zero_torque_velocity(q, p, mode))

        def derivative(_, state):
            angle, velocity = state
            return [
                velocity,
                (p.stiffness_nm * np.sin(angle) + exo_torque(angle, velocity, p, mode)) / p.inertia_kg_m2,
            ]

        solution = solve_ivp(derivative, (0, 3), [q, v], rtol=1e-10, atol=1e-12, max_step=0.005)
        if not solution.success:
            raise RuntimeError(solution.message)
        energy = 0.5 * p.inertia_kg_m2 * solution.y[1] ** 2 + potential(solution.y[0], p, mode)
        rows.append(
            dict(
                mode=mode,
                initial_q=q,
                initial_velocity=v,
                final_q=float(solution.y[0, -1]),
                final_velocity=float(solution.y[1, -1]),
                max_energy_residual_j=float(max(abs(energy))),
            )
        )
    table(output / "work_energy_ode.csv", rows)
    q = np.linspace(-0.3, 0.3, 1001)
    heel = -np.arcsin(0.19 * p.foot_length_m / p.length_m)
    toe = np.arcsin(0.81 * p.foot_length_m / p.length_m) - 0.03
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    for mode, color in [("none", "#c87825"), ("gc", "#6b4e9b")]:
        axes[0].plot(q, zero_torque_velocity(q, p, mode), color=color, label=mode.upper())
        low, high = stopping_envelope(q, p, mode, heel, toe)
        axes[1].plot(q, low, color=color, label=mode.upper())
        axes[1].plot(q, high, color=color)
    for ax in axes:
        ax.set(xlabel="Forward lean (rad)", ylabel="Velocity (rad/s)")
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_title("Zero-human-torque energy line")
    axes[1].set_title("Relaxed stopping envelope · no contact/rate bounds")
    fig.suptitle("YF · SI work-energy diagnostic")
    fig.tight_layout()
    fig.savefig(output / "work_energy.png", dpi=180)
    plt.close(fig)
    return dict(
        ode_runs=len(rows),
        max_energy_residual_j=max(r["max_energy_residual_j"] for r in rows),
        model="Undamped unconstrained YF; E=I*v²/2+U; saturated GC work integrated in SI.",
    )


def yf_report(root, output):
    specification = json.loads((root / "yf_validation/protocol.json").read_text())
    rows = []
    for case in specification["cases"]:
        directory = root / "yf_validation/cases" / case
        protocol = json.loads((directory / "protocol.json").read_text())
        results = json.loads((directory / "outcomes.json").read_text())
        if len(results) != 9 or [(r["group"], r["state"]) for r in results] != [
            (r["group"], r["state"]) for r in protocol["states"]
        ]:
            raise ValueError("Incomplete YF validation cohort")
        inside = [r for r in results if r["predicted_inside"]]
        rows.append(
            dict(
                case=case,
                selected=len(results),
                predicted_inside=len(inside),
                inside_successes=sum(r.get("valid_target_reach", False) for r in inside),
                all_successes=sum(r.get("valid_target_reach", False) for r in results),
                errors=sum(r["status"] == "rollout_error" for r in results),
            )
        )
    table(output / "yf_forward_validation.csv", rows)
    return dict(
        episodes=sum(r["selected"] for r in rows),
        predicted_inside=sum(r["predicted_inside"] for r in rows),
        inside_successes=sum(r["inside_successes"] for r in rows),
        all_successes=sum(r["all_successes"] for r in rows),
        positive_value_successes=sum(r["all_successes"] - r["inside_successes"] for r in rows),
        errors=sum(r["errors"] for r in rows),
        cases=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=["hjb", "paired", "contact", "timing", "energy", "yf"],
        default=["hjb", "paired", "contact", "timing", "energy", "yf"],
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((args.input / "hjb/manifest.json").read_text())["protocol"]
    summary = {"protocol_id": protocol["id"], "acceptance": protocol["acceptance"]}
    for part in args.parts:
        if part == "hjb":
            summary[part] = hjb_report(args.input, args.out, protocol["acceptance"])
        elif part == "paired":
            summary[part] = paired_report(args.input, args.out)
        elif part == "contact":
            summary[part] = contact_report(args.input, args.out)
        elif part == "timing":
            summary[part] = timing_report(args.input, args.out, protocol["acceptance"])
        elif part == "yf":
            summary[part] = yf_report(args.input, args.out)
        else:
            summary[part] = energy_report(args.out)
    write(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
