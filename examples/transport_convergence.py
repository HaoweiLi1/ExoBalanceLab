"""Solve and assess all 18 frozen human families after analytic acceptance.

The original 225-point preparations supply identical physical domains, goals and
MT/MRTD parameters at every refinement. Compact problem.json copies can be used
instead, so the new transport campaign does not require repeating MATLAB/SDP.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from scipy.io import loadmat

from examples.report_validation import area, overlap, table, write

ROOT = Path(__file__).resolve().parents[1]


def protocol():
    specification = json.loads((ROOT / "experiments/transport_convergence_v3.json").read_text())
    original = json.loads((ROOT / "experiments/validation_v1.json").read_text())
    if specification["acceptance"] != original["acceptance"]:
        raise ValueError("The original numerical standards must not change")
    return specification


def family_name(case):
    return case["id"].rsplit("_n", 1)[0]


def identifier(name, n, step):
    return f"{name}_n{n}_dt{step:g}"


def run(prepared, reference, output, resolutions, time_step, selected):
    """Run only with the exact numerical sources that passed the analytic cohort."""
    from exobalancelab.hjb_transport import load_problem

    specification = protocol()
    measured = json.loads((reference / "summary.json").read_text())
    if not measured["passed"] or measured["acceptance"] != specification["analytic_acceptance"]:
        raise ValueError("The independent analytic/time references have not passed")
    for filename, digest in measured["sources"].items():
        folder = (
            "examples"
            if filename in {"benchmark_transport.py", "hjb_analytic_reference.py"}
            else "exobalancelab"
        )
        if hashlib.sha256((ROOT / folder / filename).read_bytes()).hexdigest() != digest:
            raise ValueError("Repeat the analytic reference for the changed source: " + filename)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "protocol.json").exists() and json.loads(
        (output / "protocol.json").read_text()
    ) != specification:
        raise ValueError("Use a new output directory for a different protocol")
    write(output / "protocol.json", specification)
    families = specification["families"]
    if selected:
        if set(selected) - {family_name(c) for c in families}:
            raise ValueError("Unknown family selection")
        families = [c for c in families if family_name(c) in selected]
    environment = dict(os.environ)
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    for case in families:
        name = family_name(case)
        source = prepared / (name + ".json")
        if not source.exists():
            source = prepared / (name + "_n225x225x225") / "problem.mat"
        inputs = load_problem(source)
        physical = json.dumps(inputs, sort_keys=True, indent=2, allow_nan=False) + "\n"
        digest = hashlib.sha256(physical.encode()).hexdigest()
        for n in resolutions:
            key = identifier(name, n, time_step)
            destination = output / "solved" / key
            if (destination / "summary.json").exists():
                previous = json.loads((destination / "summary.json").read_text())
                expected_sources = {
                    k: v
                    for k, v in measured["sources"].items()
                    if k not in {"benchmark_transport.py", "hjb_analytic_reference.py"}
                }
                if (
                    previous["stage"] != "solved"
                    or previous["sources"] != expected_sources
                    or previous["input_sha256"] != digest
                    or previous["resolution"] != [n] * 3
                    or previous["requested_time_step_s"] != time_step
                ):
                    raise ValueError("Incomplete or different solver/input: " + key)
                continue
            log = output / "solver_logs" / (key + ".log")
            log.parent.mkdir(exist_ok=True)
            write(
                log.with_suffix(".environment.json"),
                {
                    key: environment.get(key)
                    for key in [
                        "JAX_PLATFORMS",
                        "XLA_PYTHON_CLIENT_PREALLOCATE",
                        "XLA_PYTHON_CLIENT_ALLOCATOR",
                        "XLA_CLIENT_MEM_FRACTION",
                        "XLA_PYTHON_CLIENT_MEM_FRACTION",
                        "OPENBLAS_NUM_THREADS",
                    ]
                },
            )
            print("START " + key, flush=True)
            with log.open("w") as file:
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "exobalancelab.hjb_transport",
                        "--problem",
                        str(source),
                        "--out",
                        str(destination),
                        "--resolution",
                        str(n),
                        "--step",
                        str(time_step),
                        "--projection-only",
                    ],
                    env=environment,
                    stdout=file,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print("DONE " + key, flush=True)


def report(output, resolutions, temporal_resolution):
    """Require every family at the finest pair and both preselected time pilots."""
    specification = protocol()
    if json.loads((output / "protocol.json").read_text()) != specification:
        raise ValueError("Stored protocol differs from the frozen campaign")
    if len(resolutions) < 2 or any(b <= a for a, b in zip(resolutions[:-1], resolutions[1:])):
        raise ValueError("Provide at least two increasing resolutions")
    limits = specification["acceptance"]
    primary = specification["primary_time_step_s"]
    rows, fields, summaries = {}, {}, {}
    for case in specification["families"]:
        name = family_name(case)
        grids = [(n, primary) for n in resolutions]
        if name in specification["pilots"]:
            grids += [(temporal_resolution, s) for s in specification["temporal_comparison_steps_s"]]
        for n, dt in dict.fromkeys(grids):
            key = identifier(name, n, dt)
            source = output / "solved" / key
            summary = json.loads((source / "summary.json").read_text())
            if (
                summary["stage"] != "solved"
                or summary["resolution"] != [n] * 3
                or summary["requested_time_step_s"] != dt
            ):
                raise ValueError("Incomplete or incompatible solve: " + key)
            for field in ["profile", "exo", "direction", "target_definition", "mt_scale", "mrtd_scale"]:
                if summary[field] != case[field]:
                    raise ValueError("Changed case contract: " + key + "/" + field)
            if (
                hashlib.sha256((source / "projection.mat").read_bytes()).hexdigest()
                != summary["projection_sha256"]
            ):
                raise ValueError("Changed projection: " + key)
            if hashlib.sha256((source / "problem.json").read_bytes()).hexdigest() != summary["input_sha256"]:
                raise ValueError("Changed physical input: " + key)
            for filename, digest in summary["sources"].items():
                archived = source / filename
                if not archived.exists():
                    archived = output / "sources" / filename
                if hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
                    raise ValueError("Changed archived numerical source: " + filename)
            data = loadmat(source / "projection.mat")
            field = data["theta_rad"].ravel(), data["velocity_rad_s"].ravel(), data["projected_value"]
            fields[key], summaries[key] = field, summary
            rows[key] = dict(
                id=key,
                family=name,
                resolution=n,
                time_step_s=dt,
                area=area(*field, case["direction"]),
                **{
                    k: case[k]
                    for k in ["profile", "exo", "direction", "target_definition", "mt_scale", "mrtd_scale"]
                },
            )
    for row in rows.values():
        baseline = next(
            r
            for r in rows.values()
            if r["exo"] == "none"
            and all(
                r[k] == row[k]
                for k in [
                    "profile",
                    "direction",
                    "target_definition",
                    "mt_scale",
                    "mrtd_scale",
                    "resolution",
                    "time_step_s",
                ]
            )
        )
        row["assistance_effect_percent"] = 100 * (row["area"] / baseline["area"] - 1)

    def comparison(coarse, fine, temporal=False):
        a, b = summaries[coarse["id"]], summaries[fine["id"]]
        for key in ["target", "effective_profile", "method", "duration_s", "sources", "input_sha256"]:
            if a[key] != b[key]:
                raise ValueError("Changed comparison contract: " + key)
        error = 100 * abs(fine["area"] - coarse["area"]) / fine["area"]
        iou = overlap(fields[coarse["id"]], fields[fine["id"]], fine["direction"])
        effect = abs(fine["assistance_effect_percent"] - coarse["assistance_effect_percent"])
        passed = (
            error <= limits["temporal_area_relative_percent" if temporal else "spatial_area_relative_percent"]
            and iou >= limits["temporal_projection_iou" if temporal else "common_grid_projection_iou"]
        )
        if not temporal:
            passed = passed and effect <= limits["spatial_assistance_effect_pp"]
        return dict(
            coarse=coarse["id"],
            fine=fine["id"],
            family=fine["family"],
            area_error_percent=error,
            projection_iou=iou,
            assistance_effect_change_pp=effect,
            passed=bool(passed),
        )

    checks = []
    for left, right in zip(resolutions[:-1], resolutions[1:]):
        for case in specification["families"]:
            name = family_name(case)
            checks.append(
                comparison(rows[identifier(name, left, primary)], rows[identifier(name, right, primary)])
            )
    temporal = [
        comparison(
            *(
                rows[identifier(name, temporal_resolution, dt)]
                for dt in specification["temporal_comparison_steps_s"]
            ),
            temporal=True,
        )
        for name in specification["pilots"]
    ]
    finest = checks[-len(specification["families"]) :]
    destination = output / "report"
    destination.mkdir(exist_ok=True)
    table(destination / "areas.csv", list(rows.values()), lineterminator="\n")
    table(destination / "spatial.csv", checks, lineterminator="\n")
    table(destination / "temporal.csv", temporal, lineterminator="\n")
    result = dict(
        scope="Projected reduced-model regions: spatial and numerical-time refinement",
        acceptance=limits,
        finest_spatial_pair=resolutions[-2:],
        spatial=checks,
        temporal=temporal,
        spatial_passed=sum(c["passed"] for c in finest),
        spatial_total=len(finest),
        temporal_passed=sum(c["passed"] for c in temporal),
        temporal_total=len(temporal),
        accepted=all(c["passed"] for c in finest + temporal),
    )
    write(destination / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["solve", "report"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[561, 673])
    parser.add_argument("--step", type=float, default=0.01)
    parser.add_argument("--families", nargs="+")
    parser.add_argument("--temporal-resolution", type=int, default=673)
    args = parser.parse_args()
    if args.stage == "solve":
        if args.prepared is None or args.reference is None:
            parser.error("solve requires --prepared and --reference")
        run(args.prepared, args.reference, args.out, args.resolutions, args.step, args.families)
    else:
        result = report(args.out, args.resolutions, args.temporal_resolution)
        raise SystemExit(0 if result["accepted"] else 1)
