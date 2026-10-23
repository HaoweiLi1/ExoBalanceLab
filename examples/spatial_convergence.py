"""Prepare, solve and report the fixed-target spatial repair campaign.

Use the basic environment for preparation/reporting and environment-hjb.yml
for solving. The old validation_v1 directory is an immutable input.
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


def cases(protocol):
    source = json.loads((ROOT / "docs/results/validation_v1/hjb/manifest.json").read_text())
    families = [c for c in source["cases"] if c["resolution"] == 225 and c["cfl"] == 0.8]
    for family in families:
        for resolution in protocol["repair_grids"]:
            stem = family["id"].rsplit("_n", 1)[0]
            yield dict(
                family,
                id=stem + "_n" + "x".join(map(str, resolution)),
                source_id=family["id"],
                resolution=resolution,
                save_history=False,
            )


def prepare(protocol, output, baseline, authors, dependencies):
    import matlab
    import matlab.engine

    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    engine.addpath(str(ROOT / "matlab"), nargout=0)
    try:
        for case in cases(protocol):
            destination = output / "prepared" / case["id"]
            original = baseline / "prepared" / case["source_id"].rsplit("_cfl", 1)[0]
            before = json.loads((original / "summary.json").read_text())
            if not (destination / "summary.json").exists():
                print("PREPARE " + case["id"], flush=True)
                engine.feval(
                    "solve_paper_hjb",
                    str(authors),
                    str(dependencies),
                    str(destination),
                    "profile",
                    case["profile"],
                    "exo",
                    case["exo"],
                    "direction",
                    case["direction"],
                    "resolution",
                    matlab.double([[v] for v in case["resolution"]]),
                    "mt_scale",
                    float(case["mt_scale"]),
                    "mrtd_scale",
                    float(case["mrtd_scale"]),
                    "target_definition",
                    case["target_definition"],
                    "frozen_target_file",
                    str(original / "computed_target.mat"),
                    "target_field",
                    "width_normalized",
                    "align_upright",
                    True,
                    "align_zero",
                    True,
                    "prepare_only",
                    True,
                    "quiet",
                    True,
                    nargout=0,
                )
            after = json.loads((destination / "summary.json").read_text())
            for key in ["target", "effective_profile", "method", "duration_s", "output_step_s"]:
                if after[key] != before[key]:
                    raise ValueError("Changed physical problem: " + case["id"] + "/" + key)
            if after["target_field"] != "width_normalized" or not after["aligned_upright"]:
                raise ValueError("Different target discretization")
    finally:
        engine.quit()


def solve(protocol, output):
    environment = dict(
        os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform"
    )
    for case in cases(protocol):
        problem = output / "prepared" / case["id"] / "problem.mat"
        destination = output / "solved" / case["id"]
        with problem.open("rb") as file:
            problem_hash = hashlib.file_digest(file, "sha256").hexdigest()
        if (destination / "summary.json").exists():
            previous = json.loads((destination / "summary.json").read_text())
            if previous["stage"] != "solved" or previous["problem_sha256"] != problem_hash:
                raise ValueError("Incomplete or different run: " + str(destination))
            continue
        log = output / "solver_logs" / (case["id"] + ".txt")
        log.parent.mkdir(parents=True, exist_ok=True)
        print("SOLVE " + case["id"], flush=True)
        with log.open("w") as file:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "exobalancelab.hjb_solver",
                    "--problem",
                    str(problem),
                    "--out",
                    str(destination),
                    "--projection-only",
                    "--flux",
                    "godunov",
                ],
                env=environment,
                stdout=file,
                stderr=subprocess.STDOUT,
                check=True,
            )


def report(protocol, output):
    rows, arrays, summaries = [], {}, {}
    for case in cases(protocol):
        path = output / "solved" / case["id"]
        summary = json.loads((path / "summary.json").read_text())
        if summary["stage"] != "solved" or summary["flux"] != "godunov":
            raise ValueError("Unsolved or different flux: " + case["id"])
        source = loadmat(path / "projection.mat")
        arrays[case["id"]] = (
            source["theta_rad"].ravel(),
            source["velocity_rad_s"].ravel(),
            source["projected_value"],
        )
        summaries[case["id"]] = summary
        rows.append(
            dict(
                case,
                area=area(*arrays[case["id"]], case["direction"]),
                actual_resolution=summary["resolution"],
                solver_sha256=summary["solver_sha256"],
                projection_sha256=hashlib.sha256((path / "projection.mat").read_bytes()).hexdigest(),
            )
        )
    for row in rows:
        baseline = next(
            r
            for r in rows
            if r["exo"] == "none"
            and all(
                r[k] == row[k]
                for k in ["profile", "direction", "target_definition", "mt_scale", "mrtd_scale", "resolution"]
            )
        )
        row["assistance_effect_percent"] = 100 * (row["area"] / baseline["area"] - 1)
    checks = []
    limits = protocol["frozen_acceptance"]
    for coarse_grid, fine_grid in zip(protocol["repair_grids"][:-1], protocol["repair_grids"][1:]):
        for fine in [r for r in rows if r["resolution"] == fine_grid]:
            coarse = next(
                r for r in rows if r["resolution"] == coarse_grid and r["source_id"] == fine["source_id"]
            )
            a, b = summaries[coarse["id"]], summaries[fine["id"]]
            for key in [
                "target",
                "effective_profile",
                "solver_sha256",
                "method",
                "duration_s",
                "cfl",
                "target_field",
            ]:
                if a[key] != b[key]:
                    raise ValueError("Incomparable grids: " + key)
            error = 100 * abs(fine["area"] - coarse["area"]) / fine["area"]
            effect = abs(fine["assistance_effect_percent"] - coarse["assistance_effect_percent"])
            iou = overlap(arrays[coarse["id"]], arrays[fine["id"]], fine["direction"])
            checks.append(
                dict(
                    coarse=coarse["id"],
                    fine=fine["id"],
                    area_error_percent=error,
                    assistance_effect_change_pp=effect,
                    projection_iou=iou,
                    passed=bool(
                        error <= limits["spatial_area_relative_percent"]
                        and effect <= limits["spatial_assistance_effect_pp"]
                        and iou >= limits["common_grid_projection_iou"]
                    ),
                )
            )
    destination = output / "report"
    destination.mkdir(parents=True, exist_ok=True)
    table(destination / "areas.csv", rows)
    table(destination / "spatial_acceptance.csv", checks)
    result = dict(
        cases=len(rows),
        checks=len(checks),
        passed=sum(c["passed"] for c in checks),
        limits=limits,
        comparisons=checks,
    )
    write(destination / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "solve", "report"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/spatial_convergence_v2.json")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/spatial_convergence/repair")
    parser.add_argument("--baseline", type=Path, default=ROOT / "logs/validation_v1/hjb")
    parser.add_argument("--authors", type=Path, help="Pinned AnkleExoBRS checkout; required for preparation")
    parser.add_argument("--dependencies", type=Path, default=ROOT / "logs/hjb_dependencies")
    args = parser.parse_args()
    if args.stage == "prepare" and args.authors is None:
        parser.error("prepare requires --authors pointing to the pinned AnkleExoBRS checkout")
    protocol = json.loads(args.protocol.read_text())
    original = json.loads((ROOT / "experiments/validation_v1.json").read_text())
    if protocol["frozen_acceptance"] != original["acceptance"]:
        raise ValueError("The frozen acceptance criteria must not change")
    args.out.mkdir(parents=True, exist_ok=True)
    frozen = args.out / "protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("Use a new output directory for a changed protocol")
    write(frozen, protocol)
    if args.stage == "prepare":
        prepare(protocol, args.out, args.baseline, args.authors, args.dependencies)
    elif args.stage == "solve":
        solve(protocol, args.out)
    else:
        report(protocol, args.out)


if __name__ == "__main__":
    main()
