"""Prepare/solve the predeclared target, MT/MRTD and numerical-error campaign.

Run preparation in the MATLAB Engine environment, solving in environment-hjb.yml.
Each solve is a fresh process so GPU allocations cannot accumulate across cases.
The manifest is frozen before preparing any problem; resumptions verify its hash.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cases(protocol):
    """Unique, predeclared cases; independent scans share their nominal baseline."""
    h = protocol["hjb"]
    items = []

    def add(profile, mode, direction, n, mt=1, mrtd=1, target=None, cfl=0.8, history=False):
        target = target or h["target_definition"]
        name = (
            f"{profile}_{mode}_{direction}_{target}_mt{round(mt * 100)}"
            f"_mrtd{round(mrtd * 100)}_n{n}_cfl{cfl:g}"
        )
        if any(item["id"] == name for item in items):
            return
        items.append(
            dict(
                id=name,
                profile=profile,
                exo=mode,
                direction=direction,
                resolution=n,
                mt_scale=mt,
                mrtd_scale=mrtd,
                target_definition=target,
                cfl=cfl,
                save_history=history,
            )
        )

    # Establish target dependence first, before interpreting strength effects.
    for n in [h["primary_resolution"], h["refinement_resolution"]]:
        for target in ["recomputed", "source_entry", "common_ankle"]:
            for mode in h["strength_modes"]:
                add("OM", mode, "forward", n, target=target)
    for n in [h["primary_resolution"], h["refinement_resolution"]]:
        scales = (
            (
                [(1, 1)]
                + [(x, 1) for x in h["strength_scales"][1:]]
                + [(1, x) for x in h["strength_scales"][1:]]
            )
            if n == h["primary_resolution"]
            else h["refinement_scales"]
        )
        for mt, mrtd in scales:
            for mode in h["strength_modes"]:
                for direction in h["directions"]:
                    add(
                        "YF",
                        mode,
                        direction,
                        n,
                        mt,
                        mrtd,
                        history=n == h["primary_resolution"] and mt == mrtd == 1,
                    )
    for mode in h["strength_modes"]:
        add("YF", mode, "forward", h["primary_resolution"], cfl=h["cfl_pair"][1])
    return items


def manifest(protocol_path, output):
    raw = protocol_path.read_bytes()
    specification = {
        "protocol_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol": json.loads(raw),
        "cases": cases(json.loads(raw)),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists():
        if json.loads(path.read_text()) != specification:
            raise ValueError("Existing manifest differs; use a new campaign directory")
    else:
        path.write_text(json.dumps(specification, indent=2) + "\n")
    return specification


def prepare(spec, output, authors, dependencies):
    import matlab
    import matlab.engine

    engine = matlab.engine.start_matlab("-nodesktop -nosplash")
    try:
        engine.addpath(str(ROOT / "matlab"), nargout=0)
        h = spec["protocol"]["hjb"]
        for item in spec["cases"]:
            # CFL changes reuse exactly the same prepared fields.
            path = output / "prepared" / item["id"].rsplit("_cfl", 1)[0]
            if (path / "summary.json").exists():
                continue
            print("Preparing " + item["id"], flush=True)
            engine.feval(
                "solve_paper_hjb",
                str(authors),
                str(dependencies),
                str(path),
                "profile",
                item["profile"],
                "exo",
                item["exo"],
                "direction",
                item["direction"],
                "resolution",
                matlab.double([[item["resolution"]]] * 3),
                "mt_scale",
                float(item["mt_scale"]),
                "mrtd_scale",
                float(item["mrtd_scale"]),
                "target_definition",
                item["target_definition"],
                "prepare_only",
                True,
                "align_zero",
                h["align_zero"],
                "duration_s",
                h["horizon_s"],
                "output_step_s",
                h["output_step_s"],
                "velocity_buffer_rad_s",
                h["velocity_buffer_rad_s"],
                "method",
                h["method"],
                "quiet",
                True,
                nargout=0,
            )
    finally:
        engine.quit()


def solve_all(spec, output):
    env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_ALLOCATOR="platform")
    failures = []
    for item in spec["cases"]:
        result = output / "solved" / item["id"]
        if (result / "summary.json").exists() and json.loads((result / "summary.json").read_text()).get(
            "stage"
        ) == "solved":
            continue
        prepared = output / "prepared" / item["id"].rsplit("_cfl", 1)[0] / "problem.mat"
        if not prepared.exists():
            failures.append(dict(id=item["id"], error="No prepared input"))
            continue
        command = [
            sys.executable,
            "-m",
            "exobalancelab.hjb_solver",
            "--problem",
            str(prepared),
            "--out",
            str(result),
            "--cfl",
            str(item["cfl"]),
        ]
        if not item["save_history"]:
            command += ["--projection-only"]
        log = output / "solver_logs" / (item["id"] + ".txt")
        log.parent.mkdir(exist_ok=True)
        print("Solving " + item["id"], flush=True)
        with log.open("a") as file:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=file, stderr=subprocess.STDOUT)
        if process.returncode:
            failures.append(dict(id=item["id"], returncode=process.returncode, log=str(log)))
        (output / "failures.json").write_text(json.dumps(failures, indent=2) + "\n")
    if failures:
        raise RuntimeError(f"{len(failures)} solves failed; see failures.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "solve", "manifest"])
    parser.add_argument("--protocol", type=Path, default=ROOT / "experiments/validation_v1.json")
    parser.add_argument("--out", type=Path, default=ROOT / "logs/validation_v1/hjb")
    parser.add_argument("--authors", type=Path)
    parser.add_argument("--dependencies", type=Path)
    args = parser.parse_args()
    spec = manifest(args.protocol, args.out.resolve())
    if args.command == "prepare":
        if args.authors is None or args.dependencies is None:
            parser.error("Preparation requires --authors and --dependencies")
        prepare(spec, args.out.resolve(), args.authors.resolve(), args.dependencies.resolve())
    elif args.command == "solve":
        solve_all(spec, args.out.resolve())
    print(f"{len(spec['cases'])} cases: {args.out.resolve()}")


if __name__ == "__main__":
    main()
