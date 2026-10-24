"""Curate compact, hash-checked transport evidence without dense 3D fields."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from examples.report_validation import write
from examples.transport_convergence import family_name, identifier, protocol, report


def curate(source, reference, physics, destination, resolutions, temporal_resolution):
    """Copy every required family and time pilot, preserving the actual source files."""
    if destination.exists():
        raise FileExistsError(destination)
    specification = protocol()
    result = report(source, resolutions, temporal_resolution)
    if not result["accepted"]:
        raise ValueError("Full spatial/time acceptance is required for this release dataset")
    reference_summary = json.loads((reference / "summary.json").read_text())
    physics_summary = json.loads((physics / "summary.json").read_text())
    if (
        not reference_summary["passed"]
        or not physics_summary["passed"]
        or len(physics_summary["cases"]) != 18
    ):
        raise ValueError("Independent reference/physics evidence is incomplete")
    destination.mkdir(parents=True)
    shutil.copy2(source / "protocol.json", destination / "protocol.json")
    shutil.copytree(reference, destination / "analytic_reference")
    shutil.copytree(physics, destination / "physics")
    (destination / "sources").mkdir()
    (destination / "inputs").mkdir()
    sources = {}
    runs = []
    for case in specification["families"]:
        name = family_name(case)
        grids = [(n, specification["primary_time_step_s"]) for n in resolutions]
        if name in specification["pilots"]:
            grids += [(temporal_resolution, dt) for dt in specification["temporal_comparison_steps_s"]]
        for n, dt in dict.fromkeys(grids):
            key = identifier(name, n, dt)
            original = source / "solved" / key
            target = destination / "solved" / key
            summary = json.loads((original / "summary.json").read_text())
            if summary["sources"] != physics_summary["sources"]:
                raise ValueError("Different numerical sources in the physics audit")
            if any(reference_summary["sources"][k] != v for k, v in summary["sources"].items()):
                raise ValueError("Different solver sources in the analytic acceptance")
            target.mkdir(parents=True)
            for filename in ["summary.json", "projection.mat", "problem.json"]:
                shutil.copy2(original / filename, target / filename)
            environment = source / "solver_logs" / (key + ".environment.json")
            if environment.exists():
                shutil.copy2(environment, target / "environment.json")
            for filename, digest in summary["sources"].items():
                archived = destination / "sources" / filename
                if archived.exists() and hashlib.sha256(archived.read_bytes()).hexdigest() != digest:
                    raise ValueError("Mixed numerical sources in one campaign")
                shutil.copy2(original / filename, archived)
                sources[filename] = digest
            shutil.copy2(original / "problem.json", destination / "inputs" / (name + ".json"))
            runs.append(
                dict(
                    id=key,
                    input_sha256=summary["input_sha256"],
                    projection_sha256=summary["projection_sha256"],
                )
            )
    write(
        destination / "manifest.json",
        dict(
            protocol=specification["id"],
            sources=sources,
            cases=runs,
            resolutions=resolutions,
            temporal_resolution=temporal_resolution,
            contents="Compact projections and frozen inputs; no dense 3D fields or feedback histories",
        ),
    )
    rebuilt = report(destination, resolutions, temporal_resolution)
    if rebuilt != result:
        raise ValueError("Curated report differs from the actual run report")
    return rebuilt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--physics", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resolutions", type=int, nargs="+", default=[561, 673])
    parser.add_argument("--temporal-resolution", type=int, default=673)
    args = parser.parse_args()
    curate(args.source, args.reference, args.physics, args.out, args.resolutions, args.temporal_resolution)
