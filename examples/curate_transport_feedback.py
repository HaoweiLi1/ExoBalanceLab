"""Package complete feedback outcomes and compact provenance, excluding dense policy tables."""

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

from examples.validate_transport_feedback import report, specification, write


def curate(source, policies, output):
    spec = specification()
    profiles = list(dict.fromkeys(name.split("_")[0] for name in spec["families"]))
    result = report(source, profiles)
    expected = (
        len(profiles)
        * 2
        * len(spec["groups"])
        * spec["per_group_per_population"]
        * len(spec["physics_steps_s"])
    )
    if result["episodes"] != expected:
        raise ValueError("The complete declared experiment is required, including failed rollouts")
    output.mkdir(parents=True, exist_ok=False)
    for path in source.glob("*.json"):
        shutil.copy2(path, output / path.name)
    shutil.copytree(source / "sources", output / "sources")
    rows = []
    for episode in sorted((source / "episodes").iterdir()):
        target = output / "episodes" / episode.name
        target.mkdir(parents=True)
        for name in [
            "result.json",
            "config.json",
            "metrics.json",
            "episode.npz",
            "model.xml",
            "provenance.json",
        ]:
            if (episode / name).exists():
                shutil.copy2(episode / name, target / name)
        record = json.loads((episode / "result.json").read_text())
        rows.append(
            dict(
                profile=record["profile"],
                exo=record["exo"],
                index=record["index"],
                group=record["group"],
                physics_step_s=record["config"]["timestep_s"],
                predicted_inside=record["predicted_inside"],
                status=record["status"],
                valid_reach=record.get("valid_target_reach"),
                target_reached=record.get("target_reached"),
                contact_valid=record.get("contact_valid"),
                control_p99_us=record.get("metrics", {}).get("control_p99_us"),
                error=record.get("error"),
            )
        )
    with (output / "outcomes.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for name in spec["families"]:
        original = policies / name
        metadata = json.loads((original / "summary.json").read_text())
        if metadata["stage"] != "exported" or not metadata["accepted_projection_identical"]:
            raise ValueError("Every policy must match its accepted numerical reference")
        target = output / "policies" / name
        target.mkdir(parents=True)
        for filename in ["summary.json", "problem.json", "projection.mat"]:
            shutil.copy2(original / filename, target / filename)
        shutil.copytree(original / "sources", target / "sources")
    if report(output, profiles) != result:
        raise ValueError("Curated feedback outcomes differ")
    files = {
        str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(output.rglob("*"))
        if p.is_file()
    }
    write(output / "checksums.json", files)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    curate(args.source, args.policies, args.out)
