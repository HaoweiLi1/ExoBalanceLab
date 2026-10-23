"""Publish compact inputs for rebuilding the validation report, excluding large fields."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)

    def copy(relative):
        source, destination = args.input / relative, args.out / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    manifest = json.loads((args.input / "hjb/manifest.json").read_text())
    copy(Path("hjb/manifest.json"))
    copy(Path("hjb/failures.json"))
    preparation_index = {}
    for case in manifest["cases"]:
        directory = Path("hjb/solved") / case["id"]
        summary = json.loads((args.input / directory / "summary.json").read_text())
        if summary["stage"] != "solved":
            raise ValueError("Incomplete solve: " + case["id"])
        for name in ["summary.json", "projection.mat"]:
            copy(directory / name)
        snapshot = args.input / directory / "source.py"
        digest = hashlib.sha256(snapshot.read_bytes()).hexdigest()
        if digest != summary["solver_sha256"]:
            raise ValueError("Solver snapshot hash mismatch")
        destination = args.out / "solver_sources" / (digest + ".py")
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(snapshot, destination)
        prepared = Path("hjb/prepared") / case["id"].rsplit("_cfl", 1)[0]
        copy(prepared / "computed_target.mat")
        sources = {
            str(path.relative_to(args.input / prepared / "source")): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in sorted((args.input / prepared / "source").rglob("*.m"))
        }
        fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        for relative in sources:
            destination = args.out / "preparation_sources" / fingerprint / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.input / prepared / "source" / relative, destination)
        preparation_index[case["id"]] = dict(
            target=str(prepared / "computed_target.mat"), sources=fingerprint, files=sources
        )
    (args.out / "preparation_index.json").write_text(json.dumps(preparation_index, indent=2) + "\n")
    for study in [
        "paired",
        "timing",
        "paired_refinement",
        "paired_design_coupled",
        "paired_design_coupled_refinement",
    ]:
        for name in ["protocol.json", "outcomes.json", "source_hashes.json"]:
            copy(Path(study) / name)
    for relative in [
        "paired_refinement/contact_diagnosis.json",
        "timing/outcomes_instantaneous_endpoint.json",
        "timing/endpoint_reassessment.json",
    ]:
        if (args.input / relative).exists():
            copy(Path(relative))
    if (args.input / "contact_followup/diagnosis.json").exists():
        for name in ["protocol.json", "source_hashes.json", "outcomes.json", "diagnosis.json"]:
            copy(Path("contact_followup") / name)
        outcomes = json.loads((args.input / "contact_followup/outcomes.json").read_text())
        for case in outcomes:
            if case["status"] == "completed":
                copy(Path("contact_followup/cases") / case["id"] / "contact_window.npz")
    yf = json.loads((args.input / "yf_validation/protocol.json").read_text())
    for name in ["protocol.json", "source_hashes.json"]:
        copy(Path("yf_validation") / name)
    for case in yf["cases"]:
        for name in ["protocol.json", "outcomes.json", "projection.png"]:
            copy(Path("yf_validation/cases") / case / name)
    for pair in ["OF_paper_fixed_dt0.0005_state02", "OF_human_fixed_dt0.0005_state07"]:
        for mode in ["none", "gc", "pd"]:
            directory = Path("paired/cases") / (pair + "_" + mode)
            for name in ["result.json", "trace.npz", "trace_metadata.json", "model.xml"]:
                copy(directory / name)
    checksums = {
        str(path.relative_to(args.out)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(args.out.rglob("*"))
        if path.is_file()
    }
    (args.out / "checksums.json").write_text(json.dumps(checksums, indent=2) + "\n")
    print(f"Curated {len(checksums)} files; full fields, caches and unselected traces remain local.")


if __name__ == "__main__":
    main()
