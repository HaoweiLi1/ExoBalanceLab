"""Rebuild selected reports from committed records; no new physics or HJB solves."""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from exobalancelab.rendering import comparison_video, plot_comparison
from exobalancelab.simulation import Episode, RunConfig

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("logs/rebuilt_results"))
    parser.add_argument("--video", action="store_true", help="also render the two recorded comparisons")
    args = parser.parse_args()
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    for name, protocol, grid in [
        ("population", "hjb_paper225/population", False),
        ("of_grid", "hjb_grid_study/comparison", True),
        ("om_grid", "hjb_om_gc_followup/comparison", True),
    ]:
        command = [sys.executable, "-m", "exobalancelab.hjb_regions"]
        if grid:
            command.append("--grid-study")
        subprocess.run(
            command
            + ["--protocol", f"docs/results/{protocol}/curated_protocol.json", "--out", str(output / name)],
            cwd=ROOT,
            check=True,
        )
    subprocess.run(
        [
            sys.executable,
            "docs/results/hjb_om_target_override/report_target_override.py",
            "docs/results/hjb_om_target_override/curated_protocol.json",
            str(output / "target_diagnostic"),
        ],
        cwd=ROOT,
        check=True,
    )
    command = [sys.executable, "-m", "exobalancelab.hjb_comparison"]
    for label, directory in [
        ("Source domain / 100 ms", "baseline"),
        ("Wider domain / 100 ms", "buffer05_dt01"),
        ("Wider domain / 10 ms", "buffer05_dt001"),
    ]:
        command += ["--run", f"{label}=docs/results/hjb_refinement/{directory}"]
    subprocess.run(command + ["--out", str(output / "time_comparison")], cwd=ROOT, check=True)
    transport_source = ROOT / "docs/results/transport_v3"
    transport = output / "transport"
    shutil.copytree(transport_source, transport)
    for module, arguments in [
        (
            "examples.benchmark_transport",
            ["--report-only", "--out", str(transport / "analytic_reference")],
        ),
        ("examples.transport_convergence", ["report", "--out", str(transport)]),
        (
            "examples.plot_transport_convergence",
            ["--reference", str(transport / "analytic_reference"), "--out", str(transport / "exact.png")],
        ),
        (
            "examples.plot_transport_convergence",
            ["--report", str(transport / "report/summary.json"), "--out", str(transport / "spatial.png")],
        ),
    ]:
        subprocess.run([sys.executable, "-m", module, *arguments], cwd=ROOT, check=True)
    for relative in ["analytic_reference/summary.json", "report/summary.json"]:
        if json.loads((transport / relative).read_text()) != json.loads(
            (transport_source / relative).read_text()
        ):
            raise ValueError("Rebuilt transport results differ from the recorded report: " + relative)
    for name in ["human", "paper_constraints"]:
        episodes = []
        for mode in ["none", "gc", "pd"]:
            source = ROOT / "docs" / "results" / name / mode
            with np.load(source / "episode.npz", allow_pickle=False) as stored:
                arrays = {key: stored[key] for key in stored.files}
            episodes.append(
                Episode(
                    RunConfig(**json.loads((source / "config.json").read_text())),
                    (source / "model.xml").read_text(),
                    arrays,
                    json.loads((source / "metrics.json").read_text()),
                )
            )
        plot_comparison(episodes, output / f"{name}_signals.png")
        if args.video:
            comparison_video(episodes, output / f"{name}.mp4")
    print(f"Rebuilt reports from committed records in {output}")


if __name__ == "__main__":
    main()
