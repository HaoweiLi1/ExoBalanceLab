"""Render the predeclared paired cases from saved MuJoCo states; no outcome selection."""

import argparse
import json
from pathlib import Path

import numpy as np

from examples.validate_transport_feedback import report, specification, write
from exobalancelab.rendering import comparison_video, plot_comparison
from exobalancelab.simulation import Episode, RunConfig


def render(source, output):
    spec = specification()
    profiles = json.loads((source / "summary.json").read_text())["profiles"]
    report(source, profiles)
    output.mkdir(parents=True, exist_ok=False)
    manifest = []
    for profile in profiles:
        cases = json.loads((source / f"selection_{profile}.json").read_text())["samples"]
        for case in cases:
            paths = [
                source
                / "episodes"
                / f"{profile}_{mode}_{case['index']:02d}_dt{spec['physics_steps_s'][-1]:g}"
                for mode in ["none", "gc"]
            ]
            results = [json.loads((p / "result.json").read_text()) for p in paths]
            if not all(r["status"] == "completed" for r in results):
                continue
            episodes = []
            for path in paths:
                with np.load(path / "episode.npz", allow_pickle=False) as stored:
                    arrays = {name: stored[name] for name in stored.files}
                episodes.append(
                    Episode(
                        RunConfig(**json.loads((path / "config.json").read_text())),
                        (path / "model.xml").read_text(),
                        arrays,
                        json.loads((path / "metrics.json").read_text()),
                    )
                )
            goal_names = ["ANKLE" if r["selected_goal"] == 0 else "TOE" for r in results]
            titles = [
                f"{mode} / {goal} TARGET"
                for mode, goal in zip(["NO ASSISTANCE", "GRAVITY COMP."], goal_names, strict=True)
            ]
            comparison_video(episodes, output / f"transport_{profile.lower()}.mp4", titles=titles)
            plot_comparison(episodes, output / f"transport_{profile.lower()}_signals.png")
            manifest.append(
                dict(
                    profile=profile,
                    index=case["index"],
                    group=case["group"],
                    state=case["state"],
                    valid_reach=[r["valid_target_reach"] for r in results],
                    selected_goals=goal_names,
                    sources=[str(p.relative_to(source)) for p in paths],
                    physics_step_s=spec["physics_steps_s"][-1],
                    playback_speed=1.0,
                )
            )
            break
        else:
            manifest.append(dict(profile=profile, error="No completed pair in the frozen sample"))
    write(output / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    render(args.source, args.out)
