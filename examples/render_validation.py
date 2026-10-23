"""Render one recorded none/GC/PD triplet; select a case explicitly in the report."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from exobalancelab.rendering import comparison_video, plot_comparison
from exobalancelab.simulation import Episode, RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--pair", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    episodes, records = [], []
    for mode in ["none", "gc", "pd"]:
        path = args.cases / (args.pair + "_" + mode)
        result = json.loads((path / "result.json").read_text())
        if result["status"] != "completed":
            raise ValueError("A failed simulation has no complete trace to render")
        with np.load(path / "trace.npz") as data:
            arrays = dict(data)
        episodes.append(
            Episode(
                RunConfig(**result["config"]), (path / "model.xml").read_text(), arrays, result["metrics"]
            )
        )
        records.append(
            dict(
                case=path.name,
                trace_sha256=hashlib.sha256((path / "trace.npz").read_bytes()).hexdigest(),
                result_sha256=hashlib.sha256((path / "result.json").read_bytes()).hexdigest(),
            )
        )
    comparison_video(episodes, args.out)
    plot_comparison(episodes, args.out.with_name(args.out.stem + "_signals.png"))
    args.out.with_suffix(".json").write_text(
        json.dumps(
            {
                "pair": args.pair,
                "sources": records,
                "selection": (
                    "Explicit illustration chosen after the full cohort; all outcomes remain in the report."
                ),
                "rendering": "Recorded states at 10 ms intervals; no new integration during rendering.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
