"""Offline MuJoCo evidence videos with synchronized state/torque and contact overlays.

Renders recorded physics states; no visual interpolation modifies the simulation.
Frames are streamed to FFmpeg, so memory usage is independent of video duration.
"""

from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from exobalancelab.simulation import Episode

MODES = {"none": "NO ASSISTANCE", "gc": "GRAVITY COMPENSATION", "pd": "PD ASSISTANCE"}
COLORS = {"none": "#9fc9db", "gc": "#64d9ba", "pd": "#ffad70"}


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def _overlay(frame: np.ndarray, episode: Episode, index: int, title: str) -> Image.Image:
    scene = Image.fromarray(frame)
    canvas = Image.new("RGB", (scene.width, scene.height + 244), "#0c1723")
    canvas.paste(scene, (0, 88))
    draw = ImageDraw.Draw(canvas)
    width, height = canvas.size
    a, mode = episode.arrays, episode.config.exo
    draw.rectangle((0, 0, width, 88), fill="#0c1723")
    draw.text((20, 12), title, font=_font(21), fill=COLORS[mode])
    draw.text(
        (20, 44),
        f"{episode.config.profile}  |  t = {a['time_s'][index]:.2f} s",
        font=_font(17),
        fill="#d7e2ec",
    )
    y = height - 156
    draw.rectangle((0, y, width, height), fill="#0c1723")
    human = (
        a["human_applied_nm"][index, [2, 5]].sum()
        if episode.config.model == "human"
        else a["human_torque_nm"][index]
    )
    exo = a["exo_torque_nm"][index]
    draw.text((20, y + 10), f"Human ankle {human:+6.1f} Nm", font=_font(18), fill="#9fc9db")
    draw.text((20, y + 38), f"Device ankle {exo:+6.1f} Nm", font=_font(18), fill="#ffad70")
    draw.text(
        (20, y + 67),
        f"Lean {np.degrees(a['lean_rad'][index]):+5.1f} deg | speed {a['velocity_rad_s'][index]:+.3f} rad/s",
        font=_font(15),
        fill="white",
    )
    valid = (
        a["normal_n"][index] > 0
        and a["cop_margin_m"][index] >= 0
        and a["friction_margin_n"][index] >= 0
        and a["foot_pitch_rad"][index] < 0.035
        and a["foot_slide_m"][index] < 0.02
    )
    label = "CONTACT VALID" if valid else "CONTACT CONSTRAINT LOST"
    draw.text((20, y + 95), label, font=_font(15), fill="#64d9ba" if valid else "#ff827b")
    source = "Required reactions" if episode.config.model == "paper" else "Measured floor contact"
    draw.text((20, y + 121), source, font=_font(13), fill="#b2bdcb")
    if a["push_n"][index] != 0:
        draw.text((20, 102), f"PUSH {a['push_n'][index]:+.0f} N", font=_font(22), fill="#ffad70")
    return canvas


def comparison_video(
    episodes: list[Episode],
    output: str | Path,
    fps: int = 30,
    width: int = 480,
    height: int = 640,
    titles: list[str] | None = None,
) -> Path:
    """Render synchronized panels, an MP4, a poster and a compact animated preview."""
    if not episodes or fps <= 0:
        raise ValueError("At least one episode and positive fps are required")
    if titles is None:
        titles = [MODES[episode.config.exo] for episode in episodes]
    if len(titles) != len(episodes):
        raise ValueError("Provide one title per episode")
    durations = [episode.config.duration_s for episode in episodes]
    if not np.allclose(durations, durations[0]):
        raise ValueError("Paired videos require matching episode durations")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    renderers, models, data_objects = [], [], []
    previews = []
    try:
        for episode in episodes:
            model = mujoco.MjModel.from_xml_string(episode.xml)
            models.append(model)
            data_objects.append(mujoco.MjData(model))
            renderers.append(mujoco.Renderer(model, height=height - 244, width=width))
        camera = mujoco.MjvCamera()
        camera.azimuth, camera.elevation = 125, -10
        camera.distance = 3.1
        camera.lookat[:] = [0.05, 0, 0.85]
        frame_count = round(durations[0] * fps)
        with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=1) as writer:
            for frame_i in range(frame_count):
                time_s = frame_i / fps
                panels = []
                for episode, model, data, renderer, title in zip(
                    episodes, models, data_objects, renderers, titles, strict=True
                ):
                    a = episode.arrays
                    index = min(np.searchsorted(a["time_s"], time_s), len(a["time_s"]) - 1)
                    data.qpos[:], data.qvel[:] = a["qpos"][index], a["qvel"][index]
                    data.ctrl[:] = np.r_[a["human_applied_nm"][index], a["exo_applied_nm"][index]]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    panels.append(np.asarray(_overlay(renderer.render(), episode, index, title)))
                combined = np.concatenate(panels, axis=1)
                writer.append_data(combined)
                if frame_i == 0:
                    Image.fromarray(combined).save(output.with_name(output.stem + "_poster.png"))
                if frame_i % max(1, round(fps / 8)) == 0:
                    preview = Image.fromarray(combined)
                    preview.thumbnail((960, 480))
                    previews.append(preview.copy())
        previews[0].save(
            output.with_suffix(".gif"),
            save_all=True,
            append_images=previews[1:],
            duration=1000 * max(1, round(fps / 8)) / fps,
            loop=0,
        )
    finally:
        for renderer in renderers:
            renderer.close()
    return output


def plot_comparison(episodes: list[Episode], output: str | Path) -> None:
    """Scientific time series from the same recorded states as the video."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for episode in episodes:
        a = episode.arrays
        color = COLORS[episode.config.exo]
        label = episode.config.exo.upper()
        axes[0].plot(a["time_s"], np.degrees(a["lean_rad"]), label=label, color=color)
        human = (
            a["human_applied_nm"][:, [2, 5]].sum(axis=1)
            if episode.config.model == "human"
            else a["human_torque_nm"]
        )
        axes[1].plot(a["time_s"], human, label=label, color=color)
        axes[2].plot(a["time_s"], a["exo_torque_nm"], label=label, color=color)
    for axis, label in zip(axes, ["Lean (deg)", "Human ankle torque (Nm)", "Device ankle torque (Nm)"]):
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.set_facecolor("#172431")
        axis.legend()
    axes[-1].set_xlabel("Simulation time (s)")
    figure.suptitle("Matched initial state and disturbance | " + episodes[0].config.model)
    figure.savefig(output, dpi=160)
    plt.close(figure)
