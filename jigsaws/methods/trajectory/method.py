"""3D tool-tip trajectory plot: expert (highest GRS) vs novice (lowest GRS)."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from ...config import Hand
from ...deps import require
from ...errors import DataError
from ...kinematics import Instrument
from ..base import Command, Context, Method

LOGGER = logging.getLogger(__name__)

_MTM_TIPS = {
    Hand.LEFT: Instrument.MTF_L,
    Hand.RIGHT: Instrument.MTF_R,
}
_FIXED_TIPS = {
    "psm1": Instrument.PSM_1,
    "psm2": Instrument.PSM_2,
}


def _plot(
    expert_positions,
    novice_positions,
    *,
    expert_name: str,
    novice_name: str,
    title: str,
    output: Path,
) -> Path:
    matplotlib = require("matplotlib", extra="plot")
    matplotlib.use("Agg")
    pyplot = require("matplotlib.pyplot", extra="plot")
    pyplot.rc("font", family="Times New Roman")

    figure = pyplot.figure()
    axis = figure.add_subplot(projection="3d")
    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z")
    axis.plot(
        expert_positions[:, 0],
        expert_positions[:, 1],
        expert_positions[:, 2],
        c="b",
        label=f"expert ({expert_name})",
    )
    axis.plot(
        novice_positions[:, 0],
        novice_positions[:, 1],
        novice_positions[:, 2],
        c="r",
        label=f"novice ({novice_name})",
    )
    axis.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=300, bbox_inches="tight")
    pyplot.close(figure)
    return output


class TrajectoryMethod(Method):
    name = "trajectory"
    summary = "3D expert/novice tool-tip trajectory visualisation"

    def commands(self) -> tuple[Command, ...]:
        return (Command("plot", "Plot expert and novice 3D trajectories", self._plot,
                        self._configure),)

    @staticmethod
    def _configure(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--hand", choices=[hand.value for hand in Hand], default=None,
                            help="Left or right MTM tip (used with --tip mtm)")
        parser.add_argument("--tip", choices=("mtm", "psm1", "psm2"), default="mtm",
                            help="Instrument tip to plot: MTM hand, PSM1 or PSM2")
        parser.add_argument("--expert", default=None, help="Expert trial name override")
        parser.add_argument("--novice", default=None, help="Novice trial name override")
        parser.add_argument("--output", type=Path, default=None)

    def _plot(self, ctx: Context, args: argparse.Namespace) -> None:
        dataset = ctx.dataset
        hand = Hand(args.hand) if args.hand else ctx.settings.trajectory_hand
        expert_name = args.expert or dataset.ranked_trials(descending=True)[0]
        novice_name = args.novice or dataset.ranked_trials(descending=False)[0]
        if expert_name == novice_name:
            raise DataError("Expert and novice trial resolve to the same trial; pass --expert/--novice")

        instrument = _MTM_TIPS[hand] if args.tip == "mtm" else _FIXED_TIPS[args.tip]
        columns = instrument.position_columns
        expert_matrix = dataset.load_kinematics(expert_name)
        novice_matrix = dataset.load_kinematics(novice_name)

        if args.tip == "mtm":
            name = f"3d_trajectory_{hand.value}_hand.png"
            title = f"3D_{ctx.task.value}_{hand.value}_hand"
        else:
            name = f"3d_trajectory_{args.tip}.png"
            title = f"3D_{ctx.task.value}_{args.tip.upper()}"
        output = args.output or (
            ctx.settings.artifacts_dir / "figures" / ctx.task.value / name
        )
        path = _plot(
            expert_matrix[:, columns],
            novice_matrix[:, columns],
            expert_name=expert_name,
            novice_name=novice_name,
            title=title,
            output=output,
        )

        report = {
            "task": ctx.task.value,
            "tip": args.tip,
            "hand": hand.value if args.tip == "mtm" else None,
            "instrument": instrument.value,
            "expert_trial": expert_name,
            "expert_score": dataset[expert_name].score,
            "novice_trial": novice_name,
            "novice_score": dataset[novice_name].score,
            "figure": str(path),
        }
        path.with_suffix(".json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        LOGGER.info("Wrote trajectory plot: %s", path)


METHOD = TrajectoryMethod()
