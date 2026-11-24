"""The single command-line entry point.

Global options (dataset paths, task, verbosity) are resolved once into a
:class:`Settings` and every method command receives the same :class:`Context`.
Legacy command names from the pre-refactor scripts are translated to the new
commands so existing shell scripts keep working.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from .config import TASKS, Hand, Settings, Task
from .errors import JigsawsError
from .methods import Command, Context, discover_methods

LOGGER = logging.getLogger(__name__)

_LEGACY_COMMANDS: dict[tuple[str, str], tuple[str, str]] = {
    ("helper", "generate_metadata"): ("dataset", "info"),
    ("helper", "generate_gesture_clips"): ("dataset", "clips"),
    ("hmm", "train_observations"): ("hmm", "clusters"),
    ("hmm", "train_model"): ("hmm", "train"),
    ("hmm", "test"): ("hmm", "evaluate"),
    ("stip", "cluster"): ("stipbof", "cluster"),
    ("stip", "classify"): ("stipbof", "classify"),
    ("kinematic", "feature_classify"): ("kinematic", "svc"),
    ("kinematic", "box_plot"): ("kinematic", "boxplot"),
    ("kinematic", "box_classify"): ("kinematic", "svc-csv"),
    ("kinematic", "trajectory_plot"): ("trajectory", "plot"),
}

_OPTIONS_WITH_VALUE = {
    "--task",
    "--data-root",
    "--experimental-setup",
    "--artifacts-dir",
    "--stip-features",
    "--kinematic-features",
    "--trajectory-hand",
}


def translate_legacy_args(argv: Sequence[str]) -> list[str]:
    """Rewrite ``hmm test`` style commands to their modern equivalents."""
    tokens = list(argv)
    positions: list[int] = []
    skip_next = False
    for index, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if token.startswith("-"):
            if "=" not in token and token in _OPTIONS_WITH_VALUE:
                skip_next = True
            continue
        positions.append(index)
        if len(positions) == 2:
            break
    if len(positions) != 2:
        return tokens
    key = (tokens[positions[0]], tokens[positions[1]])
    replacement = _LEGACY_COMMANDS.get(key)
    if replacement:
        LOGGER.debug("Translating legacy command %s -> %s", key, replacement)
        tokens[positions[0]], tokens[positions[1]] = replacement
    return tokens


def _add_task_option(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--task",
        choices=TASKS,
        default=argparse.SUPPRESS,
        help="JIGSAWS task (may also be given before the method name)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="JIGSAWS surgical-skill assessment: unified, method-based CLI.",
    )
    parser.add_argument("--task", choices=TASKS, default=None, help="JIGSAWS task")
    parser.add_argument("--data-root", default=None, help="Dataset root directory")
    parser.add_argument(
        "--experimental-setup",
        default=None,
        help="Directory containing Experimental_setup (auto-detected by default)",
    )
    parser.add_argument("--artifacts-dir", default=None, help="Where methods write outputs")
    parser.add_argument("--stip-features", default=None, help="STIP raw feature root")
    parser.add_argument(
        "--kinematic-features", default=None, help="Directory holding <task>_feature.npy"
    )
    parser.add_argument(
        "--trajectory-hand",
        choices=[hand.value for hand in Hand],
        default=None,
        help="Hand rendered by the 3D trajectory plot",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    verbosity.add_argument("-q", "--quiet", action="store_true", help="Warnings only")

    subparsers = parser.add_subparsers(dest="method", metavar="method", required=True)
    subparsers.add_parser(
        "methods",
        help="List registered methods and their commands",
        description="List registered methods and their commands",
    )

    for method in discover_methods().values():
        family = subparsers.add_parser(
            method.name,
            help=method.summary,
            description=method.summary,
            aliases=list(method.aliases),
        )
        family.set_defaults(_method=method)
        commands = family.add_subparsers(
            dest="command", metavar="command", required=True
        )
        for command in method.commands():
            child = commands.add_parser(
                command.name,
                help=command.help,
                description=command.help,
                aliases=list(command.aliases),
            )
            _add_task_option(child)
            if command.configure is not None:
                command.configure(child)
            child.set_defaults(_command=command)
    return parser


def _configure_logging(verbose: bool, quiet: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING if quiet else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
        force=True,
    )


def _list_methods() -> None:
    for method in discover_methods().values():
        print(f"{method.name}: {method.summary}")
        for command in method.commands():
            aliases = f" (aliases: {', '.join(command.aliases)})" if command.aliases else ""
            print(f"  {command.name:<16} {command.help}{aliases}")
        print()


def main(argv: Sequence[str] | None = None) -> int:
    arguments = translate_legacy_args(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(arguments)
    _configure_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))

    if args.method == "methods":
        _list_methods()
        return 0

    try:
        task = Task.parse(getattr(args, "task", None))
        settings = Settings.resolve(
            data_root=args.data_root,
            experimental_setup_root=args.experimental_setup,
            artifacts_dir=args.artifacts_dir,
            stip_features_dir=args.stip_features,
            kinematic_features_dir=args.kinematic_features,
            trajectory_hand=args.trajectory_hand,
        )
        context = Context(settings=settings, task=task)
        command: Command = args._command
        LOGGER.debug("Running %s/%s for %s", args._method.name, command.name, task.value)
        command.run(context, args)
    except JigsawsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
