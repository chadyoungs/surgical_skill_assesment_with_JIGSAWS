import argparse
import os
from exception import OperationNotFoundError


def argument_parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)

    parser.add_argument(
        "--task",
        default="Suturing",
        type=str,
        choices=["Suturing", "Knot_Tying", "Needle_Passing"],
    )

    parser.add_argument(
        "--option",
        help="Determines the operation to perform",
        choices=("image_stitch", "generate_gesture_clips", "generate_metadata"),
        required=True,
    )

    return parser.parse_args()


def main() -> None:
    args = argument_parse()
    option = args.option
    os.environ["JIGSAWS_TASK"] = args.task

    from metadata_generation import MetaData
    from surgeme_generation import get_metadata, make_dirs, video_surgeme_generation
    from tools import image_stitch

    if option == "image_stitch":
        stitch_img_save_name = "image_stitch.jpg"
        image_stitch([], stitch_img_save_name)
    elif option == "generate_gesture_clips":
        res = get_metadata()
        make_dirs(res)
        video_surgeme_generation(res)
    elif option == "generate_metadata":
        MetaData().generate_metadata()
    else:
        raise OperationNotFoundError("check the option argument")


if __name__ == "__main__":
    main()
