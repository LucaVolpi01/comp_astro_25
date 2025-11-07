import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *
from daneel.detection import transit_method


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input",
        dest="input_file",
        type=str,
        required=True,
        help="Input par file to pass",
    )

    parser.add_argument(
        "-d",
        "--detect",
        dest="detect",
        required=False,
        help="Initialise detection algorithms for Exoplanets",
        action="store_true",
    )

    parser.add_argument(
        "-a",
        "--atmosphere",
        dest="atmosphere",
        required=False,
        help="Atmospheric Characterisazion from input transmission spectrum",
        action="store_true",
    )

    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="test transit",
        action='store_const',
        default=transit_method.transit,
		const=transit_method.transit,
    )

    args = parser.parse_args()

    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")

    input_pars = Parameters(args.input_file).params

    if args.detect:
        pass
    if args.atmosphere:
        pass
    if args.transit:
        transit_method.transit(input_pars)

    finish = datetime.datetime.now()
    print(f"Daneel finishes at {finish}")


if __name__ == "__main__":
    main()
