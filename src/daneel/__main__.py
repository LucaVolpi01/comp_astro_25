import datetime
import argparse
from daneel.parameters import Parameters
from daneel.detection import *

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
        help="Atmospheric action to run",
        type=str,              
    )


    parser.add_argument(
        "-t",
        "--transit",
        dest="transit",
        required=False,
        help="test transit",
        action='store_true',
    )

    parser.add_argument(
        "-dt",
        "--detection",
        dest="dt_algorithm",
        type=str,
        required=False,
        help="Initialise detection algorithms for Exoplanets",
    #    action="store_true",
    )

    parser.add_argument(
        "-dream",
        "--dream",
        dest="dream",
        required=False,
        help="Calls the GAN to check the light curve images created from the TESS data file",
        action="store_true",
    )


    args = parser.parse_args()


    """Launch Daneel"""
    start = datetime.datetime.now()
    print(f"Daneel starts at {start}")
       

    if args.detect:
        pass
    
    if args.atmosphere == "model":
        from daneel.atmosphere import model
        print("##############  MODEL  ###################")
        model.main(args.input_file)
        
    if args.atmosphere == "retrieve":
        from daneel.atmosphere import model
        from daneel.atmosphere import retrieval
        print("##############  RETRIEVE  ###################")
        model.main(args.input_file)
        retrieval.main(args.input_file)
        
    if args.transit:
        from daneel.detection import transit_method

        print("##############  TRANSIT  ###################")
        input_pars = Parameters(args.input_file).params
        transit_method.transit(input_pars)
        
    if args.dt_algorithm == "rf":
        from daneel.detection import random_forest
        print("##############  RANDOM FOREST  ###################")
        input_pars = Parameters(args.input_file).params
        random_forest.rf_main(input_pars)
        
    if args.dt_algorithm == "cnn":
        from daneel.detection import cnn
        print("##############  CNN  ###################")
        input_pars = Parameters(args.input_file).params
        cnn.cnn_main(input_pars)
        
    if args.dream:
        from daneel.dream import GAN
        print("##############  GAN  ###################")
        input_pars = Parameters(args.input_file).params
        GAN.GAN_main(input_pars)

    finish = datetime.datetime.now()
    print(f"Total runtime: {finish - start}")


if __name__ == "__main__":
    main()
