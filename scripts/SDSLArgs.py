import argparse


def sdcl_arg_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("filename", help="the input file", type=str)

    parser.add_argument("--mode", type=str, default="sdcl", choices=["bmc", "sdcl"],
                        help="bmc or sdcl or single-shot")
    parser.add_argument("--solver", type=str, default="kissat", choices=["kissat", "cadical"], help="cadical or kissat")
    parser.add_argument("--config-file", type=str, default="./configs/kissat/kissat_exp.csv", help="config file")

    # BMC parameters
    parser.add_argument("--start", help="BMC start", type=int, default=0)
    parser.add_argument("-k", "--k", help="BMC bounds", type=int, default=1000)
    parser.add_argument("-s", "--step", help="BMC step", type=int, default=1)

    # MCMC Sampling parameters
    parser.add_argument("--criterion", help="cost function to evaluate a strategy", type=str, default="conflicts",
                        choices=["time", "conflicts"])

    parser.add_argument("--samples", help="number of samples to draw per sampling epoch", type=int, default=100)
    parser.add_argument("--sampling-budget", help="time budget of sampling", type=int, default=1440)

    # Learning parameters
    parser.add_argument("--once", action="store_true", help="Issue one sampling run")
    parser.add_argument("--training-method", type=str, default="rf", choices=["rf", "dt", "tuning", "nn", "svm"],
                        help="learning procedure")
    parser.add_argument("--n-trees", type=int, default=50, help="number of trees in random forest")
    parser.add_argument("--tree-depth", type=int, default=5, help="number of trees in random forest")
    parser.add_argument("--mcmc-beta", type=float, default=20, help="MCMC beta")
    
    # Others
    parser.add_argument("-w", "--working-dir", help="directory to store the files", type=str, default="./data/")
    parser.add_argument("-v", "--verbosity", help="verbosity", type=int, default=1)
    parser.add_argument("--seed", help="random seed", type=int, default=0)

    # Debug
    parser.add_argument("--adaptive", action="store_true", help="Adaptive proposal")
    parser.add_argument("--greedy-init", action="store_true", help="greedy init")
    parser.add_argument("--no-flex-tree", action="store_true", help="Do not dynamically change tree depth")

    parser.add_argument("--debug", action="store_true", help="DEBUG: debugging mode")
    parser.add_argument("--shadow", action="store_true", help="DEBUG: run the default config for each problem")
    parser.add_argument("--store-training-data", action="store_true", help="DEBUG: store the training data")
    parser.add_argument("--store-training-data-only", action="store_true", help="DEBUG: stop right after last training round "
                                                                                "is completed")
    parser.add_argument("--training-data", type=str, default=None, help="DEBUG: the training data")


    return parser.parse_args()
