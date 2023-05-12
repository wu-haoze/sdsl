#!/usr/bin/env python3

import random

import SDSLArgs
import Utils
from SDSLBMC import BMC
from ConfigTrainer import ConfigTrainer
from os.path import isdir
import pickle
import numpy as np

wd ="/barrett/scratch/haozewu/learning-schedule/"
if not isdir(wd):
    wd =""

# Binaries
binaries = dict()
binaries["pono"]=wd + "./bins/pono"
binaries["boolector"]=wd + "./bins/boolector"
binaries["cadical"]=wd + "./bins/cadical"
binaries["kissat"]=wd + "./bins/kissat"

FALSE = 10
TRUE = 20
SAT = 10
UNSAT = 20

args = SDSLArgs.sdcl_arg_parser()
print(args)

random.seed(args.seed)
np.random.seed(args.seed)

Utils.initialize_working_directory(args)


if args.mode in ["bmc", "sdcl"]:
    solver = BMC(args, binaries)
    solver.bmc_with_sdcl()
if args.mode in ["train"]:
    with open(args.training_data, 'rb') as handle:
        pv_to_index, data = pickle.load(handle)
    trainer = ConfigTrainer(data, pv_to_index, args.training_method)
    trainer.train()
