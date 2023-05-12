from os import mkdir
from os.path import join, basename, isdir
from subprocess import run
import math
import numpy as np


def initialize_working_directory(args):
    if not isdir(args.working_dir):
        mkdir(args.working_dir)
    EXP_DIR = join(args.working_dir, basename(args.filename))
    DATA_DIR = join(EXP_DIR, "train_instances")
    CONFIG_DIR = join(EXP_DIR, "configs")
    if not isdir(EXP_DIR):
        mkdir(EXP_DIR)
    if not isdir(DATA_DIR):
        mkdir(DATA_DIR)
    elif args.mode != "ambsearch":
        run(f"rm -rf {DATA_DIR}/*", shell=True)
    if not isdir(CONFIG_DIR):
        mkdir(CONFIG_DIR)
    else:
        run(f"rm -rf {CONFIG_DIR}/*", shell=True)


def is_power_of_two(v: float) -> bool:
    v = math.log2(v)
    return math.ceil(v) == math.floor(v)


def smallest_element(lst, l):
    for i, x in enumerate(lst):
        if all(x <= y * l for j, y in enumerate(lst) if i != j):
            return i
    return -1


def min_ind_other_than(lst, index):
    min_index = -1
    min_val = 100000
    for i, e in enumerate(lst):
        if i != index and e < min_val:
            min_val = e
            min_index = i
    return min_index


def draw_top_key(prob_map, keys_subset):
    prob_sub_map = dict()
    for key in keys_subset:
        prob_sub_map[key] = prob_map[key]
    max_val = max(prob_sub_map.values())
    max_keys = [key for key, value in prob_sub_map.items() if abs(value - max_val) <= 0.001]
    selected_key = np.random.choice(max_keys)
    return selected_key
