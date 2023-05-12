from typing import List, Dict
from SolverParameter import SolverParameter

import random


def load_configurations(filename: str) -> List[SolverParameter]:
    configurations = []
    content = []
    with open(filename, "r") as file:
        for line in file:
            if line[0] == "#":
                continue
            row = line.strip().split(',')
            content.append(row)
    for row in content:
        parameter_name = row[0]
        default_value = row[1]
        weight = int(row[2])
        values = row[3:]
        configurations.append(SolverParameter(parameter_name, default_value, values, weight))
    return configurations


def write_configurations(filename: str, configurations: List[SolverParameter]):
    with open(filename, "w") as file:
        for parameter in configurations:
            parameter_name, default_value, weight, values = parameter.parameter_name, parameter.default_value, parameter.weight, parameter.values
            file.write(f"{parameter_name},{default_value},{weight},{','.join(values)}\n")


def compare_configurations(config1: Dict[str, str], config2: Dict[str, str]) -> Dict[str, str]:
    assert (len(config1) == len(config2))
    diff = dict()
    for c in config1:
        v1, v2 = config1[c], config2[c]
        if v1 != v2:
            diff[c] = v2
    return diff

def default_configuration(configurations : List[SolverParameter]) -> Dict[str,str]:
    config = dict()
    for p in configurations:
        config[p.parameter_name] = p.default_value
    return config


def sample_configurations(configurations : List[SolverParameter], n_samples : int) -> List[Dict[str,str]]:
    samples = []
    config = dict()
    for p in configurations:
        config[p.parameter_name] = p.default_value
    samples.append(config)

    for _ in range(n_samples-1):
        config = dict()
        for p in configurations:
            config[p.parameter_name] = random.choice(p.values)
        samples.append(config)
    return samples


def config_to_string(config: Dict[str, str]) -> str:
    s = ""
    for pair in config.items():
        s += pair[0] + "+" + pair[1] + ","
    return s
