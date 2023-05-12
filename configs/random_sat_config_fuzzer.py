import sys
import random
import time
import subprocess
import concurrent.futures
import copy
from subprocess import PIPE

options = {}

with open(sys.argv[1]) as in_file:
    for line in in_file.readlines():
        lst = line[:-1].split(",")
        options[lst[0]] = lst[3:]

print(options)

def run_command(i, solver, benchmark, options):
    cmd = f"{solver} {benchmark} --quiet"
    for o in options:
        cmd += f" --{o}={random.choice(options[o])}"
    tic = time.perf_counter()
    output = subprocess.run(cmd.split(),stdout=PIPE, stderr=PIPE)
    if output.returncode != 20:
        msg = cmd + " not ok"
        print(msg)
        return False
    else:
        print(cmd + " ok")
        return True

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    solver, benchmark = sys.argv[2], sys.argv[3]
    options_c = copy.copy(options)
    results = [executor.submit(run_command, i, solver, benchmark, options_c) for i in range(10000)]
    for f in concurrent.futures.as_completed(results):
        if not f.result():
            break
