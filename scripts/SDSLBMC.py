import argparse
import math
import os
import random
import shutil
import time
from os.path import join, basename
from subprocess import run, PIPE
from typing import List, Dict
from copy import copy
import pickle
import numpy as np

import GlobalConfigurations as GC
from ConfigsHandler import load_configurations, compare_configurations, config_to_string
import Proposer
from ConfigTrainer import ConfigTrainer

UNKNOWN = 0
UNSAT = 20
SAT = 10


class BMC:
    def __init__(self, args: argparse.ArgumentParser, binaries: Dict[str, str]):
        self.EXP_DIR = join(args.working_dir, basename(args.filename))
        self.DATA_DIR = join(self.EXP_DIR, "train_instances")
        self.CONFIG_DIR = join(self.EXP_DIR, "configs")

        self.args = args

        # Solver and configurations
        self.solver = binaries[args.solver]
        if args.solver == "kissat":
            self.time_keyword = "process-time:"
            self.solver += f" --time="
        elif args.solver == "cadical":
            self.time_keyword = "c total process time since initialization:"
            self.solver += f" -t "
        self.solver_configurations, self.name_to_parameter = self.load_configurations(args.config_file)
        shutil.copyfile(args.config_file, join(self.CONFIG_DIR, basename(args.config_file)))

        self.initial_config: Dict[str, str] = dict()
        for o in self.solver_configurations:
            self.initial_config[o.parameter_name] = o.default_value
        self.initial_config_str = config_to_string(self.initial_config)

        # BMC
        self.filename = args.filename
        self.dumper = binaries["pono"]
        self.converter = binaries["boolector"]

        # Sampling
        self.training_instances: Dict[int, str] = dict()  # from step to cnf
        self.trained_instances: Dict[int, str] = dict()  # from step to cnf

        self.proposer = Proposer.Proposer(self.solver_configurations, self.name_to_parameter, self.args.adaptive)

        self.current_config = copy(self.initial_config)
        self.best_config = copy(self.initial_config)
        self.best_sampled_config = copy(self.initial_config)

        self.data: Dict[str, Dict[str, List[float]]] = dict()  # filename to config str to cost
        self.time_data: Dict[str, Dict[str, List[float]]] = dict()  # filename to config str to cost

        self.training_data = []
        self.pv_to_index: Dict[str, int] = dict()
        for o in self.solver_configurations:
            for i, v in enumerate(o.values):
                self.pv_to_index[o.parameter_name + "+" + v] = i

        if args.training_data is not None:
            with open(args.training_data, 'rb') as handle:
                _, self.training_data_pickled = pickle.load(handle)
        else:
            self.training_data_pickled = None
        self.training_data_store_path = join(self.EXP_DIR, "data.pickle") if args.store_training_data else None

        # Training
        self.not_sampling_anymore = self.args.mode == "bmc"
        self.not_training_anymore = False

        # Statistics
        self.start_time = time.perf_counter()
        self.sampled_iterations = 0
        self.last_sampled_step = -1
        
        self.sdcl_time = 0
        self.bmc_time = 0
        self.train_time = 0
        if self.args.shadow:
            with open(join(self.EXP_DIR, "bmc_vs_sdcl.csv"), 'w') as out_file:
                out_file.write(f"bmc,sdcl,train\n")

        if self.training_data_pickled is not None:
            max_step = -1
            for i, v in enumerate(self.training_data_pickled):
                cnf = v[0]["cnf"]
                step = int(basename(cnf).split("_")[0].split("step")[1])
                features = copy(v[0])
                features["cnf"] = join(self.DATA_DIR, basename(cnf))
                new_v = features, v[1]
                self.training_data_pickled[i] = new_v

                if step > max_step:
                    max_step = step
                    self.training_instances[max_step] = join(self.DATA_DIR, basename(cnf))

            self.log(0, "Loading from pre-collected data...")
            for step, sampled_cnf in self.training_instances.items():
                self.pv_to_index["cnf+" + sampled_cnf] = step
                for config_dict, score in self.training_data_pickled:
                    if config_dict["cnf"] == sampled_cnf:
                        self.training_data.append((config_dict, score))

            self.update_current_config(max_step + 1, False)
            self.not_sampling_anymore = True
            self.not_training_anymore = True
            self.args.start = max(self.args.start, max_step + self.args.step)

    def load_configurations(self, config_file):
        all_solver_configurations = load_configurations(config_file)
        solver_configurations = []
        name_to_parameter = dict()
        num_parameter_values = 0
        for p in all_solver_configurations:
            if len(p.values) > 1:
                solver_configurations.append(p)
                name_to_parameter[p.parameter_name] = p
                num_parameter_values += len(p.values)
        if self.args.debug:
            for so in solver_configurations:
                self.debug_msg(so.dump())
        self.log(0, f"{len(solver_configurations)} tunable parameters, {num_parameter_values} parameter values")
        return solver_configurations, name_to_parameter

    def time_passed(self):
        return time.perf_counter() - self.start_time

    def bmc_with_sdcl(self):
        i = self.args.start
        while i < self.args.k + 1:
            self.log(0, f"\nBMC checking at bound: {i}")
            query_name = self.dump_bmc_query(max(i - self.args.step, -1), i)
            if query_name == UNSAT:
                self.log(0, f"  BMC check at bound {i} unsatisfiable (trivial)")
            else:
                result, score, time_in_seconds, _ = self.solve_cnf(query_name, self.current_config, 0)
                self.log(0, f"    BMC check runtime for this step: {time_in_seconds}")
                self.log(0, f"    BMC check runtime at bound {i}: {round(self.time_passed(), 2)}")
                if self.args.shadow:
                    result_s, _, time_in_seconds_s, _ = self.solve_cnf(query_name, self.initial_config, 0)
                    self.sdcl_time += time_in_seconds
                    self.bmc_time += time_in_seconds_s
                    self.log(0, f"Shadowing: current config: {round(time_in_seconds, 2)} "
                                f"({round(self.sdcl_time, 2)}), "
                                f"initial config: {round(time_in_seconds_s, 2)} "
                                f"({round(self.bmc_time, 2)}), "
                                f"better? [{'yes' if time_in_seconds < time_in_seconds_s else 'no'}]")
                    with open(join(self.EXP_DIR, "bmc_vs_sdcl.csv"), 'a') as out_file:
                        out_file.write(f"{self.bmc_time},{self.sdcl_time},{round(self.train_time,2)}\n")
                if result == SAT:
                    self.log(0, f"  BMC check at bound {i} satisfiable")
                    return
                elif result != UNSAT:
                    self.log(0, f"  BMC check at bound {i} error")
                    return
                else:
                    self.log(0, f"  BMC check at bound {i} unsatisfiable")
                    if self.args.mode == "sdcl":
                        if score > GC.MIN_CONFLICTS_TO_SAMPLE or (self.args.criterion == "time" and score > 0.1):
                            self.log(0, f"  Adding candidate sample step {i}")
                            self.training_instances[i] = query_name

                            if self.should_sample(i):
                                self.sample_training_data(i, self.args.samples)
                            if self.sampled_iterations > 0 and not self.not_training_anymore:
                                self.update_current_config(i, not self.not_sampling_anymore)
                                if (not self.not_sampling_anymore) and (not self.should_sample(i)):
                                    # corner case
                                    self.log(1, "Hitting corner case!")
                                    self.not_sampling_anymore = True
                                    self.update_current_config(i, not self.not_sampling_anymore)
                                if self.args.once or self.not_sampling_anymore:
                                    self.not_training_anymore = True

            i += self.args.step
        return

    def should_sample(self, step) -> bool:
        if self.not_sampling_anymore:
            return False
        if self.args.once:
            total_time = 0
            for i in self.training_instances:
                total_time += self.time_data[self.training_instances[i]][self.initial_config_str]
            if total_time * self.args.samples > self.args.sampling_budget:
                self.log(1, "Hit critical mass...")
                self.not_sampling_anymore = True
                if len(self.training_instances) > 1:
                    del self.training_instances[step]
                    self.log(1, f"Sampling on {len(self.training_instances)} instances...")
                    return True
            return False
        else:
            if self.train_time < self.args.sampling_budget:
                self.log(1, f"Sampling on the current step...")
                return True
            else:
                self.log(1, f"Not sampling anymore...")
                return False

    def dump_bmc_query(self, proven: int, bound: int):
        self.log(1, f"Dumping cnf, proven {proven}, to prove {bound}")
        query_name = join(self.DATA_DIR, f"step{bound}_reached{proven}.smt2")
        cmd = f"{self.dumper} --engine bmc --reached-k {proven} -k {bound} --dump " + \
              f"--working-dir {self.DATA_DIR} --bmc-neg-bad-step-all {self.filename}"
        run(cmd.split(), stdout=PIPE, stderr=PIPE)

        with open(query_name[:-4] + "cnf", 'w') as f:
            cmd = f"{self.converter} -dd {query_name}"
            run(cmd.split(), stdout=f)
        os.remove(query_name)
        query_name = query_name[:-4] + "cnf"
        # Check if the problem is trivial
        if open(query_name).readlines()[0] == "unsat\n":
            os.remove(query_name)
            return UNSAT
        else:
            self.data[query_name] = dict()
            self.time_data[query_name] = dict()
            return query_name

    def solve_cnfs(self, cnf_names: List[str], config: Dict[str, str], base_config: Dict[str, str]):
        return_codes = []
        scores = []
        time_in_seconds = []
        cached = []
        base_config_str = config_to_string(base_config)
        for cnf_name in cnf_names:
            limit = math.ceil(self.data[cnf_name][base_config_str] * GC.LOCAL_SEARCH_TIMEOUT_FACTOR)
            return_code_i, score_i, time_in_seconds_i, cached_i = self.solve_cnf(cnf_name, config, limit)
            return_codes.append(return_code_i)
            scores.append(score_i)
            time_in_seconds.append(time_in_seconds_i)
            cached.append(cached_i)
        return return_codes, scores, time_in_seconds, cached

    def solve_cnf(self, cnf_name: str, config: Dict[str, str], limit: int = 0) -> (
            int, float, bool):
        self.log(3, f"Solving cnf {basename(cnf_name)}")
        if self.args.debug:
            self.debug_msg(f"  with limit {round(limit)}")

        config_str = config_to_string(config)

        return_code = UNKNOWN
        if cnf_name in self.data and config_str in self.data[cnf_name]:
            self.log(3, f"\tsolver run cached")
            cached = True
            score = self.data[cnf_name][config_str]
            time_in_seconds = self.time_data[cnf_name][config_str]
        else:
            self.log(3, f"\tsolver run not cached")
            cached = False
            if limit > 0 and self.args.criterion == "time":
                timeout = math.ceil(limit)
            else:
                timeout = 1000000

            cmd = f"{self.solver}{timeout} {cnf_name}"
            for c in config:
                cmd += f" --{c}={config[c]}"

            if limit > 0 and self.args.criterion == "conflicts":
                cmd += f" --conflicts={limit}"
            score = None
            time_in_seconds = None
            output = run(cmd.split(), stdout=PIPE, stderr=PIPE)
            return_code = output.returncode
            o = output.stdout.decode().split("\n")
            for line in o:
                if self.time_keyword in line:
                    time_in_seconds = float(line.split()[-2])
                    if self.args.criterion == "time":
                        score = time_in_seconds
                elif self.args.criterion == "conflicts" and "c conflicts:" in line:
                    score = int(line.split()[2])
            assert (score is not None)
            assert (time_in_seconds is not None)
            if return_code not in [SAT, UNSAT]:
                time_in_seconds *= 5
                score *= 5
                self.log(1, f"timeout... using PAR5, {round(time_in_seconds, 2)}, {score}")

            self.data[cnf_name][config_str] = score
            self.time_data[cnf_name][config_str] = time_in_seconds

        return return_code, score, time_in_seconds, cached

    def propose(self, config: Dict[str, str]) -> (str, str, str):
        parameter_name, old_value, new_value = self.proposer.propose(config)
        return parameter_name, old_value, new_value

    def accept(self, new_score: float, last_accepted_score: float, base_score: float = 1) -> bool:
        self.log(3, f"  Current cost: {round(new_score, 2)}, last accepted cost: {round(last_accepted_score, 2)}")
        if new_score <= last_accepted_score:
            self.log(3, "    Probability to accept: 1")
            accepted = True
        else:
            prob = math.exp((-new_score + last_accepted_score) / base_score * self.args.mcmc_beta)
            self.log(3, "    Probability to accept: {}".format(round(prob, 2)))
            accepted = random.random() < prob
        self.log(2, f"    Accepted ? {'yes' if accepted else 'no'}")
        return accepted

    def sample_training_data(self, step: int, samples: int):
        tic = time.perf_counter()
        training_instances = list(self.training_instances.values())

        best_score = sum(self.solve_cnfs(training_instances, self.best_config, self.initial_config)[1])
        initial_score = sum(self.solve_cnfs(training_instances, self.initial_config, self.initial_config)[1])
        self.log(1, f"Score of previous best config: {best_score}, score of initial config: {initial_score}")
        if best_score < initial_score:
            best_config = copy(self.best_config)
            prev_best_better = True
        else:
            best_score = initial_score
            best_config = copy(self.initial_config)
            prev_best_better = config_to_string(self.initial_config) == config_to_string(self.best_config)

        total_time = 0
        for training_instance in training_instances:
            total_time += self.time_data[training_instance][config_to_string(best_config)]
        estimated_time = self.train_time + total_time * samples
        if estimated_time > self.args.sampling_budget:
            if self.train_time + total_time * samples / 2 <= self.args.sampling_budget:
                self.log(0, f"Not enough time to train {samples} times, but has enough time to train {samples/2} times")
                samples = int(samples / 2)
            else:
                self.log(0, f"Estimate to exceed training budget {estimated_time}, not training anymore")
                self.not_sampling_anymore = True
                sampling_time = round(time.perf_counter() - tic, 2)
                self.log(0, f"\n  Sampling runtime at bound {step}: {sampling_time}")
                self.train_time += sampling_time
                return

        if self.args.greedy_init:
            score, config = best_score, copy(self.best_config)
        else:
            score, config = initial_score, copy(self.initial_config)
        diff = compare_configurations(self.initial_config, config)
        self.log(1, "MCMC initialization: " + str(diff))

        self.log(1, f"\nSampling {samples} times at bound: {step}")
        for it in range(samples):
            if it % 10 == 0 and self.args.adaptive:
                score, config = initial_score, copy(self.initial_config)
            self.log(1, f"  ---- sample {it}  current score: {round(score, 2)} "
                        f"best score: {round(best_score, 2)} ----")
            parameter_name, old_value, new_value = self.propose(config)
            assert (parameter_name is not None)
            new_config = copy(config)
            new_config[parameter_name] = new_value
            _, new_scores, _, cached = self.solve_cnfs(training_instances, new_config, best_config)
            new_score = sum(new_scores)
            if True not in cached:
                self.proposer.record_proposal_effect(parameter_name, old_value, new_value, score,
                                                     new_score)
            self.log(2, f"    Proposal: {parameter_name} {old_value} => {new_value}, "
                        f"new score: {round(new_score, 2)}")
            if self.accept(new_score, score, initial_score):
                score, config = new_score, copy(new_config)
                if score < best_score:
                    best_score, best_config = new_score, copy(new_config)

        toc = time.perf_counter()
        sampling_time = round(toc - tic, 2)
        tic = toc
        self.log(0, f"\n  Sampling runtime at bound {step}: {sampling_time}")
        self.train_time += sampling_time
                    
        self.log(1, f"Sampling done, training score: {round(initial_score, 2)} -> "
                    f"{round(best_score, 2)}")
        diff = compare_configurations(self.initial_config, best_config)
        self.log(1, "Diff: " + str(diff))
        self.best_sampled_config = best_config

        for k, v in self.training_instances.items():
            self.trained_instances[k] = v
            if k > self.last_sampled_step:
                self.last_sampled_step = k
        self.training_instances = dict()

        self.sampled_iterations += 1

    def prepare_training_data(self):
        if self.training_data_pickled is None:
            self.log(1, "Preparing training data...")
            self.training_data.clear()
            for step, sampled_cnf in self.trained_instances.items():
                self.pv_to_index["cnf+" + sampled_cnf] = step
                config_to_time = self.data[sampled_cnf]
                if self.initial_config_str not in config_to_time:
                    continue
                base_time = config_to_time[self.initial_config_str]
                for config, new_time in config_to_time.items():
                    config_dict = dict()
                    config_dict["cnf"] = sampled_cnf
                    for pv in config[:-1].split(","):
                        p, v = pv.split("+")
                        config_dict[p] = v
                    score = new_time / base_time
                    self.training_data.append((config_dict, score))
            self.log(1, f"{len(self.training_data)} training points added...")

    def store_training_data(self):
        if self.training_data_store_path is not None:
            if os.path.isfile(self.training_data_store_path):
                os.remove(self.training_data_store_path)
            with open(self.training_data_store_path, 'wb') as handle:
                pickle.dump((self.pv_to_index, self.training_data), handle, protocol=pickle.HIGHEST_PROTOCOL)
            self.log(1, f"\nStored {len(self.training_data)} training points\n")
        if self.args.store_training_data_only:
            self.log(0, "Not training anymore, quitting")
            exit(0)

    def update_current_config(self, step: int, mock: bool):
        self.log(1, f"\nTraining at bound: {step}, method: {self.args.training_method}\n")
        self.prepare_training_data()
        self.store_training_data()
        tic = time.perf_counter()
        if self.args.training_method == "tuning":
            self.best_config = copy(self.best_sampled_config)
            if not mock:
                self.current_config = self.best_sampled_config
        else:
            n_trees = self.args.n_trees
            tree_depth = self.args.tree_depth
            self.log(1, f"Training with depth {tree_depth}, ntrees{n_trees}")
            trainer = ConfigTrainer(self.training_data, self.pv_to_index, self.args.training_method,
                                    n_trees, tree_depth)
            score = trainer.train()
            if (not self.args.no_flex_tree) and ((not mock) or self.args.once):
                if score < 0.9 and len(self.training_data) < 90:
                    self.log(1, "Training score too low, fall back")
                    self.current_config = copy(self.initial_config)
                    return
                if score >= 0.85: # Good enough
                    pass
                else:
                    if score <= 0.65: # Really bad, need a much bigger forest
                        depth = int(tree_depth * 5)
                        n_trees = int(n_trees * 2)
                    elif score <= 0.8: # bad, need more depth
                        depth = int(tree_depth * 2)
                    else: # Could be better # 0.8 - 0.85
                        depth = int(tree_depth * 1.6)
                    self.log(1, f"Retraining with depth {depth}, ntrees{n_trees}")
                    trainer = ConfigTrainer(self.training_data, self.pv_to_index, self.args.training_method,
                                            n_trees, depth)

                    score = trainer.train()
                self.log(1, f"Final train score: {score}") 
                    
            if self.args.debug:
                if self.args.training_method == "dt":
                    trainer.visualize(join(self.EXP_DIR, f"decision_tree_step{step}.png"))
                elif self.args.training_method == "rf":
                    trainer.visualize_trees(join(self.EXP_DIR, f"decision_tree"))
                    
            # Sample 500 configs and evaluate with trainer
            best_score, best_config, best_updates = 1, copy(self.initial_config), 0
            for sa in range(GC.NUMBER_OF_MCMC_SAMPLES_FOR_REGRESSOR):
                if sa % 10 == 0 and self.args.adaptive:
                    score, config = 1, copy(self.initial_config)
                elif sa % self.args.samples == 0:
                    score, config = 1, copy(self.initial_config)
                parameter_name, old_value, new_value = self.propose(config)
                assert (parameter_name is not None)
                new_config = copy(config)
                new_config[parameter_name] = new_value
                new_score = trainer.evaluate(step + 1, new_config)
                if self.accept(new_score, score):
                    updates = 0
                    for k in new_config:
                        if self.initial_config[k] != new_config[k]:
                            updates += 1
                    train_score, config = new_score, copy(new_config)
                    if new_score < best_score or \
                            (new_score == best_score and updates < best_updates):
                        self.log(1, f"{best_score} --> {new_score}")
                        best_score, best_config = new_score, copy(new_config)
                        best_updates = updates

            self.log(1, f"  Predicted training score: {round(best_score, 2)}")

            self.best_config = copy(best_config)
            if not mock:
                self.current_config = copy(best_config)

        diff = compare_configurations(self.initial_config, self.best_sampled_config)
        self.log(1, "Current best sampled config: " + str(diff))

        diff = compare_configurations(self.initial_config, self.best_config)
        self.log(1, "Current best config: " + str(diff))

        diff = compare_configurations(self.initial_config, self.current_config)
        self.log(0, "Current config: " + str(diff))

        train_time = round(time.perf_counter() - tic, 2)
        self.log(0, f"  Training and predicting runtime at bound {step}: {train_time}\n")
        self.train_time += train_time

    def log(self, verbosity: int, msg: str):
        if self.args.verbosity >= verbosity:
            print(msg)


    def debug_msg(self, msg: str):
        print(f"DEBUG:    {msg}")
