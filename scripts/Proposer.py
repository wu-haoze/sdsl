from typing import List, Dict
import SolverParameter
from Utils import draw_top_key
from collections import OrderedDict
import random
import GlobalConfigurations as GC


class Proposer:
    def __init__(self, solver_configurations, name_to_parameter, adaptive: bool):
        self.solver_configurations:List[SolverParameter] = solver_configurations
        self.name_to_parameter = name_to_parameter
        self.ema_alpha = GC.EMA_ALPHA
        self.tabu_size = GC.TABU_SIZE
        self.adaptive = adaptive

        self.short_term_memory: List[str] = []
        self.parameter_value_to_score = OrderedDict()  # represents how likely the option is to be drawn

        # Some book-keeping
        self.parameter_value_to_proposal_count = dict()

        for p in self.solver_configurations:
            for v in p.values:
                pv = p.parameter_name + "+" + v
                self.parameter_value_to_score[pv] = 1
                self.parameter_value_to_proposal_count[pv] = 1 if v == p.default_value else 0

    def propose(self, config: Dict[str, str]) -> (str, str, str):
        active_parameter_value: List[str] = []
        for c in config:
            if c in self.name_to_parameter:
                p = self.name_to_parameter[c]
                for v in p.values:
                    if v != config[c]:
                        active_parameter_value.append(c + "+" + v)

        if self.adaptive:
            for pv in self.short_term_memory:
                if pv in active_parameter_value and len(active_parameter_value) > 1:
                    active_parameter_value.remove(pv)
            pv = draw_top_key(self.parameter_value_to_score, active_parameter_value)
        else:
            pv = random.choice(active_parameter_value)
        parameter_name, value = pv.split("+")
        self.remember_visited(parameter_name, value)
        self.parameter_value_to_proposal_count[pv] += 1
        return parameter_name, config[parameter_name], value

    def remember_visited(self, parameter_name: str, value: str):
        val = parameter_name + "+" + value
        if val in self.short_term_memory:
            return
        self.short_term_memory.append(val)
        if len(self.short_term_memory) > self.tabu_size:
            self.short_term_memory = self.short_term_memory[1:]

    def record_proposal_effect(self, parameter_name: str, value_before: str, value_after: str,
                               before: float, after: float):
        ratio = before / after
        # if ratio is greater than 1, the value after is better, bump up score of pv_after, bump down score of pv_before
        pv_after = parameter_name + "+" + value_after
        if pv_after in self.parameter_value_to_score:
            self.parameter_value_to_score[pv_after] *= 1 - self.ema_alpha
            self.parameter_value_to_score[pv_after] += self.ema_alpha * ratio

    def remove_parameter_value(self, parameter_name, value):
        self.parameter_value_to_score.pop(parameter_name + "+" + value)
        self.parameter_value_to_proposal_count.pop(parameter_name + "+" + value)