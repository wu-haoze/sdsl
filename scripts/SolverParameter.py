from typing import List


class SolverParameter:
    def __init__(self, parameter_name: str, default_value: str, values: List[str], weight: int = 1):
        self.parameter_name = parameter_name
        self.default_value = default_value
        self.values = values
        self.weight = weight

    def dump(self):
        s = f"parameter name: {self.parameter_name}, default value: {self.default_value}, weight: {self.weight},"
        s += f" possible values: {', '.join(self.values)}"
        return s
