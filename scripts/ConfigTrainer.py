from typing import Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import tree


class ConfigTrainer:
    def __init__(self, data, pv_to_index: Dict[str, int], method: str, n_trees: int, tree_depth: int):
        self.feature_names = list(data[0][0].keys())
        self.pv_to_index = pv_to_index
        self.X = []
        self.y = []
        for config, val in data:
            x = [pv_to_index[key + "+" + config[key]] for key in self.feature_names]
            self.X.append(x)
            self.y.append(val)
        self.method = method
        if method == "rf":
            self.regressor = RandomForestRegressor(n_estimators=n_trees, max_depth=tree_depth, random_state=0)
        elif method == "dt":
            self.regressor = DecisionTreeRegressor(max_depth=5, max_features="sqrt", random_state=0)
        elif method == "nn":
            self.regressor = MLPRegressor(hidden_layer_sizes=(100, 100), random_state=0, max_iter=2000)
        elif method == "svm":
            self.regressor = SVR()

    def train(self, test=False):
        if test:
            x_train, x_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2)
            self.regressor.fit(x_train, y_train)
            print("Train score:", self.regressor.score(x_train, y_train))
            print("Test score:", self.regressor.score(x_test, y_test))
            return 0
        else:
            self.regressor.fit(self.X, self.y)
            score = round(self.regressor.score(self.X, self.y), 2)
            print("  Train score:", score)
            return score

    def evaluate(self, step: int, config: Dict[str, str]) -> float:
        x = [[step if key == "cnf" else self.pv_to_index[key + "+" + config[key]] for key in self.feature_names]]
        return self.regressor.predict(x)[0]

    def visualize(self, fig_name):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(50,40))
        _ = tree.plot_tree(self.regressor,
                           feature_names=self.feature_names,
                           class_names=["Good", "Bad"],
                           filled=True)
        fig.savefig(fig_name)

    def visualize_trees(self, fig_prefix):
        import matplotlib.pyplot as plt
        for i, estimator in enumerate(self.regressor.estimators_):
            fig = plt.figure(figsize=(50, 40))
            _ = tree.plot_tree(estimator,
                               feature_names=self.feature_names)
            #plt.show()
            fig.savefig(fig_prefix + f"{i}.png")
