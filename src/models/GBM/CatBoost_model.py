import numpy as np
from time import time
import pprint
import pandas as pd
from skopt import BayesSearchCV
from catboost import CatBoostRegressor

class CatBoost():
    def __init__(self, args,data) -> None:
        super().__init__()

        self.model = CatBoostRegressor(
            iterations=10000,
            learning_rate=0.05,
            depth=6,
            metric_period=1000,
            random_seed=args.seed,
            task_type="GPU",
            devices='0',
            od_type='Iter',
            od_wait= 1000
            )
        self.data = data
        self.cat_features = list(range(0, self.data['X_train'].shape[1]))
        self.seed = args.seed
        self.use_best_model = args.use_best_model

    def train(self):
        self.model.fit(
            self.data['X_train'],
            self.data['y_train'],
            eval_set=(self.data['X_valid'],self.data['y_valid']),
            cat_features=self.cat_features,
            use_best_model=self.use_best_model,
            verbose=True
            )
        
    def pred(self, test):
        return self.model.predict(test)