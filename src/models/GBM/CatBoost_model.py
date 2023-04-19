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

    # Reporting util for different optimizers
    def report_perf(self, optimizer, X, y, title="model", callbacks=None):
        """
        A wrapper for measuring time and performances of different optmizers
        
        optimizer = a sklearn or a skopt optimizer
        X = the training set 
        y = our target
        title = a string label for the experiment
        """
        start = time()
        
        if callbacks is not None:
            optimizer.fit(X, y, callback=callbacks)
        else:
            optimizer.fit(X, y)
            
        d=pd.DataFrame(optimizer.cv_results_)
        best_score = optimizer.best_score_
        best_score_std = d.iloc[optimizer.best_index_].std_test_score
        best_params = optimizer.best_params_
        
        print((title + " took %.2f seconds,  candidates checked: %d, best CV score: %.3f "
            + u"\u00B1"+" %.3f") % (time() - start, 
                                    len(optimizer.cv_results_['params']),
                                    best_score,
                                    best_score_std))    
        print('Best parameters:')
        pprint.pprint(best_params)
        print()
        return best_params
    
    def hyperparameter_tuning(self):
        self.search_space= {
            'iterations' : (100000,100001),
            'learning_rate' : (0.001,0.01,'log-uniform'),
            'depth' : (6,10),
            # 'random_strength' : (1e-9,10,'log_uniform'),
            'bagging_temperature' : (0.0,1.0),
            'l2_leaf_reg' : (2,100)
        }
        self.bayessearchcv = BayesSearchCV(
            estimator=self.model,
            search_spaces=self.search_space,
            cv=5,
            n_points=1,
            n_iter=50,
            verbose=False,
            random_state=self.seed
        )
        self.best_params = self.report_perf(
            self.bayessearchcv,
            pd.concat([self.data['X_train'] , self.data['X_valid']], axis=0), 
            pd.concat([self.data['y_train'] , self.data['y_valid']], axis=0),
            )
        return self.best_params

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