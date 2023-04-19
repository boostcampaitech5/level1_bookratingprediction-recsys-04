import lightgbm as lgb

class LGBM():
    def __init__(self, args,data) -> None:
        super().__init__()
        self.data = data
        self.dataloader()
        self.param = {
            'learning_rate' : args.lr,
            'max_depth' : 10,
            'boosting' : 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'is_training_metric' : True,
            'num_leaves':144,
            'feature_fraction' : 0.9,
            'bagging_function': 0.8,
            'bagging_freq' : 5,
            'seed' : args.seed,
            # "device" : "gpu"
        }

    def dataloader(self):
        self.train_ds = lgb.Dataset(self.data['X_train'], label=self.data['y_train'])
        self.valid_ds = lgb.Dataset(self.data['X_valid'], label=self.data['y_valid'])

    def train(self):
        self.trained_model = lgb.train(
            params=self.param,
            train_set=self.train_ds, 
            num_boost_round = 10000,
            valid_sets=[self.valid_ds],
            verbose_eval=100, 
            early_stopping_rounds=100
            )
        
    def pred(self, test):
        predicts = self.trained_model.predict(test)
        return predicts