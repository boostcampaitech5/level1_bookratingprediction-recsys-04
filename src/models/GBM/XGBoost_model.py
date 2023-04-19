import xgboost as xgb
from wandb.xgboost import WandbCallback
from sklearn.preprocessing import LabelEncoder


class XGBoost():
    def __init__(self, args, data) -> None:
        super().__init__()
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist",
            }
        self.data = data
        self.seed = args.seed
        self.num_boost_round = args.num_boost_round
        self.reg = args.reg
        self.evals = [(data['X_valid'], data['y_valid'])]

    def dataloader(self,data):
        self.dtrain_reg = xgb.DMatrix(data['X_train'],data['y_train'], enable_categorical=True)
        self.dvalid_reg = xgb.DMatrix(data['X_valid'],data['y_valid'], enable_categorical=True)
        self.evals = [(self.dvalid_reg, 'validation')]

    def train(self):
        if self.reg:
            self.dataloader(self.data)
            self.model = xgb.train(
                params=self.params,
                dtrain = self.dtrain_reg,
                num_boost_round = self.num_boost_round,
                # eval_metric='rmse',
                evals=self.evals,
                verbose_eval = 1000,
                early_stopping_rounds = 50,
                callbacks=[WandbCallback(log_model=True)]
                )
        # else:
        #     self.model = xgb.XGBClassifier(
        #         num_boost_round = self.num_boost_round,
        #         eval_metric='rmse',
        #         evals=self.evals,
        #         verbose_eval = 1000,
        #         early_stopping_rounds = 50,
        #         callbacks=[WandbCallback(log_model=True)],
        #         # n_estimators=1000,
        #         # learning_rate=0.01, 
        #         max_depth=10, 
        #         random_state = self.seed,
        #         tree_method="gpu_hist",
        #         ).fit(
        #         X=self.data['X_train'], 
        #         y=self.data['y_train'],)
    def pred(self, test):
        if self.reg:
            self.dtest_reg = xgb.DMatrix(test, enable_categorical=True)
            predict = self.model.predict(self.dtest_reg)
        # else:
        #     predict = self.model.predict(test)
        #     sample = self.model.predict_proba(test)
        return predict