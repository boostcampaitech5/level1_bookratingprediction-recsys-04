import xgboost as xgb
from wandb.xgboost import WandbCallback


class XGBoost():
    def __init__(self, args, data) -> None:
        super().__init__()
        self.params = {
            "objective": "reg:squarederror",
            "tree_method": "gpu_hist",
            }
        self.data = data
        self.dataloader(data)
        self.seed = args.seed
        self.num_boost_round = args.num_boost_round

    def dataloader(self,data):
        self.dtrain_reg = xgb.DMatrix(data['X_train'],data['y_train'], enable_categorical=True)
        self.dvalid_reg = xgb.DMatrix(data['X_valid'],data['y_valid'], enable_categorical=True)
        self.evals = [(self.dvalid_reg, 'validation')]

    def train(self):
        self.model = xgb.train(
            params=self.params,
            dtrain = self.dtrain_reg,
            num_boost_round = self.num_boost_round,
            evals=self.evals,
            verbose_eval = 1000,
            early_stopping_rounds = 50,
            callbacks=[WandbCallback(log_model=True)]
            )
        
    def pred(self, test):
        self.dtest_reg = xgb.DMatrix(test, enable_categorical=True)
        predict = self.model.predict(self.dtest_reg)
        return predict