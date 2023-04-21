import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader

### cat boost
import pickle
import catboost as cb
import catboost
import optuna
from sklearn.metrics import mean_squared_error
from skopt import BayesSearchCV
import numpy as np
from sklearn.model_selection import KFold

def main(args):
    Setting.seed_everything(args.seed)

    data = context_data_load(args)
    data = context_data_split(args, data)
    # [['user_id', 'isbn', 'age', 'location_city', 'location_state', 'location_country', 'category', 'publisher', 'language', 'book_author']]
    X_train = (data['X_train']*1).round().astype(int)
    X_test = (data['X_valid']*1).round().astype(int)
    y_train = data['y_train']
    y_test = data['y_valid']

    sub_test = (data['test']*1).round().astype(int)

    # X_train['location_score'] = X_train['location_score'].round().astype(int)
    # X_test['location_score'] =  X_test['location_score'].round().astype(int)
    # sub_test['location_score'] = sub_test['location_score'].round().astype(int)
    X_train['location'] = ((((X_train['location_city'] + X_train['location_state'] + X_train['location_country'])/ 3) + X_train['language'])/2)
    X_train['location'] = (X_train['location'] * 123).round().astype(int)
    X_test['location'] = ((((X_test['location_city'] + X_test['location_state'] + X_test['location_country'])/ 3) + X_test['language'])/2)
    X_test['location'] = (X_test['location'] * 123).round().astype(int)
    sub_test['location'] = ((((sub_test['location_city'] + sub_test['location_state'] + sub_test['location_country'])/ 3) + sub_test['language'])/2)
    sub_test['location'] = (sub_test['location'] * 123).round().astype(int)

    # X_train['user'] = ((X_train['user_id'] + X_train['age'] + X_train['category'])*100 // 3).astype(int)
    # X_test['user'] = ((X_test['user_id'] + X_test['age'] + X_test['category'])*100 // 3).astype(int)
    # sub_test['user'] = ((sub_test['user_id'] + sub_test['age'] + sub_test['category'])*100 // 3).astype(int)

    # X_train = X_train.drop(["location_city", "location_state", "location_country"], axis=1)
    # X_test = X_test.drop(["location_city", "location_state", "location_country"], axis=1)
    # sub_test = sub_test.drop(["location_city", "location_state", "location_country"], axis=1)

    X_train = X_train.drop(["location_city", "location_state","location_country","language"], axis=1)
    X_test = X_test.drop(["location_city", "location_state","location_country","language"], axis=1)
    sub_test = sub_test.drop(["location_city", "location_state","location_country","language"], axis=1)

    print(sub_test.head())
    cat_features = [0,1,2,3,4,5,6]

    early_stop_rounds = 40

    # model = cb.CatBoostRegressor(
    #                             iterations=10000,
    #                             learning_rate=0.05,
    #                             depth=6,
    #                             cat_features=cat_features,
    #                             metric_period=1000,
    #                             task_type="GPU",
    #                             devices='0',
    #                             od_type='Iter',
    #                             od_wait= 1000,
    #                             verbose=1000)
    
    model = cb.CatBoostRegressor(
                                iterations=8000,
                                learning_rate=0.03,
                                bootstrap_type='Bayesian',
                                # subsample=0.7,
                                depth=10,
                                cat_features=cat_features,
                                loss_function='RMSE',
                                border_count=254,
                                # l2_leaf_reg=5,
                                task_type='GPU',
                                # random_strength=0.1,
                                # bagging_temperature=1,
                                # min_data_in_leaf=1,
                                # max_ctr_complexity=4,
                                verbose=500)

    # Fit the model to the train set
    model.fit(
            X_train, y_train,
            eval_set = (X_test,y_test),
            early_stopping_rounds=early_stop_rounds,
            use_best_model = True)
    
    with open('my_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Print RMSE
    print("RMSE:", mean_squared_error(y_test, y_pred, squared=False))
    
    test_pred = model.predict(sub_test)
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    submission['rating'] = test_pred
    submission.to_csv('jhg_model.csv', index=False)

    

if __name__ == '__main__':

    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=False, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    args = parser.parse_args()
    main(args)