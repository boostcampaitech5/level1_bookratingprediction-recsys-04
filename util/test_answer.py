import argparse
import pandas as pd
from sklearn.metrics import mean_squared_error

def main(args):
    y = pd.read_csv(args.answer_path)
    y_pred = pd.read_csv(args.target_path)
    loss = mean_squared_error(y['rating'], y_pred['rating'])**0.5
    print("RMSE Loss : ", loss)

def test_rmse(y_pred):
    y = pd.read_csv('/opt/ml/code/best.csv')
    loss = mean_squared_error(y['rating'], y_pred)**0.5
    print("RMSE Loss : ", loss)
    return loss

if __name__ == "__main__":
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--answer_path', type=str, default='/opt/ml/code/best.csv', help="RMSE 스코어를 비교하기 위한 BEST CSV 파일 경로")
    arg('--target_path', type=str, default='/opt/ml/code/answer_train.csv', help='RMSE 스코어를 비교할 CSV 파일 경로.')
    
    args = parser.parse_args()
    main(args)