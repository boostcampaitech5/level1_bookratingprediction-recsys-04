import argparse
import pandas as pd
from test_answer import test_rmse

def main(args):
    post_process_df = pd.read_csv('/opt/ml/data/test_rating_rule_based.csv')
    y_target = pd.read_csv(args.target_path)

    y_target.update(post_process_df,overwrite=True)

    # file name
    file_name = 'Cold_start_'+args.target_path.split('/')[-1]
    path = args.target_path.split('/')
    path[-1] = file_name
    path = '/'.join(path)

    test_rmse(y_target)
    y_target.to_csv(path)

if __name__ == "__main__":
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    arg('--target_path', type=str, default='/opt/ml/code/submit/CatBoost.csv', help="cold start 후처리")
    
    args = parser.parse_args()
    main(args)