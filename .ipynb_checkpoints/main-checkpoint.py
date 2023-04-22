import time
import argparse
import pandas as pd
from src.utils import Logger, Setting, models_load
from src.data import context_data_load, context_data_split, context_data_loader
from src.data import dl_data_load, dl_data_split, dl_data_loader
from src.data import image_data_load, image_data_split, image_data_loader
from src.data import text_data_load, text_data_split, text_data_loader
from src.train import train, test
from surprise import Reader
from surprise.dataset import DatasetAutoFolds
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, StackingRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor


def main(args):
    Setting.seed_everything(args.seed)


    ######################## DATA LOAD
    print(f'--------------- {args.model} Load Data ---------------')
    if args.model in ('FM', 'FFM','XGBoost','CatBoost','RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor'):
        data = context_data_load(args)
    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_load(args)
    elif args.model == 'CNN_FM':
        data = image_data_load(args)
    elif args.model == 'DeepCoNN':
        import nltk
        nltk.download('punkt')
        data = text_data_load(args)
    elif args.model == 'SVD':
        data = context_data_load(args)
    else:
        pass


    ######################## Train/Valid Split
    print(f'--------------- {args.model} Train/Valid Split ---------------')
    if args.model in ('FM', 'FFM'):
        data = context_data_split(args, data)
        data = context_data_loader(args, data)

    elif args.model in ('NCF', 'WDN', 'DCN'):
        data = dl_data_split(args, data)
        data = dl_data_loader(args, data)

    elif args.model=='CNN_FM':
        data = image_data_split(args, data)
        data = image_data_loader(args, data)

    elif args.model=='DeepCoNN':
        data = text_data_split(args, data)
        data = text_data_loader(args, data)
    
    elif args.model in ('XGBoost','CatBoost','RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor'):
        data = context_data_split(args, data)

    else:
        pass

    ####################### Setting for Log
    setting = Setting()

    log_path = setting.get_log_path(args)
    setting.make_dir(log_path)

    logger = Logger(args, log_path)
    logger.save_args()


    ######################## Model
    print(f'--------------- INIT {args.model} ---------------')
    if(args.model == 'XGBoost'):
        model = t(data['X_train'], data['y_train'])
    elif args.model == 'CatBoost':
        model = CatBoostRegressor()
    elif args.model == 'RandomForestRegressor':
        model = RandomForestRegressor()
    elif args.model == 'ExtraTreesRegressor':
        model = ExtraTreesRegressor()
    elif args.model == 'HistGradientBoostingRegressor':
        model = HistGradientBoostingRegressor(max_depth = 8,max_iter=5000, learning_rate=0.01)
    else:
        model = models_load(args,data)
    


    ######################## TRAIN
    print(f'--------------- {args.model} TRAINING ---------------')
    if(args.model == 'XGBoost'):
        model.fit(data['X_train'], data['y_train'])
    elif args.model == 'CatBoost':
        model.fit(data['X_train'], data['y_train'])
    elif args.model == 'RandomForestRegressor':
        model.fit(data['X_train'], data['y_train'])
    elif args.model == 'ExtraTreesRegressor':
        model.fit(data['X_train'], data['y_train'])
    elif args.model == 'HistGradientBoostingRegressor':
        model.fit(data['X_train'], data['y_train'])
    elif args.model == 'SVD':
        model.fit(data['X_train'], data['y_train'])
    else:
        model = train(args, model, data, logger, setting)
        
    



    ######################## INFERENCE
    print(f'--------------- {args.model} PREDICT ---------------')
    if(args.model == 'XGBoost','CatBoost','RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor'):
        predicts = model.predict(data['test'])
    elif args.model == 'SVD':
        predicts = model(test_data)
    else:
        predicts = test(args, model, data, setting)


    ######################## SAVE PREDICT
    print(f'--------------- SAVE {args.model} PREDICT ---------------')
    submission = pd.read_csv(args.data_path + 'sample_submission.csv')
    if args.model in ('FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN','XGBoost','SVD','CatBoost','RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor'):
        submission['rating'] = predicts
    else:
        pass

    filename = setting.get_submit_filename(args)
    submission.to_csv(filename, index=False)


if __name__ == "__main__":


    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument


    ############### BASIC OPTION
    arg('--data_path', type=str, default='/opt/ml/data/', help='Data path를 설정할 수 있습니다.')
    arg('--saved_model_path', type=str, default='./saved_models', help='Saved Model path를 설정할 수 있습니다.')
    arg('--model', type=str, choices=['FM', 'FFM', 'NCF', 'WDN', 'DCN', 'CNN_FM', 'DeepCoNN','XGBoost','SVD','CatBoost','RandomForestRegressor','ExtraTreesRegressor','HistGradientBoostingRegressor'],
                                help='학습 및 예측할 모델을 선택할 수 있습니다.')
    arg('--data_shuffle', type=bool, default=True, help='데이터 셔플 여부를 조정할 수 있습니다.')
    arg('--test_size', type=float, default=0.2, help='Train/Valid split 비율을 조정할 수 있습니다.')
    arg('--seed', type=int, default=42, help='seed 값을 조정할 수 있습니다.')
    arg('--use_best_model', type=bool, default=True, help='검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.')


    ############### TRAINING OPTION
    arg('--batch_size', type=int, default=1024, help='Batch size를 조정할 수 있습니다.')
    arg('--epochs', type=int, default=10, help='Epoch 수를 조정할 수 있습니다.')
    arg('--lr', type=float, default=1e-3, help='Learning Rate를 조정할 수 있습니다.')
    arg('--loss_fn', type=str, default='RMSE', choices=['MSE', 'RMSE'], help='손실 함수를 변경할 수 있습니다.')
    arg('--optimizer', type=str, default='ADAM', choices=['SGD', 'ADAM'], help='최적화 함수를 변경할 수 있습니다.')
    arg('--weight_decay', type=float, default=1e-6, help='Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.')


    ############### GPU
    arg('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='학습에 사용할 Device를 조정할 수 있습니다.')


    ############### FM, FFM, NCF, WDN, DCN Common OPTION
    arg('--embed_dim', type=int, default=16, help='FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--dropout', type=float, default=0.2, help='NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.')
    arg('--mlp_dims', type=list, default=(16, 16), help='NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.')


    ############### DCN
    arg('--num_layers', type=int, default=3, help='에서 Cross Network의 레이어 수를 조정할 수 있습니다.')


    ############### CNN_FM
    arg('--cnn_embed_dim', type=int, default=64, help='CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--cnn_latent_dim', type=int, default=12, help='CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')


    ############### DeepCoNN
    arg('--vector_create', type=bool, default=True, help='DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.')
    arg('--deepconn_embed_dim', type=int, default=32, help='DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.')
    arg('--deepconn_latent_dim', type=int, default=10, help='DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.')
    arg('--conv_1d_out_dim', type=int, default=50, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')
    arg('--kernel_size', type=int, default=3, help='DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.')
    arg('--word_dim', type=int, default=768, help='DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.')
    arg('--out_dim', type=int, default=32, help='DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.')

    ############### XGBoost
    arg('--max_depth', type=int, default=6, help='Xgboost에서 트리의 최대 깊이를 설정')
    arg('--eta', type=int, default=0.3, help='Xgboost에서의 학습률로 클수록 과적합 가능성이 높음.')
    arg('--num_boost_around', type=int, default=10, help='학습에 사용될 weak learner의 반복 수')
    arg('--min_child_weight', type=int, default=1, help='leaf node에 포함되는 최소 관측치의 수로 작을 수록 과적합 가능성 높음')


    args = parser.parse_args()
    main(args)
