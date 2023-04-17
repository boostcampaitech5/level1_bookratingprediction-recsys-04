# level1_bookratingprediction-recsys-04

```bash

    usage: main.py [-h] [--data_path DATA_PATH] [--saved_model_path SAVED_MODEL_PATH] [--model {FM,FFM,NCF,WDN,DCN,CNN_FM,DeepCoNN}] [--data_shuffle DATA_SHUFFLE] [--test_size TEST_SIZE] [--seed SEED]
               [--use_best_model USE_BEST_MODEL] [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--lr LR] [--loss_fn {MSE,RMSE}] [--optimizer {SGD,ADAM}] [--weight_decay WEIGHT_DECAY] [--device {cuda,cpu}]
               [--embed_dim EMBED_DIM] [--dropout DROPOUT] [--mlp_dims MLP_DIMS] [--num_layers NUM_LAYERS] [--cnn_embed_dim CNN_EMBED_DIM] [--cnn_latent_dim CNN_LATENT_DIM] [--vector_create VECTOR_CREATE]
               [--deepconn_embed_dim DEEPCONN_EMBED_DIM] [--deepconn_latent_dim DEEPCONN_LATENT_DIM] [--conv_1d_out_dim CONV_1D_OUT_DIM] [--kernel_size KERNEL_SIZE] [--word_dim WORD_DIM] [--out_dim OUT_DIM]

parser

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        Data path를 설정할 수 있습니다.
  --saved_model_path SAVED_MODEL_PATH
                        Saved Model path를 설정할 수 있습니다.
  --model {FM,FFM,NCF,WDN,DCN,CNN_FM,DeepCoNN}
                        학습 및 예측할 모델을 선택할 수 있습니다.
  --data_shuffle DATA_SHUFFLE
                        데이터 셔플 여부를 조정할 수 있습니다.
  --test_size TEST_SIZE
                        Train/Valid split 비율을 조정할 수 있습니다.
  --seed SEED           seed 값을 조정할 수 있습니다.
  --use_best_model USE_BEST_MODEL
                        검증 성능이 가장 좋은 모델 사용여부를 설정할 수 있습니다.
  --batch_size BATCH_SIZE
                        Batch size를 조정할 수 있습니다.
  --epochs EPOCHS       Epoch 수를 조정할 수 있습니다.
  --lr LR               Learning Rate를 조정할 수 있습니다.
  --loss_fn {MSE,RMSE}  손실 함수를 변경할 수 있습니다.
  --optimizer {SGD,ADAM}
                        최적화 함수를 변경할 수 있습니다.
  --weight_decay WEIGHT_DECAY
                        Adam optimizer에서 정규화에 사용하는 값을 조정할 수 있습니다.
  --device {cuda,cpu}   학습에 사용할 Device를 조정할 수 있습니다.
  --embed_dim EMBED_DIM
                        FM, FFM, NCF, WDN, DCN에서 embedding시킬 차원을 조정할 수 있습니다.
  --dropout DROPOUT     NCF, WDN, DCN에서 Dropout rate를 조정할 수 있습니다.
  --mlp_dims MLP_DIMS   NCF, WDN, DCN에서 MLP Network의 차원을 조정할 수 있습니다.
  --num_layers NUM_LAYERS
                        에서 Cross Network의 레이어 수를 조정할 수 있습니다.
  --cnn_embed_dim CNN_EMBED_DIM
                        CNN_FM에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.
  --cnn_latent_dim CNN_LATENT_DIM
                        CNN_FM에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.
  --vector_create VECTOR_CREATE
                        DEEP_CONN에서 text vector 생성 여부를 조정할 수 있으며 최초 학습에만 True로 설정하여야합니다.
  --deepconn_embed_dim DEEPCONN_EMBED_DIM
                        DEEP_CONN에서 user와 item에 대한 embedding시킬 차원을 조정할 수 있습니다.
  --deepconn_latent_dim DEEPCONN_LATENT_DIM
                        DEEP_CONN에서 user/item/image에 대한 latent 차원을 조정할 수 있습니다.
  --conv_1d_out_dim CONV_1D_OUT_DIM
                        DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.
  --kernel_size KERNEL_SIZE
                        DEEP_CONN에서 1D conv의 kernel 크기를 조정할 수 있습니다.
  --word_dim WORD_DIM   DEEP_CONN에서 1D conv의 입력 크기를 조정할 수 있습니다.
  --out_dim OUT_DIM     DEEP_CONN에서 1D conv의 출력 크기를 조정할 수 있습니다.
```

```
    #1
    python main.py --model=DCN

```
