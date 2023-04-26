import numpy as np
import torch
import torch.nn as nn

# FFM모델을 구현합니다.
# feature간의 상호작용을 파악하기 위해서 잠재백터를 두는 과정을 보여줍니다.
# FFM은 FM과 다르게 필드별로 여러개의 잠재백터를 가지므로 필드 개수만큼의 embedding parameter를 선언합니다.
class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, ffm_field_dims: np.ndarray, latent_dim: int):
        super().__init__()
        self.num_fields = len(ffm_field_dims) # 필드 개수(feature 개수 아닐까)
        self.embed_dim = sum(ffm_field_dims)
        self.vs = torch.nn.Parameter(torch.rand(self.num_fields, self.embed_dim, latent_dim), requires_grad = True) # 임베딩마다 임베딩 행렬을 만듦
        torch.nn.init.xavier_uniform_(self.vs) # 모든 field의 임베딩을 초기화
        self.linear = nn.Linear(self.embed_dim, 1, bias=True) # 출력: batch size * 1

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        linear = self.linear(x)
        feature_matmul = torch.matmul(x,self.vs).permute(1,0,2)
        feature_interaction_term = torch.matmul(feature_matmul,feature_matmul.permute(0, 2, 1))
        tri = torch.triu(feature_interaction_term,diagonal=1)
        output = linear + torch.sum(tri,1).sum(1,keepdim=True)
        return output

# factorization을 통해 얻은 feature를 embedding 합니다.
class FeaturesEmbedding(nn.Module):
    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int32)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)


    def forward(self, x: torch.Tensor):
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)

# 이미지 특징 추출을 위한 기초적인 CNN Layer를 정의합니다.
class CNN_Base(nn.Module):
    def __init__(self, ):
        super(CNN_Base, self).__init__()
        self.cnn_layer = nn.Sequential(
                                        nn.Conv2d(3, 6, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        nn.Conv2d(6, 12, kernel_size=1, stride=1),
                                        nn.Conv2d(12, 16, kernel_size=3, stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(kernel_size=3, stride=2),
                                        )
    def forward(self, x):
        x = self.cnn_layer(x)
        x = x.view(-1, 16 * 1 * 1)
        return x


# 기존 유저/상품 벡터와 이미지 벡터를 결합하여 FM으로 학습하는 모델을 구현합니다.
class CNN_FFM(torch.nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.field_dims = np.array([len(data['user2idx']), len(data['isbn2idx'])], dtype=np.uint32)
        self.embedding = FeaturesEmbedding(self.field_dims, args.cnn_embed_dim)
        self.cnn = CNN_Base()
        self.concat_field_dims = np.array((args.cnn_embed_dim, args.cnn_embed_dim, 16 * 1 * 1), dtype=np.longlong)
        self.ffm = FieldAwareFactorizationMachine(
                                        ffm_field_dims=self.concat_field_dims,
                                        latent_dim=args.cnn_latent_dim,
                                        )


    def forward(self, x):
        user_isbn_vector, img_vector = x[0], x[1]
        user_isbn_feature = self.embedding(user_isbn_vector)
        img_feature = self.cnn(img_vector)
        feature_vector = torch.cat([
                                    user_isbn_feature.view(-1, user_isbn_feature.size(1) * user_isbn_feature.size(2)),
                                    img_feature
                                    ], dim=1)
        output = self.ffm(feature_vector)
        return output.squeeze(1)