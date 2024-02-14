import torch

from torchfm.layer import CategoricalDnn,FeaturesEmbedding, CrossNetwork, MultiLayerPerceptron


class EntityDCN(torch.nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.

    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    #기존 field_dims는 누적합때매 사용한거고 여기선 누적합 안 쓰니 빼고
    #cate 컬럼을 저장해줄 파리미터로 사용
    def __init__(self, categorical_nuniques_to_dims,num_numerical_features,
                 fc_layers_construction,field_dims, 
                 embed_dim, num_layers, mlp_dims, dropout, output_dim):
        super().__init__()
        
        self.output_dim = output_dim
        
        #self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embedding = CategoricalDnn(categorical_nuniques_to_dims,
                                                         num_numerical_features,
                                                         embed_dim,
                                                         fc_layers_construction,
                                                         field_dims,
                                                         dropout)
        
   
        #아래 주석처리는 다양한 방식으로 embedding-input-size과 output-size를 설정함에 따라 나온 값들의 변화
        #self.embed_output_dim = len(embedding_columns)*embed_dim + num_numerical_features + (len(categorical_nuniques_to_dims)-len(embedding_columns))
        #self.embed_output_dim = sum(field_dims) + num_numerical_features
        #self.embed_output_dim = sum_unique + num_numerical_features 
        #self.embed_output_dim = sum(categorical_nuniques_to_dims[1]) + num_numerical_features
        #self.embed_output_dim = sum(field_dims) + fc_layers_construction[-1] + (len(categorical_nuniques_to_dims)-len(embedding_columns))       
        #self.embed_output_dim = len(categorical_nuniques_to_dims+ num_numerical_features) *embed_dim 
        #self.cn = CrossNetwork(self.input_dim, num_layers)
        #self.mlp = MultiLayerPerceptron(self.input_dim, mlp_dims, dropout, output_layer=False)
        #self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, 1)       

        self.embed_output_dim = len(categorical_nuniques_to_dims)*embed_dim + num_numerical_features
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.linear = torch.nn.Linear(mlp_dims[-1] + self.embed_output_dim, output_dim)

        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        embed_x = self.embedding(x)
        
        x_l1 = self.cn(embed_x)
        h_l2 = self.mlp(embed_x)

        #해당 cat방식은 parallel한 방식으로 각 network에서 수행한 Cross-network & Deep-nework를 평행하게 결합한 방식(여기서는 parallel방식 선택)
        x_stack = torch.cat([x_l1, h_l2], dim=1)
   
        #embeding값 -> Cross-network -> Cross-network출력값-> Deep-nework로 바꾸는 stack방식도 존재함
        #x_stack = self.mlp(x_l1)
        
        p = self.linear(x_stack)
        
        if self.output_dim == 1:
            #binary-classificatio activation function : sigmoid
            return torch.sigmoid(p.squeeze(1))
        elif self.output_dim == 3:
            #multi-classificatio activation function : softmax
            return torch.nn.functional.softmax(p,dim=1)
        else:
            print(f'output_dim error : ',self.output_dim)
            exit()


