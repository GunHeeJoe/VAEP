import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import torch.nn.init as init

class FeaturesLinear(torch.nn.Module):

    #def __init__(self, field_dims, categorical_nuniques_to_dims,num_numerical_features,fc_layers_construction,dropout_probability,output_dim=3):
    def __init__(self, field_dims,output_dim=3):
        super().__init__()
        
        #feature_linear는 embeddim=16이 아닌 output_dim=3으로 임베딩해야함
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        #self.fc = CategoricalDnn(categorical_nuniques_to_dims,num_numerical_features,fc_layers_construction,field_dims,output_dim, dropout_probability)

        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

        
    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """

        #x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return torch.sum(self.fc(x), dim=1) + self.bias


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """       
        x = x + x.new_tensor(self.offsets).unsqueeze(0)

        return self.embedding(x)

class CategoricalDnn(torch.nn.Module):
    def __init__(self,
                 categorical_nuniques_to_dims,
                 num_numerical_features,
                 embed_dim,
                 fc_layers_construction,
                 field_dims,
                 dropout_probability):
        super(CategoricalDnn, self).__init__()
        
        self.categorical_nuniques_to_dims = categorical_nuniques_to_dims
        self.field_dims = field_dims
        self.embed_dim = embed_dim
        self._make_embedding_layers(categorical_nuniques_to_dims)
        self._make_fc_layers(num_numerical_features, fc_layers_construction, dropout_probability)


    #categorical데이터에 대해서만 임베딩을 실시한다.
    def _make_embedding_layers(self, nuniques_to_dims):
        """
        Define Embedding Layers of categorical variables.
         Properties:
             self.embedding_layer_list         :   List of Embedding Layers applied to categorical variable
             self.num_categorical_features     :   Total number of categorical variables before Embedding
             self.num_embedded_features        :   Total number of embedded features from categorical variables
             self.num_each_embedded_features   :   Number of embedded features for each categorical variable
        """
        #categorical feature개수 = 110개
        self.num_categorical_features = len(nuniques_to_dims)
        self.num_embedded_features = 0
        self.num_each_embedded_features = []
        self.embedding_layer_list = torch.nn.ModuleList()
        
        for i,nunique_to_dim in enumerate(nuniques_to_dims):
            # if i not in embedding_columns:
            #     continue
            
            num_uniques = nunique_to_dim[0] + 1
            
            #아래 주석은 다양한 embed_size를 설정한 경우
            #target_dim = nunique_to_dim[1]
            #target_dim = num_uniques//2
            #num_uniques = sum(self.field_dims)
            #target_dim = nunique_to_dim[1]
            #target_dim = nunique_to_dim[0] + 1
        
            target_dim = self.embed_dim
       
            self.embedding_layer_list.append(
                torch.nn.Sequential(
                    #누적합으로도 임베딩 사이즈를 설정한 경우
                    #torch.nn.Embedding(sum(self.field_dims), target_dim),
                    torch.nn.Embedding(num_uniques, target_dim),
                    torch.nn.BatchNorm1d(target_dim),
                    torch.nn.ReLU(inplace=False),
                )
            )
            
            #모든 target_dim를 쌓아놓음
            #ex)categorical개수가 110개 이면 -> 110 x 3 = 330
            self.num_embedded_features += target_dim
            
            #각 target_dim개수 : 모두 3만 있음 => [3,3,3,3,3,3,.......]
            self.num_each_embedded_features.append(target_dim)
        
    #numerical 임베딩을 실시한다.
    #본 연구에서는 numerical feature는 정규화 후 linear과정까지 거치고 categoricla과 concat하면
    #성능하락을 보임. 즉 해당 linear과정에서 정보 손실이 발생하므로 numerical feature는 정규화만 수행함
    def _make_fc_layers(self, num_numerical_features, fc_layers_construction, dropout_p):
        """
        Define input layer, hidden layer and output layer of the Fully Connected Layers.
         Properties:
             self.fc_layer_list   :   List of Fully Connected Layers that take embedded categorical features and
                                      numerical variables as inputs
             self.output_layer    :   Output layer
        """
        

        num_input = num_numerical_features
        #numerical만 linear할 것

        self.fc_layer_list = torch.nn.ModuleList()
        
        for i,num_output in enumerate(fc_layers_construction):
            layer = torch.nn.Sequential(
                    torch.nn.Dropout(dropout_p) if dropout_p else torch.nn.Sequential(),
                    torch.nn.Linear(num_input, num_output),
                    torch.nn.BatchNorm1d(num_output),
                    torch.nn.ReLU(inplace=False)
                )

            self.fc_layer_list.append(layer)
            num_input = num_output
        
        #해당 작업은 최종결과 3개로 출력했을 때, 즉 마무리 출력결과를 위한 output-layer이다
        # 여기서는 임베딩작업만이 목적이므로 해당 layer사용을 안함
        # self.output_layer = torch.nn.Sequential(
        #     torch.nn.Dropout(dropout_p) if dropout_p else torch.nn.Sequential(),
        #     torch.nn.Linear(num_input, 3)
        # )

    def forward(self, input):
        # Split the input into categorical variables and numerical variables

        categorical_input = input[:, :self.num_categorical_features].long()
        #categorical_input = categorical_input + categorical_input.new_tensor(self.offsets).unsqueeze(0)
        
        numerical_input = input[:, self.num_categorical_features:]

        embedded = torch.zeros(input.size()[0], self.num_embedded_features)

        #아래 주석은 누적합을 사용했을 때 embedded size설정 코드
        #embedded = torch.zeros(input.size()[0], sum_unique)
        
        start_index = 0
        
        # Embed the categorical variables
        for i, emb_layer in enumerate(self.embedding_layer_list):
            #모든 embedded_feature는 3이므로 gorl_index는 3,6,9,12,15....219,222이 나올 예정
            gorl_index = start_index + self.num_each_embedded_features[i]
                  
            embedded[:, start_index:gorl_index] = emb_layer(categorical_input[:, i])

            start_index = gorl_index
        
        embedded = embedded.to('cuda:0')
      
        #numerical data는 fully-connected-layer를 거치지 않음
        # for hidden_layer in self.fc_layer_list:
        #     numerical_input = hidden_layer(numerical_input.float())
        
        out = torch.cat([embedded.float(),numerical_input.float()], dim=1).to('cuda:0')

        #cate&num를 합친 out데이터를 output_layer를 한번 통과할까 여부도 제외시킴
        #return self.output_layer(out)
        return out
    
class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        
        #embedding된 x값이 나옴([4096,64])
        #embedding된 x값을 더한 후 제곱한 값
        #embedding된 x값을 제곱한 후 더한 값
        #두 값의 차이=ix는 [4096,64]형태의 값이 들어있음
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        #print("before ix : ",ix.shape,ix)
        
        if self.reduce_sum:
            #print("if sum value : ",torch.sum(ix,dim=1,keepdim=True).shape, torch.sum(ix,dim=1,keepdim=True))
            ix = torch.sum(ix, dim=1, keepdim=True)
        #print("after ix : " ,ix.shape, ix)
    
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 3))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class InnerProductNetwork(torch.nn.Module):

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        return torch.sum(x[:, row] * x[:, col], dim=2)


class OuterProductNetwork(torch.nn.Module):

    def __init__(self, num_fields, embed_dim, kernel_type='mat'):
        super().__init__()
        num_ix = num_fields * (num_fields - 1) // 2
        if kernel_type == 'mat':
            kernel_shape = embed_dim, num_ix, embed_dim
        elif kernel_type == 'vec':
            kernel_shape = num_ix, embed_dim
        elif kernel_type == 'num':
            kernel_shape = num_ix, 1
        else:
            raise ValueError('unknown kernel type: ' + kernel_type)
        self.kernel_type = kernel_type
        self.kernel = torch.nn.Parameter(torch.zeros(kernel_shape))
        torch.nn.init.xavier_uniform_(self.kernel.data)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        if self.kernel_type == 'mat':
            kp = torch.sum(p.unsqueeze(1) * self.kernel, dim=-1).permute(0, 2, 1)
            return torch.sum(kp * q, -1)
        else:
            return torch.sum(p * q * self.kernel.unsqueeze(0), -1)


class CrossNetwork(torch.nn.Module):

    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
    
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])
        

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """

        x0 = x
        for i in range(self.num_layers):
            #print(x.shape)
            xw = self.w[i](x)
            x = x0 * (xw + self.b[i]) + x
        
        return x


class AttentionalFactorizationMachine(torch.nn.Module):

    def __init__(self, embed_dim, attn_size, dropouts):
        super().__init__()
        self.attention = torch.nn.Linear(embed_dim, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.fc = torch.nn.Linear(embed_dim, 1)
        self.dropouts = dropouts

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        num_fields = x.shape[1]
        row, col = list(), list()
        for i in range(num_fields - 1):
            for j in range(i + 1, num_fields):
                row.append(i), col.append(j)
        p, q = x[:, row], x[:, col]
        inner_product = p * q
        attn_scores = F.relu(self.attention(inner_product))
        attn_scores = F.softmax(self.projection(attn_scores), dim=1)
        attn_scores = F.dropout(attn_scores, p=self.dropouts[0], training=self.training)
        attn_output = torch.sum(attn_scores * inner_product, dim=1)
        attn_output = F.dropout(attn_output, p=self.dropouts[1], training=self.training)
        return self.fc(attn_output)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class AnovaKernel(torch.nn.Module):

    def __init__(self, order, reduce_sum=True):
        super().__init__()
        self.order = order
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        batch_size, num_fields, embed_dim = x.shape
        a_prev = torch.ones((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
        for t in range(self.order):
            a = torch.zeros((batch_size, num_fields + 1, embed_dim), dtype=torch.float).to(x.device)
            a[:, t+1:, :] += x[:, t:, :] * a_prev[:, t:-1, :]
            a = torch.cumsum(a, dim=1)
            a_prev = a
        if self.reduce_sum:
            return torch.sum(a[:, -1, :], dim=-1, keepdim=True)
        else:
            return a[:, -1, :]
