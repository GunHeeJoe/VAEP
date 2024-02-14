from torch.utils.data import Sampler

from torchfm.model.afn import AdaptiveFactorizationNetwork
from torchfm.model.xdfm import ExtremeDeepFactorizationMachineModel
from torchfm.model.wd import WideAndDeepModel
from torchfm.model.pnn import ProductNeuralNetworkModel
from torchfm.model.nfm import NeuralFactorizationMachineModel
from torchfm.model.ncf import NeuralCollaborativeFiltering
from torchfm.model.lr import LogisticRegressionModel
from torchfm.model.fnn import FactorizationSupportedNeuralNetworkModel
from torchfm.model.fnfm import FieldAwareNeuralFactorizationMachineModel

from torchfm.model.fm import FactorizationMachineModel
from torchfm.model.ffm import FieldAwareFactorizationMachineModel
from torchfm.model.dfm import DeepFactorizationMachineModel

from torchfm.model.dcn import DeepCrossNetworkModel
from torchfm.model.EntityDCN import EntityDCN

from torchfm.model.afm import AttentionalFactorizationMachineModel
from torchfm.model.afi import AutomaticFeatureInteractionModel

from sklearn.preprocessing import StandardScaler

from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

import torch
import numpy as np
import pandas as pd
import random 

import os
import warnings
import tqdm
import copy


from torch.utils.tensorboard import SummaryWriter
CUDA_LAUNCH_BLOCKING=1
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

warnings.simplefilter(action='ignore', category=FutureWarning)

from datetime import datetime
now = datetime.now()
schedule = str(now.date()) + " " + str(now.hour) + "시 " + str(now.minute) + "분"

model_name = 'entitiy_dcn'
epoch = 1
learning_rate = 1e-4
weight_decay =  1e-2

name = f'runs/ now={schedule} & model={model_name} : epoch={epoch} & lr={learning_rate} & weight={weight_decay}'
writer = SummaryWriter(name)

class OverSampler(Sampler):
    def __init__(self, class_vector, batch_size):

        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector
        self.indices = list(range(len(self.class_vector)))
        self.batch_size = batch_size
        uni_label = torch.unique(class_vector)
        uni_label_sorted, _ = torch.sort(uni_label)

        # [0,1,2] 총 3개의 class_vector값이 label화 되어ㅣㅆ음
        uni_label_sorted = uni_label_sorted.detach().numpy()
        label_bin = torch.bincount(class_vector.int()).detach().numpy()

        # 기존 train_data_loader에는 {0.0: 983346, 1.0: 13272, 2.0: 2159}개가 들어있어서
        # dictionary형태로 변환
        label_to_count = dict(zip(uni_label_sorted, label_bin))

        # 전체개수=998,777=len(class_vector) // 전체 target개수
        # label_to_count[float(label)] : 각 label별 개수
        # 즉 각 target의 가중치를 계산함
        # target=0 : 998777 / 983347 = 1.015인 weight가 983346개 있음
        # target=1 : 998777 / 13272 = 75.25인 weight가 13272개 있음
        # target=2 : 998777 / 2159 = 462.61인 weight가 2159개 있음
        weights = [len(class_vector) / label_to_count[float(label)]
                       for label in class_vector]

        # 즉 target별 가중치를 이용해서 모든 target이 골고루 sampler되도록 sampling함
        self.weights = torch.DoubleTensor(weights)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.batch_size, replacement=True))

    def __len__(self):
        return len(self.class_vector)
    
def load_targets(data_loader):
    targets, predicts = list(), list()

    with torch.no_grad():
         for fields, target in data_loader:
             targets.extend(target.tolist())

    return np.array(targets)

class CustomDataset():  
    def __init__(self, train_valid_test, scaler, folder):
        super(CustomDataset, self).__init__()
        self.scaler = scaler
        self.folder = folder

        self.X, self.Y = self.get_xy(train_valid_test)

        self.cate_X, self.num_X = self.process_data(train_valid_test)
        self.X = pd.concat([self.cate_X,self.num_X],axis=1)

        self.items = pd.concat([self.cate_X, self.num_X], axis=1).astype('float').to_numpy()
        self.targets = self.Y.astype('float').to_numpy()
        
        self.params = self.get_model_parameters()
        
        #가중치가 float32타입이므로 linear할 때, 타입오류뜸으로 float으로 통일함
        self.items = self.X.astype('float').to_numpy()
        self.targets = self.Y.astype('float').to_numpy()

        #categorical만 임베딩을 실시하는데, embed_input_size를 누적합으로도 사용할 때가 있음
        #본 연구에서는 사용하지 않았지만, 모든 임베딩 input size를 누적합으로 사용하는 방법도 있다.
        self.field_dims = np.max(self.cate_X, axis=0) + 1

        # print("item.shape : ", self.items.shape)
        # print("target.shape : ", self.targets.shape)

        self.user_field_idx = np.array((0, ), dtype=np.compat.long)
        self.item_field_idx = np.array((1,), dtype=np.compat.long)

    def read_data(self, data_type):
        X = pd.read_csv(f'../data/{self.folder}/{data_type}/X_{data_type}.csv')
        Y = pd.read_csv(f'../data/{self.folder}/{data_type}/Y_{data_type}.csv')
        return X, Y

    def get_xy(self, train_valid_test):
        data_type = {1: 'train', 2: 'valid', 3: 'test'}.get(train_valid_test, None)
        if data_type:
            X, Y = self.read_data(data_type)

            #train-dataset은 shuffling수행
            #dataloader에서는 sampler객체때문에, shiffle인자를 받을 수가 없음
            if train_valid_test == 1:
                indices = np.random.permutation(X.shape[0])
                X = X.iloc[indices].reset_index(drop=True)
                Y = Y.iloc[indices].reset_index(drop=True)
            return X, Y
        else:
            raise ValueError("Invalid train_valid_test value. It should be 1, 2, or 3.")

    def process_data(self, train_valid_test):
        #categorical, nunerical 데이터를 나눠서 저장한다음 합칠예정
        cate_X = pd.DataFrame()
        num_X = pd.DataFrame()
        numerical_features = []

        for col in self.X.columns:
            #골득실은 int로 되어있지만, 실제로는 카테고리가 아닌 수치형속성이므로 numercal로 타입변환 시켜줌
            if col.startswith('goal'):
                self.X[col] = self.X[col].astype('float64')

            #bool, int32, int64타입은 categorical이므로 int32로 통일
            if self.X[col].dtype in ["bool", "int32", 'int64']:
                cate_X[col] = self.X[col].astype("int32")
            #float64타입은 numerical데이터인데, 정규화 수행함
            elif self.X[col].dtype == 'float64':
                numerical_features.append(col)
            else:
                self.X = self.X.drop(columns=col)
                #print("drop object column : ",col)
                #raise ValueError(f"Unexpected type for column {col}: {self.X[col].dtype}")
        
        # Apply fit_transform or transform depending on the dataset type
        if train_valid_test == 1:           
            a = self.scaler.fit_transform(self.X[numerical_features])
        else:
            a = self.scaler.transform(self.X[numerical_features])

        num_X[numerical_features] = pd.DataFrame(a, index=self.X.index)

        return cate_X, num_X

    # Deep cross newtwork모델의 파라미터를 설정하는 함수
    def get_model_parameters(self):
        # Get categorical variable unique values
        categorical_cols = self.cate_X.columns.tolist()
        
        #type클래스는 본 데이터에서(La Liga) 17,20번 action이 빠져있음
        #본 연구에서는 임베딩 input size를 각 컬럼 max개수 + 1로 설정함
        unique_values = [self.X[col].max() for col in categorical_cols]

        # 모든 embed output size를 3으로 설정함
        embed_dim = self.get_embed_dim()

        categorical_nuniques_to_dims = list(zip(unique_values, [embed_dim] * len(unique_values)))
        
        # 앞에 어디까지가 categorical data인지 알기위한 변수
        # Count number of numerical variables
        numerical_cols = self.num_X.columns.tolist()
        #fc_layer도 사용할 수 있음(수치형 속성은 정규화만 수행하기로 함)
        fc_layers_construction = [len(numerical_cols)]
        
        model_params = {
            'categorical_nuniques_to_dims': categorical_nuniques_to_dims,
            'num_numerical_features': len(numerical_cols),
            'fc_layers_construction': fc_layers_construction,
            'dropout_probability': 0.2,
            'categorical_feature': categorical_cols,
            'numerical_cols': numerical_cols
        }
        return model_params
    
    def get_embed_dim(self):
        return 3

    def __getitem__(self, index):
        return self.items[index], self.targets[index]

    def __len__(self):
        return len(self.targets)

    def get_params(self):
        return self.params
    

def get_model(name, dataset, output_dim):
    # Hyperparameters are empirically determined, not opitmized.
    field_dims = dataset.field_dims
    model_params = dataset.get_params()
    embed_dim = dataset.get_embed_dim()
    
    if name == 'lr':
        return LogisticRegressionModel(field_dims)
    elif name == 'fm':
        return FactorizationMachineModel(field_dims, embed_dim=256)
    # elif name == 'hofm':
    #     return HighOrderFactorizationMachineModel(field_dims, order=3, embed_dim=16)
    elif name == 'ffm':
        return FieldAwareFactorizationMachineModel(field_dims, embed_dim=4)
    elif name == 'fnn':
        return FactorizationSupportedNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'wd':
        return WideAndDeepModel(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'ipnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='inner', dropout=0.2)
    elif name == 'opnn':
        return ProductNeuralNetworkModel(field_dims, embed_dim=16, mlp_dims=(16,), method='outer', dropout=0.2)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'entitiy_dcn':
        return EntityDCN(model_params['categorical_nuniques_to_dims'], model_params['num_numerical_features'], model_params['fc_layers_construction'],
                           field_dims, embed_dim=embed_dim, num_layers=3,mlp_dims=(16,16), dropout=0.2,output_dim=output_dim)
    elif name == 'dcn':
        return DeepCrossNetworkModel(field_dims, embed_dim=16, num_layers=3, mlp_dims=(128,64), dropout=0.2)
    elif name == 'nfm':
        return NeuralFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'ncf':
        # only supports MovieLens dataset because for other datasets user/item colums are indistinguishable
        # assert isinstance(dataset, MovieLens20MDataset) or isinstance(dataset, MovieLens1MDataset)
        return NeuralCollaborativeFiltering(field_dims, embed_dim=16, mlp_dims=(16, 16), dropout=0.2,
                                            user_field_idx=dataset.user_field_idx,
                                            item_field_idx=dataset.item_field_idx)
    elif name == 'fnfm':
        return FieldAwareNeuralFactorizationMachineModel(field_dims, embed_dim=4, mlp_dims=(64,), dropouts=(0.2, 0.2))
    elif name == 'dfm':
        return DeepFactorizationMachineModel(field_dims, embed_dim=64, mlp_dims=(64, 64), dropout=0.2)
    elif name == 'xdfm':
        return ExtremeDeepFactorizationMachineModel(
            field_dims, embed_dim=16, cross_layer_sizes=(16, 16), split_half=False, mlp_dims=(16, 16), dropout=0.2)
    elif name == 'afm':
        return AttentionalFactorizationMachineModel(field_dims, embed_dim=16, attn_size=16, dropouts=(0.2, 0.2))
    elif name == 'afi':
        return AutomaticFeatureInteractionModel(
            field_dims, embed_dim=16, atten_embed_dim=64, num_heads=2, num_layers=3, mlp_dims=(400, 400), dropouts=(0, 0, 0))
    elif name == 'afn':
        return AdaptiveFactorizationNetwork(
            field_dims, embed_dim=16, LNN_dim=1500, mlp_dims=(400, 400, 400), dropouts=(0, 0, 0))
    else:
        raise ValueError('unknown model name: ' + name)


class Score_EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
        
class Concede_EarlyStopper(object):
    def __init__(self, num_trials, save_path):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_accuracy = 0
        self.save_path = save_path

    def is_continuable(self, model, accuracy):
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            self.trial_counter = 0
            torch.save(model, self.save_path)
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False

def train(model, optimizer, data_loader, criterion, epoch, device, log_interval=20):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, leave=True)

    model = model.to(device)
    
    for i, (fields, target) in enumerate(tk0):
        fields, target = fields.to(device), target.to(device)
        target = target.long()

        # y prediction
        y = model(fields)

        loss = criterion(y, target.squeeze(dim=-1))

        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
        if (i + 1) % log_interval == 0:
            tk0.set_postfix(loss=total_loss / log_interval)
            writer.add_scalar('Evaluation/train_loss', total_loss / log_interval, i + epoch * len(data_loader))
            total_loss = 0


def valid_evaluation(model, data_loader, criterion, epoch, device, log_interval=5):
    model.eval()
    targets, predicts = list(), list()
    
    total_loss = 0
    temp = []
    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, leave=True)):
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            
            #loss = criterion(y, target.float())
            target = target.long()
            loss = criterion(y, target.squeeze(dim=-1))
            
            total_loss += loss.item()
            temp.append(i + epoch * len(data_loader))
            
            
            if (i + 1) % log_interval == 0:                
                writer.add_scalar('Evaluation/valid_loss', total_loss, i + epoch * len(data_loader))                                
                total_loss = 0
            
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    # target shape: (batch, 1), predicts shape: (batch, 3)

    #예측값은 label=0/1/2 각 확률값이 나오지만
    #평가할때는 1 vs not 1 & 2 vs not 2와 같이 득점기준 auc & 실점기준 auc를 측정함    
    targets_score = (np.array(targets) == 1).reshape(-1)
    targets_concede = (np.array(targets) == 2).reshape(-1)

    predicts_score = np.stack((1-np.array(predicts)[:,1],np.array(predicts)[:,1]), axis=1)
    predicts_concede = np.stack((1-np.array(predicts)[:,2],np.array(predicts)[:,2]) , axis=1)

    eval_roc_auc= {'score_roc_auc' : roc_auc_score(targets_score, predicts_score[:,1]) ,
                  'concede_roc_auc' : roc_auc_score(targets_concede, predicts_concede[:,1])
                  }
    
    eval_brier = {'score_brier' : brier_score_loss(targets_score, predicts_score[:,1]),
                  'concede_brier' : brier_score_loss(targets_concede, predicts_concede[:,1])}
    
    roc_auc_result = str(eval_roc_auc['score_roc_auc']) + " / " + str(eval_roc_auc['concede_roc_auc'])
    brier_result = str(eval_brier['score_brier']) + " / " + str(eval_brier['concede_brier'])
    
    print("Valid score/concede ROC AUC", roc_auc_result)
    print("Valid score/concede brier", brier_result)

    for x in temp:
        #add_scaler(폴더이름,데이터, x좌표) : 1개의 값을 writer하는 것이고
        #add_scalers(폴더이름,dict, x좌표) : 여러개 값을 writer하는 것 => s붙이는것때매...개고생했네...
        writer.add_scalars('Evaluation/ROC AUC', eval_roc_auc, x)
        writer.add_scalars('Evaluation/brier', eval_brier, x)


    return eval_roc_auc['score_roc_auc'],eval_roc_auc['concede_roc_auc']

def test_evaluation(model,data_loader, criterion, device):
    model.eval()
    targets, predicts = list(), list()
    
    total_test_loss = 0

    with torch.no_grad():
        for i, (fields, target) in enumerate(tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0, leave=True)):
            fields, target = fields.to(device), target.to(device)

            y = model(fields)

            target = target.long()
            test_loss = criterion(y, target.squeeze(dim=-1))
            total_test_loss += test_loss.item()
            
            targets.extend(target.tolist())
            predicts.extend(y.tolist())

    # target shape: (batch, 1), predicts shape: (batch, 3)
    targets_score = (np.array(targets) == 1).reshape(-1)
    targets_concede = (np.array(targets) == 2).reshape(-1)

    #1 vs 나머지 & 2 vs 나머지로 predict를 나눠서 예측한 값
    predicts_score = np.stack((1-np.array(predicts)[:,1],np.array(predicts)[:,1]), axis=1)
    predicts_concede = np.stack((1-np.array(predicts)[:,2],np.array(predicts)[:,2]) , axis=1)

    eval_roc_auc= {'score_roc_auc' : roc_auc_score(targets_score, predicts_score[:,1]) ,
                  'concede_roc_auc' : roc_auc_score(targets_concede, predicts_concede[:,1])
                  }
    
    eval_brier = {'score_brier' : brier_score_loss(targets_score, predicts_score[:,1]),
                  'concede_brier' : brier_score_loss(targets_concede, predicts_concede[:,1])}
    
    roc_auc_result = str(eval_roc_auc['score_roc_auc']) + " / " + str(eval_roc_auc['concede_roc_auc'])
    brier_result = str(eval_brier['score_brier']) + " / " + str(eval_brier['concede_brier'])
    
    print("Test score/concede ROC AUC", roc_auc_result)
    print("Test score/concede brier", brier_result)

    return pd.DataFrame(predicts,columns=['Not','scores','concedes'])

    
def main(model_name, epoch, learning_rate, batch_size, weight_decay, device, save_dir):

    print(f'\nscore & concede dataset making...')

    device = torch.device(device)
    #CustomDataset first parameter : train/valid/test
    #second parameter : score label / concede label
    train_dataset = CustomDataset(1,StandardScaler(),folder="multiclass-data") 
    valid_dataset = CustomDataset(2,train_dataset.scaler,folder="multiclass-data") 
    test_dataset = CustomDataset(3,train_dataset.scaler,folder="multiclass-data")
    
    target_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=8)
    target_vector = load_targets(target_data_loader)

    # sampler는 oversample이라기 보다는 데이터를 batch단위로 어떠한 규칙으로 추출할지 정하는 과정임
    # oversampler클래스에서는 각 target이 밸런스있기 뽑힐 수 있도록 weight를 조정해서
    # 모든 target이 골고루 뽑힐 수 있게 조정함
    # batch=4096단위로 학습할 때, ex{ 0: 1369, 1: 1312, 2: 1415}씩 뽑아서 train데이터에 활용함
    sampler = OverSampler(class_vector=torch.from_numpy(
        target_vector).view(-1), batch_size=batch_size * 244)
    
    
    train_data_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8,sampler=sampler)
    valid_data_loader = DataLoader(
        valid_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    test_data_loader = DataLoader(
        test_dataset, batch_size=batch_size, num_workers=8,shuffle=False)
    
    print(f'score & concede modeling ')
    
    model = get_model(model_name, train_dataset,output_dim=3).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    score_early_stopper = Score_EarlyStopper(num_trials=20, save_path=f"{save_dir}/{model_name}.pt")
    concede_early_stopper = Concede_EarlyStopper(num_trials=20, save_path=f"{save_dir}/{model_name}.pt")

    for epoch_i in range(epoch):
        train(model, optimizer, train_data_loader, criterion, epoch_i, device)
        score_auc, concede_auc = valid_evaluation(model, valid_data_loader, criterion, epoch_i, device)

        print('\n----------epoch :', epoch_i,"----------")
        if not score_early_stopper.is_continuable(model, score_auc) and not concede_early_stopper.is_continuable(model, concede_auc):
            print(f'early validation: score best auc: {score_early_stopper.best_accuracy}')
            print(f'early validation: concede best auc: {concede_early_stopper.best_accuracy}')
            break

    writer.close()
    #test_data_loader에서는 loss검사를 똑같이 할 필요없음
    probability = test_evaluation(model,test_data_loader, criterion, device)

    return probability

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='multiclass')
    parser.add_argument('--model', default='entitiy_dcn')
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--weight_decay', type=float, default=1e-2)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_dir', default='chkpt')
    
    args, _ = parser.parse_known_args()

    from datetime import datetime
    now = datetime.now()
    schedule = str(now.date()) + " " + str(now.hour) + "시 " + str(now.minute) + "분"

    name = f'runs/ now={schedule} & model={args.model} & data={args.data}'
    writer = SummaryWriter()

    probability = main( 
         args.model,
         args.epoch,
         args.learning_rate,
         args.batch_size,
         args.weight_decay,
         args.device,
         args.save_dir
         )

    print(probability)
    probability.to_csv(f'./result/{args.data}_prediction.csv',index=False)
