import torch
import numpy as np
from torch.utils.data import DataLoader

import os
import warnings
import tqdm
import random
from sklearn.preprocessing import StandardScaler

import catboost
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score

import pandas as pd

from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss

categorical_features = []
numerical_features = []

def evaluate(y, y_hat):
    p = sum(y) / len(y)
    base = [p] * len(y)
    brier = brier_score_loss(y, y_hat)
    print(f"  Brier score: %.5f (%.5f)" % (brier, brier / brier_score_loss(y, base)))
    ll = log_loss(y, y_hat)
    print(f"  log loss score: %.5f (%.5f)" % (ll, ll / log_loss(y, base)))
    print(f"  ROC AUC: %.5f" % roc_auc_score(y, y_hat))

# 데이터 로딩 함수
def load_data(folder = 'binary'):
    X_train = pd.read_csv(f'../data/{folder}-data/train/X_train.csv')
    Y_train = pd.read_csv(f'../data/{folder}-data/train/Y_train.csv')
    X_valid = pd.read_csv(f'../data/{folder}-data/valid/X_valid.csv')
    Y_valid = pd.read_csv(f'../data/{folder}-data/valid/Y_valid.csv')
    X_test = pd.read_csv(f'../data/{folder}-data/test/X_test.csv')
    Y_test = pd.read_csv(f'../data/{folder}-data/test/Y_test.csv')

    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    #StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer
    # scaler = StandardScaler()
    # X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    # X_valid[numerical_features] = scaler.transform(X_valid[numerical_features])
    # X_test[numerical_features] = scaler.transform(X_test[numerical_features])

    # 학습할 데이터의 인덱스를 생성
    indices = np.random.permutation(X_train.shape[0])
    X_train = X_train.iloc[indices].reset_index(drop=True)
    Y_train = Y_train.iloc[indices].reset_index(drop=True)

    print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape, X_test.shape, Y_test.shape)

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test

def test(models, X_test, Y_test, folder):
    #기존 연구는 score예측 모델 & concede예측 모델을 따로 학습시킴
    column_list = ['scores','concedes']
    Y_hat = pd.DataFrame()

    #따로 학습시키고 각각 score & concede label과 AUC로 평가하기
    for col in column_list:
        Y_hat[col] = [p[1] for p in models[col].predict_proba(X_test)]
        print(f"### Y: {col} ###")
        evaluate(Y_test[col], Y_hat[col])

    Y_hat.to_csv(f'./result/{folder}_prediction.csv',index=False)

def train(X_train, Y_train, X_valid, Y_valid):
    column_list = ['scores','concedes']
    models = {}
    
    for col in tqdm.tqdm(column_list):
        #catboost외 다양한 기계학습 모델 사용
        #XGBoost, Randomforest, logistic, decision tree, SVM
        model = catboost.CatBoostClassifier()
        model.fit(X_train,Y_train[col], cat_features=categorical_features, eval_set=[(X_valid,Y_valid[col])])
        models[col] = model
        
    return models
    
if __name__ == '__main__':
    print("soccer data analysis start\n")

    X_train, Y_train, X_valid, Y_valid, X_test, Y_test = load_data(folder='binary')
    models = train(X_train, Y_train, X_valid, Y_valid)
    test(models, X_test, Y_test, folder='binary')

    print("soccer data analysis end\n")

