import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from gensim.models.doc2vec import Doc2Vec

import argparse
import json
import os
import time
import warnings
import psutil
#import chain
#from itertools import chain
 
from utils import cal_metric, get_ids, text2words

warnings.filterwarnings('ignore')


def parse_args():
    #parser = argparse.ArgumentParser(
    parser = argparse.ArgumentParser(description="Baseline Model")
    parser.add_argument('--task', help="mortality, readmit, llos (default:mortality)", type=str, default='mortality') # mortality, readmit, or llos
    parser.add_argument('--model', help="all, lr, or rf (default:all)", type=str, default='all') # all, lr, or rf
    parser.add_argument('--inputs', help="3: T + S, 4: U, 7: U + T + S (default=4)", type=int, default=4) # 3: T + S, 4: U, 7: U + T + S
    args = parser.parse_args()
    #print("args")
    return args


def train_test_base(X_train, X_test, y_train, y_test, name):
    mtl = 1 if y_test.shape[1] > 1 else 0 # multi-label
    if name == 'lr':
        print('Start training Logistic Regression:')
        model = LogisticRegression()
        param_grid = {
            #'penalty': ['l1', 'l2'],
            #'solver' : 'liblinear'
            'penalty': ['l2']
        }
    else:
        print('Start training Random Forest:')
        model = RandomForestClassifier()
        param_grid = {
            'n_estimators': [x for x in range(20, 40, 5)],
            'max_depth': [None, 20, 40, 60, 80, 100]
        }
    if mtl:
        model = OneVsRestClassifier(model)
    else:
        y_train, y_test = y_train[:, 0], y_test[:, 0]
    print("get params for the model", model.get_params(True))
    #print("get params for the model", model.)
    t0 = time.time()
    gridsearch = GridSearchCV(model, param_grid, scoring='roc_auc', cv=5)
    gridsearch.fit(X_train, y_train)
    model = gridsearch.best_estimator_
    t1 = time.time()
    print('Running time:', t1 - t0)
    probs = model.predict_proba(X_test)
    print("best params", gridsearch.best_params_)
    print("feature name", gridsearch.n_features_in_)
    #print("feature name", gridsearch.feature_names_in_)
    metrics = []
    print("probs", probs)
    if mtl:
        for idx in range(y_test.shape[1]):
            metric = cal_metric(y_test[:, idx], probs[:, idx])
            print(idx + 1, metric)
            metrics.append(metric)
        print('Avg', np.mean(metrics, axis=0).tolist())
    else:
        metric = cal_metric(y_test, probs[:, 1])
        print(metric)
    
from getram import getram_cpu, getGPURAM, updatestop, getCPURAM, psutilvm
import threading

if __name__ == '__main__':
    bstop_threads = False
    ##t1 = threading.Thread(target = getGPURAM)
    #t1 = threading.Thread(target = getCPURAM)
    #t1 = threading.Thread(target = psutilvm)
    #t1.start()
    args = parse_args()
    task = args.task
    model = args.model
    inputs = args.inputs
    print('Running task %s using inputs %d...' % (task, inputs))
    train_ids, _, test_ids = get_ids('data/processed/files/splits.json')
    print("After reading json, train_ids: ", train_ids, "test_ids", test_ids)
    df = pd.read_csv('data/processed/%s.csv' % task).sort_values('hadm_id')
    #print("df['hadm_id]", df['hadm_id'])
    #print("Before intersect1d: df['hadm_id'].isin(train_ids)", df['hadm_id'].isin(train_ids))

    train_ids = np.intersect1d(train_ids, df['hadm_id'].tolist())
    test_ids = np.intersect1d(test_ids, df['hadm_id'].tolist())
    #print("After np.intersect1d, train_ids", train_ids)
    
    #test_ids = list(test_ids)
    #print("type of test_ids", type(test_ids))
    
    #train_ids = list(train_ids)
    #print("type of train_ids", type(train_ids), train_ids)
    
    choices = '{0:b}'.format(inputs).rjust(3, '0')
    X_train, X_test = [], []

    if choices[0] == '1':
        print('Loading notes...')
        vector_dict = json.load(open('data/processed/files/vector_dict.json'))
        X_train_notes = [np.mean(vector_dict.get(adm_id, []), axis=0) for adm_id in train_ids]
        X_test_notes = [np.mean(vector_dict.get(adm_id, []), axis=0) for adm_id in test_ids]
        X_train.append(X_train_notes)
        X_test.append(X_test_notes)
        #print(X_test_notes)
    if choices[1] == '1':
        print('Loading temporal data...')
        df_temporal = pd.read_csv('data/processed/features.csv').drop('charttime', axis=1)
        temporal_mm_dict = json.load(open('data/processed/files/feature_mm_dict.json'))
        for col in df_temporal.columns[1:]:
            col_min, col_max = temporal_mm_dict[col]
            #print("col_min ", col_min, "col_max", col_max)
            df_temporal[col] = (df_temporal[col] - col_min) / (col_max - col_min)
        #print("df_temporal", df_temporal)
        df_temporal = df_temporal.groupby(
            'hadm_id').agg(['mean', 'count', 'max', 'min', 'std'])
        df_temporal.columns = ['_'.join(col).strip() for col in df_temporal.columns.values]
        df_temporal.fillna(0, inplace=True)
        df_temporal = df_temporal.reset_index().sort_values('hadm_id')
        df_temporal_cols = df_temporal.columns[1:]
        X_train_temporal = df_temporal[df_temporal['hadm_id'].isin(np.asarray(train_ids, dtype=int))][df_temporal_cols].to_numpy()
        X_test_temporal = df_temporal[df_temporal['hadm_id'].isin(np.asarray(test_ids, dtype=int))][df_temporal_cols].to_numpy()
        X_train.append(X_train_temporal)
        X_test.append(X_test_temporal)
        #print(X_train_temporal)
    if choices[2] == '1':
        print('Loading demographics...')
        demo_json = json.load(open('data/processed/files/demo_dict.json'))
        df_demo = pd.DataFrame(demo_json.items(), columns=['hadm_id', 'demos']).sort_values('hadm_id')
        X_train_demo = df_demo[df_demo['hadm_id'].isin(train_ids)][['demos']].to_numpy()
        X_test_demo = df_demo[df_demo['hadm_id'].isin(test_ids)][['demos']].to_numpy()
        #print("X_test", X_test)
        ftmp = []
        print("X_train_demo", len(X_train_demo))
        for i in X_train_demo:
          #print("demo", i)
          i = np.array(i)
          #print("after array", i)
          for j in i:
            #print("j", j)
            #j = np.array(j)
            #print("j", j)
            ftmp.append(j)
          #c = [np.array(x) for x in X_test]
        #print("ftmp", ftmp)
        #print("ftmp as array", np.asarray(ftmp))
        #X_train.append(X_train_demo)
        stmp = []
        for i in X_test_demo:
          #print("demo", i)
          i = np.array(i)
          #print("after array", i)
          for j in i:
            #print("j", j)
            #j = np.array(j)
            #print("j", j)
            stmp.append(j)
          #c = [np.array(x) for x in X_test]
        #print("stmp", stmp)
        #print("stmp as array", np.asarray(stmp))
        #X_train.append(X_train_demo)
        
        X_train.append(ftmp)
        X_test.append(stmp)
        #print("X_test", X_test)
    
    print('Done.')
    df_cols = df.columns[1:]

    #print("Shape of X_test", X_test.shape)
    #print("Before concat : ", X_train)
    X_train = np.concatenate(X_train, axis=1)#.flat
    X_test = np.concatenate(X_test, axis=1)#.flat
    #print("After concat : ", X_train)


    y_train = df[df['hadm_id'].isin(np.asarray(train_ids, dtype=int))][df_cols].to_numpy()
    y_test = df[df['hadm_id'].isin(np.asarray(test_ids, dtype=int))][df_cols].to_numpy()
    #print("df['hadm_id'].isin(train_ids)", df['hadm_id'].isin(train_ids))
    print("y_train: type: ", type(y_train), len(y_train))
    print("y_test: type: ", type(y_test), len(y_test))
    print("X_train: type: ", type(X_train), len(X_train))
    print("X_test: type: ", type(X_test), len(X_test))
    
    print("ava ", psutil.virtual_memory().available, "total ", psutil.virtual_memory().total,
    "used ", psutil.virtual_memory().used)
    #a, b, c = getram_cpu()
    #for i in X_train:
      #print("i type", type(i))
    #  for j in i:
    #    if not (isinstance(j, float)):
    #      a=1
          #print("j type", type(j), j)
          #print("xtrain", X_train[j])
    if model == 'all':
        train_test_base(X_train, X_test, y_train, y_test, 'lr')
        train_test_base(X_train, X_test, y_train, y_test, 'rf')
    else:
        train_test_base(X_train, X_test, y_train, y_test, model)
    d, e, f = getram_cpu()

    print("ava ", psutil.virtual_memory().available, "total ", psutil.virtual_memory().total, "used ", psutil.virtual_memory().used)
    #print("total = ", d, a, " used= ", e, b, " rem= " , f,c)
    #bstop_threads=True
    #updatestop(bstop_threads)
    #t1.join()