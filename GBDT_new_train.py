
import numpy as np
import yaml
import pandas as pd
import torch
import random
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from typing import List, Any, Dict
from xgboost import XGBClassifier
from xgboost.callback import  EarlyStopping
from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False  # 禁用cudnn使用非确定性算法
    torch.use_deterministic_algorithms(True)

def merge_ko(all_lists:list):
    merge_list = []
    if(len(all_lists)==0):return merge_list
    for i in all_lists[0]:
        flag = 1
        for j,v in enumerate(all_lists):
            if(j==0):continue
            if(i not in v):
                flag = 0
                break
        if(flag):merge_list.append(i)
    return merge_list

def acqure_ori_feature(in_file,inpret_file,colum,ratio):
    pd_tmp = pd.read_csv(in_file)
    cols = list(pd_tmp)
    y = pd_tmp['label']
    for i in ['sample_id', 'label', 'Unnamed: 0']:
        if i in cols:
            cols.remove(i)
    X = pd_tmp.loc[:, cols]
    pd_test = pd.read_csv(inpret_file)
    if(ratio<1):
        nums = int(ratio * len(pd_test))
    else:nums = ratio
    #("nums::::",nums)
    fi_col = []
    if(colum in list(pd_test)):
        for v in pd_test[colum][:nums]:
            if(v in cols):fi_col.append(v)
    else:
        list_tmp = []
        for tmp_colum in list(pd.read_csv(inpret_file)):
            if('cat' in colum and 'cat' in tmp_colum):
                list_tmp.append(list(pd_test[tmp_colum])[:nums])
        ML_KO = merge_ko(list_tmp)
        #print(len(ML_KO))
        for v in ML_KO:
            if (v in cols): fi_col.append(v)
    X = X.loc[:, fi_col]
    return X,y

def acquire_feature(file):
    import_select_ko_lists = []
    if(os.path.exists(file)):
        with open(file,'r')as file:
            ko_all = file.readlines()
            for ko_tmp in ko_all:
                import_select_ko_lists.append(ko_tmp.strip().split(' '))
    return import_select_ko_lists

def check_record(paras: Dict, df_path: str) -> bool:
    """
    :param paras: need to check
    :param res_df:
    :return:
    """
    if not os.path.exists(df_path):
        return True
    print(paras)

    res_df = pd.read_csv(df_path)[list(paras.keys())]
    # print(res_df)

    # print(dict(paras))
    for d in res_df.to_dict(orient='records'):
        # print(paras)
        # print(d)
        # print()
        # import time
        # time.sleep(10)
        if d == dict(paras):
            return False
    return True

def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray,save_results_file):
    try:
        y_true, y_pred = y, net.predict(X)
        y_prob = net.predict_proba(X)
        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        #print(y_prob)
        # if(save_results_file!='NO'):
        #     if(os.path.exists(save_results_file)):
        #         print('file already exits!!!!!!!!!!!!!!!')
        #         pd_tmp = pd.read_csv(save_results_file)
        #         pd_tmp.loc[len(pd_tmp)] = ['cat_pred','cat_true']
        #         for x,y in zip(list(y_prob[:,1]),list(y_true)):
        #             pd_tmp.loc[len(pd_tmp)] = [x,y]
        #     else:
        #         print('file donot exits!!!!!!!!!!!!!!!')
        #         pd_tmp = pd.DataFrame(columns=['cat_pred','cat_true'])
        #         pd_tmp['cat_pred'] = list(y_prob[:,1])
        #         pd_tmp['cat_true'] = list(y_true)
        #     pd_tmp.to_csv(save_results_file, index=False)

        metrics = {
            'AUC': round(roc_auc_score(y_true, y_prob[:, 1]), 4),
            'ACC': round(accuracy_score(y_true, y_pred), 4),
            'Recall': round(recall_score(y_true, y_pred), 4),
            'Precision': round(precision_score(y_true, y_pred), 4),
            'F1': round(f1_score(y_true, y_pred), 4)
        }
        return metrics

    except:
        return {
            'AUC': -1.0,
            'ACC': -1.0,
            'Recall': -1.0,
            'Precision': -1.0,
            'F1': -1.0
        }

def XG_train(disease, feature, seed, use_config,use_best_losses,index_se,ko_nums,GBDT_type,file_f,in_file: str, out_file: str, **kwargs):
    # Set random seed
    setup_seed(seed)
    print('begin_XG_train:!')
    # Load feature
    if disease in ['EW-T2D', 'LC', 'C-T2D', 'IBD', 'Obesity']:
        X,y = acqure_ori_feature(in_file=in_file)
        import_select_ko_lists = acquire_feature(file=file_f)
        X = X.loc[:,import_select_ko_lists[index_se][:ko_nums]]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(type(X))
        # 划分数据
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=seed,
                                                            stratify=y)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)
    else:
        exit()

    if use_config:
        config_path = f"/hdd/wmh/Disease/Config/{disease}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            modelconfig = config['FT'][feature][seed]
    else:
        modelconfig = kwargs

    # other config
    record = OrderedDict(modelconfig)
    record['seed'] = seed
    record['feature'] = feature
    # record['feature'] = 'kg'

    logpath = out_file
    # check record
    if not check_record(record, logpath):
        print("paras has trained.")
        return
    if(GBDT_type=='XGBoost' and disease=='C-T2D'):
        if(use_best_losses):
            #print(y_train)
            x_train,x_val,y_train,y_val = train_test_split(x_train,y_train,test_size=0.2,random_state=42,stratify=y_train)
            eval_set=[(x_val, y_val)]
            net = XGBClassifier(n_estimators=50,tree_method='gpu_hist',max_depth=modelconfig['n_depth'],
                                min_child_weight=modelconfig['min_child_weight'],subsample=modelconfig['subsample'],
                                colsample_bytree=modelconfig['colsample_bytree'],random_state=seed)
            net.fit(x_train, y_train,eval_metric=["error", "logloss", "auc"],eval_set=eval_set,callbacks=[EarlyStopping(rounds=15,metric_name='logloss')])
        else:
            net = XGBClassifier(n_estimators=50,tree_method='gpu_hist',max_depth=modelconfig['n_depth'],
                                min_child_weight=modelconfig['min_child_weight'],subsample=modelconfig['subsample'],
                                colsample_bytree=modelconfig['colsample_bytree'],random_state=seed)
            net.fit(x_train, y_train)
    else:
        net = XGBClassifier(n_estimators=30,tree_method='gpu_hist',max_depth=modelconfig['n_depth'],
                            min_child_weight=modelconfig['min_child_weight'],subsample=modelconfig['subsample'],
                            colsample_bytree=modelconfig['colsample_bytree'],random_state=seed)
        net.fit(x_train, y_train)

    # test
    scores = evaluate(net, x_test, y_test)
    record.update(scores)

    try:
        res_df = pd.read_csv(logpath)
        record_df = pd.DataFrame(record, index=[0])
        res_df = pd.concat([res_df, record_df])
    except:
        res_df = pd.DataFrame(record, index=[0])
    res_df.to_csv(logpath, index=False)


def Cat_train(disease, feature, seed, use_config,GBDT_type,in_file: str, out_file: str,inpret_file: str,save_results_file:str,colum,ratio, **kwargs):
    # Set random seed
    setup_seed(seed)
    print('begin_Cat_train:!')
    # Load feature
    if disease in ['EW-T2D', 'LC', 'C-T2D', 'IBD', 'Obesity']:
        X, y = acqure_ori_feature(in_file=in_file,inpret_file=inpret_file,colum=colum,ratio=ratio)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        #print(X.shape)
        # 划分数据
        x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.2,
                                                            random_state=seed,
                                                            stratify=y)
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = np.expand_dims(y_train, axis=1).astype(np.float32)
    else:
        exit()

    if use_config:
        config_path = f"/hdd/wmh/Disease/Config/{disease}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            modelconfig = config['FT'][feature][seed]
    else:
        modelconfig = kwargs

    # other config
    record = OrderedDict(modelconfig)
    record['seed'] = seed
    record['feature'] = feature
    record['ratio'] = ratio
    record['inpret'] = colum
    # record['feature'] = 'kg'

    logpath = out_file
    # check record
    if not check_record(record, logpath):
        print("paras has trained.")
        return
    if(GBDT_type=='Catboost' and disease=='C-T2D'):
        # net = CatBoostClassifier(learning_rate=modelconfig['learning_rate'],iterations=50,depth=modelconfig['depth'],
        #                          subsample=modelconfig['subsample'],colsample_bylevel=modelconfig['colsample_bylevel'],
        #                          loss_function='Logloss',eval_metric='AUC',task_type='GPU',random_seed=seed)
        net = CatBoostClassifier(learning_rate=modelconfig['learning_rate'],iterations=100,depth=modelconfig['depth'],
                                 colsample_bylevel=modelconfig['colsample_bylevel'],
                                 eval_metric='AUC',loss_function='Logloss',random_seed=seed)
    else:
        net = CatBoostClassifier(learning_rate=modelconfig['learning_rate'],iterations=50,depth=modelconfig['depth'],
                                 colsample_bylevel=modelconfig['colsample_bylevel'],
                                 eval_metric='AUC',loss_function='Logloss',random_seed=seed)
    net.fit(x_train, y_train, verbose=True)

    # test
    scores = evaluate(net, x_test, y_test,save_results_file)
    record.update(scores)

    try:
        res_df = pd.read_csv(logpath)
        record_df = pd.DataFrame(record, index=[0])
        res_df = pd.concat([res_df, record_df])
    except:
        res_df = pd.DataFrame(record, index=[0])
    res_df.to_csv(logpath, index=False)
