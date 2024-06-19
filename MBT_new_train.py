from collections import OrderedDict
import numpy as np
import yaml
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from typing import List, Any, Dict
from skorch.callbacks import Callback
from skorch.callbacks import EarlyStopping
from skorch.dataset import ValidSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.base import BaseEstimator
from skorch import NeuralNetClassifier, NeuralNet
import torch
import random
import os
import ft_transformer as rtdl
from sklearn.preprocessing import StandardScaler
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

def acqure_ori_feature(in_file):
    pd_tmp = pd.read_csv(in_file)
    cols = list(pd_tmp)
    y = pd_tmp['label']
    for i in ['sample_id', 'label', 'Unnamed: 0']:
        if i in cols:
            cols.remove(i)
    X = pd_tmp.loc[:, cols]

    return X,y
def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray,save_results_file):
    try:
        y_true, y_pred = y, net.predict(X)
        y_prob = net.predict_proba(X)
        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
        #print(y_prob)
        if(save_results_file!='NO'):
            if(os.path.exists(save_results_file)):
                print('file already exits!!!!!!!!!!!!!!!')
                pd_tmp = pd.read_csv(save_results_file)
                pd_tmp.loc[len(pd_tmp)] = ['ft_pred','ft_true']
                for x,y in zip(list(y_prob[:,1,0]),list(y_true)):
                    pd_tmp.loc[len(pd_tmp)] = [x,y]
            else:
                pd_tmp = pd.DataFrame(columns=['ft_pred','ft_true'])
                pd_tmp['ft_pred'] = list(y_prob[:,1,0])
                pd_tmp['ft_true'] = list(y_true)
            pd_tmp.to_csv(save_results_file, index=False)

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

def acquire_feature(file):
    import_select_ko_lists = []
    if(os.path.exists(file)):
        with open(file,'r')as file:
            ko_all = file.readlines()
            for ko_tmp in ko_all:
                import_select_ko_lists.append(ko_tmp.strip().split(' '))
    return import_select_ko_lists


def save_best_dev_model(net, output_dir: str,out_file:str):
    """save the best dev model"""
    if not os.path.exists(output_dir + f"/{out_file}"):
        os.mkdir(output_dir + f"/{out_file}")
    epoch = len(net.history)
    params_path = output_dir + f"/{out_file}/model_best.pkl"
    optim_path = output_dir + f"/{out_file}/optim_best.pkl"
    history_path = output_dir + f"/{out_file}/history_best.json"
    net.save_params(f_params=params_path,
                    f_optimizer=optim_path,
                    f_history=history_path)

class ClassificationAccuracy_EW(Callback):
    def __init__(self):
        self.output_dir = f"/hde/save_models"
        self.out_file = f"ckpt_EW-T2D"
    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        # save valid scores
        if net.history[-1, 'valid_loss_best']:
            save_best_dev_model(net, self.output_dir,self.out_file)

class ClassificationAccuracy_C(Callback):
    def __init__(self):
        self.output_dir = f"/hde/save_models"
        self.out_file = f"ckpt_C-T2D"
    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        # save valid scores
        if net.history[-1, 'valid_loss_best']:
            save_best_dev_model(net, self.output_dir,self.out_file)

def FT_train(disease, feature, seed, use_config,use_best_losses,in_file: str, out_file: str,save_results_file: str, **kwargs):
    # Set random seed
    setup_seed(seed)
    print('begin_FT_train:!')
    # Load feature
    if disease in ['EW-T2D', 'C-T2D']:
        X,y = acqure_ori_feature(in_file=in_file)
        #import_select_ko_lists = acquire_feature(file=file_f)
        #X = X.loc[:,import_select_ko_lists[index_se][:ko_nums]]
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(X.shape)
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
        # dir_dict = {
        #     'emo': "/hdd/wmh/Disease/Data/AD/emo_large/",
        #     'egemaps': "/hdd/wmh/Disease/Data/AD/eGeMAPSv02/",
        #     'compare': "/hdd/wmh/Disease/Data/AD/ComParE_2016/",
        #     'liwc': "/hdd/wmh/Disease/Data/AD/linguistic/"
        # }
        # path = dir_dict[feature.lower()]
        #
        # train_path, test_path = path + "train.csv", path + "test.csv"
        # train_data = dataset(train_path, use_cols=None)  # Z-Score
        # test_data = dataset(test_path, use_cols=None)
        #
        # x_train, x_test = train_data.data.astype(np.float32), test_data.data.astype(np.float32)
        # y_train, y_test = train_data.label, test_data.label
        #
        # y_train = np.expand_dims(y_train, axis=1).astype(np.float32)


    if use_config:
        config_path = f"/hdd/wmh/Disease/Config/{disease}.yaml"
        with open(config_path) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            modelconfig = config['FT'][feature][seed]
    else:
        modelconfig = kwargs

    # other config
    modelconfig['lr'] = float(modelconfig['lr'])
    record = OrderedDict(modelconfig)
    record['seed'] = seed
    record['feature'] = feature
    # record['feature'] = 'kg'

    modelconfig['n_num_features'] = x_train.shape[1]
    modelconfig['last_layer_query_idx'] = [-1]
    modelconfig['d_out'] = 1
    modelconfig['cat_cardinalities'] = None

    # print(record)
    # print(x_train)

    device = "cuda"

    # drop some configs
    lr = modelconfig['lr']
    batch_size = int(modelconfig['batch_size'])

    modelconfig.pop('lr')
    modelconfig.pop('batch_size')

    # Init model
    model = rtdl.FTTransformer.make_default(**modelconfig).cuda()

    if disease == 'Obesity':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([0.5]))
    else:
        criterion = torch.nn.BCEWithLogitsLoss
    logpath = out_file
    # check record
    if not check_record(record, logpath):
        print("paras has trained.")
        return
    if(disease =='EW-T2D'):
        net = NeuralNetClassifier(
            model,
            max_epochs=100,
            criterion=criterion,
            lr=lr,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=ValidSplit(0.2, random_state=42),
            device=device,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=1e-4,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=5)] if(use_best_losses==False) else [EarlyStopping(patience=5), ClassificationAccuracy_EW()],
        )
    else:
        net = NeuralNetClassifier(
            model,
            max_epochs=200,
            criterion=criterion,
            lr=lr,
            # Shuffle training data on each epoch
            iterator_train__shuffle=True,
            train_split=ValidSplit(0.2, random_state=42),
            device=device,
            optimizer=torch.optim.AdamW,
            optimizer__weight_decay=1e-4,
            batch_size=batch_size,
            callbacks=[EarlyStopping(patience=15)] if(use_best_losses==False) else [EarlyStopping(patience=15), ClassificationAccuracy_C()],
        )

    net.fit(x_train, y_train)

    # test
    if(use_best_losses):
        if(disease=='C-T2D'):
            net.load_params(f_params="./ckpt/ckpt_C-T2D/model_best.pkl",
                            f_optimizer="./ckpt/ckpt_C-T2D/optim_best.pkl",
                            f_history="./ckpt/ckpt_C-T2D/history_best.json")
        else:
            net.load_params(f_params="./ckpt/ckpt_EW-T2D/model_best.pkl",
                            f_optimizer="./ckpt/ckpt_EW-T2D/optim_best.pkl",
                            f_history="./ckpt/ckpt_EW-T2D/history_best.json")
    scores = evaluate(net, x_test, y_test,save_results_file)
    record.update(scores)

    try:
        res_df = pd.read_csv(logpath)
        record_df = pd.DataFrame(record, index=[0])
        res_df = pd.concat([res_df, record_df])
    except:
        res_df = pd.DataFrame(record, index=[0])
    res_df.to_csv(logpath, index=False)




