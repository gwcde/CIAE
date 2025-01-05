import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV,train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler
import shap
from lime.lime_tabular import LimeTabularExplainer
import ft_transformer as rtdl
from skorch import NeuralNetClassifier, NeuralNet
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import torch
from skorch.callbacks import Callback
from skorch.callbacks import EarlyStopping
from sklearn.base import BaseEstimator
from skorch.dataset import ValidSplit
from captum.attr import (
    Saliency,
    IntegratedGradients,
)
import random
import yaml
import argparse
import ast
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
    torch.backends.cudnn.enabled = False  # 
    torch.use_deterministic_algorithms(True)

def acquire_feature(file):
    import_select_ko_lists = []
    if(os.path.exists(file)):
        with open(file,'r')as file:
            ko_all = file.readlines()
            for ko_tmp in ko_all:
                import_select_ko_lists.append(ko_tmp.strip().split(' '))
    return import_select_ko_lists

def acqure_ori_feature(in_file,seed):
    pd_tmp = pd.read_csv(in_file)
    cols = list(pd_tmp)
    y = pd_tmp['label']
    for i in ['sample_id', 'label', 'Unnamed: 0']:
        if i in cols:
            cols.remove(i)
    X = pd_tmp.loc[:, cols]
    all_feature = list(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    print(type(X))
    # 
    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=seed,
                                                        stratify=y)

    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = np.expand_dims(y_train, axis=1).astype(np.float32)
    return x_train,x_test,y_train,y_test,all_feature

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

class ClassificationAccuracy_C(Callback):
    def __init__(self):
        self.output_dir = f"./ckpt"
        self.out_file = f"ckpt_C-T2D"
    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        # save valid scores
        if net.history[-1, 'valid_loss_best']:
            save_best_dev_model(net, self.output_dir,self.out_file)

class ClassificationAccuracy_EW(Callback):
    def __init__(self):
        self.output_dir = f"./ckpt"
        self.out_file = f"ckpt_EW-T2D"
    def initialize(self):
        self.critical_epoch_ = -1

    def on_epoch_end(self, net, **kwargs):
        # save valid scores
        if net.history[-1, 'valid_loss_best']:
            save_best_dev_model(net, self.output_dir,self.out_file)

def acquire_model(model_type,x_train,y_train,disease,seed,**kwargs):
    modelconfig = kwargs
    if (model_type == 'Catboost'):
        if(disease == 'C-T2D'):
            net = CatBoostClassifier(learning_rate=modelconfig['learning_rate'], iterations=150, depth=modelconfig['depth'],
                                 colsample_bylevel=modelconfig['colsample_bylevel'],
                                 eval_metric='AUC', loss_function='Logloss', random_seed=seed)
        else:
            net = CatBoostClassifier(learning_rate=modelconfig['learning_rate'], iterations=50,
                                     depth=modelconfig['depth'],
                                     colsample_bylevel=modelconfig['colsample_bylevel'],
                                     eval_metric='AUC', loss_function='Logloss', random_seed=seed)
        net.fit(x_train, y_train, verbose=True)
    else:
        model = rtdl.FTTransformer.make_default(
            n_num_features=x_train.shape[1],
            cat_cardinalities=None,
            n_blocks=modelconfig['n_blocks'],
            last_layer_query_idx=[-1],  # it makes the model faster and does NOT affect its output
            d_out=1,
        )
        # X_train_val, X_val, y_train_val, y_val = train_test_split(X_train, y_train,test_size=0.2, random_state=42, stratify=y_train)

        if (disease == 'C-T2D'):
            net = NeuralNetClassifier(
                model,
                max_epochs=200,
                criterion=torch.nn.BCEWithLogitsLoss,
                lr=modelconfig['lr'],
                iterator_train__shuffle=True,
                train_split=ValidSplit(0.2, random_state=42),
                # train_split=None,
                device='cuda',
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=1e-4,
                batch_size=modelconfig['batch_size'],
                callbacks=[EarlyStopping(patience=15), ClassificationAccuracy_C()],
            )
        else:
            net = NeuralNetClassifier(
                model,
                max_epochs=100,
                criterion=torch.nn.BCEWithLogitsLoss,
                lr=modelconfig['lr'],
                iterator_train__shuffle=True,
                train_split=ValidSplit(0.2, random_state=42),
                # train_split=None,
                device='cuda',
                optimizer=torch.optim.AdamW,
                optimizer__weight_decay=1e-4,
                batch_size=modelconfig['batch_size'],
                callbacks=[EarlyStopping(patience=5), ClassificationAccuracy_EW()]
            )
        net.fit(x_train, y_train)
        if (disease == 'C-T2D'):
            net.load_params(f_params="./ckpt/ckpt_C-T2D/model_best.pkl",
                            f_optimizer="./ckpt/ckpt_C-T2D/optim_best.pkl",
                            f_history="./ckpt/ckpt_C-T2D/history_best.json")
        else:
            net.load_params(f_params="./ckpt/ckpt_EW-T2D/model_best.pkl",
                            f_optimizer="./ckpt/ckpt_EW-T2D/optim_best.pkl",
                            f_history="./ckpt/ckpt_EW-T2D/history_best.json")
    return net

def inpret_all_ML(fi_model,all_feature,x_train,x_test):
    #feature_importance
    ko_list = all_feature
    mem_weight = []
    for ko_sam, weight_tmp in zip(ko_list, fi_model.feature_importances_):
        mem_weight.append({ko_sam: weight_tmp})
    mem_weight_sorted_data = sorted(mem_weight, key=lambda x: list(x.values())[0], reverse=True)

    #shap
    explainer = shap.TreeExplainer(fi_model)
    shap.initjs()
    shap_values = explainer.shap_values(x_test)
    shap_mem_weight = []
    for i, ko in enumerate(all_feature):
        average = np.mean(np.abs((shap_values[:, i])))
        shap_mem_weight.append({ko: average})
    shap_mem_weight_sorted_data = sorted(shap_mem_weight, key=lambda x: list(x.values())[0], reverse=True)
    sum_0 = 0
    lst = []
    for i in shap_mem_weight_sorted_data:
        sum_0 = sum_0 + list(i.values())[0]
        lst.append(list(i.values())[0])
    print(sum_0)
    arr = np.array(lst)
    total = np.sum(arr)
    scaled = np.divide(arr, total)
    for i, v in enumerate(shap_mem_weight_sorted_data):
        shap_mem_weight_sorted_data[i] = {list(v.keys())[0]: scaled[i]}

    #lime
    class_names = [0, 1]
    explainer = LimeTabularExplainer(np.array(x_train), feature_names=all_feature, class_names=class_names)
    all_weight_list = []
    for i in range(len(x_test)):
        exp = explainer.explain_instance(np.array(x_test)[i], fi_model.predict_proba, num_features=len(all_feature))
        # exp.show_in_notebook(show_table=True, show_all=True)
        for k, v in exp.local_exp.items():
            res = v
        sorted_res = sorted(res, key=lambda x: x[0])
        list_tmp = np.array([value_tmp[1] for value_tmp in sorted_res])
        all_weight_list.append(list_tmp)
        # if(i==3):break
    all_weight_list = np.array(all_weight_list)
    lime_mem_weight = []
    for i, ko in enumerate(all_feature):
        average = np.mean(np.abs((all_weight_list[:, i])))
        lime_mem_weight.append({ko: average})
    lime_mem_weight_sorted_data = sorted(lime_mem_weight, key=lambda x: list(x.values())[0], reverse=True)
    lime_all_ko_list = []
    for ko_data in lime_mem_weight_sorted_data:
        if (list(ko_data.values())[0] > 0):
            lime_all_ko_list.append(list(ko_data.keys())[0])
    lst_1 = []
    for i in lime_mem_weight_sorted_data:
        lst_1.append(list(i.values())[0])
    arr_1 = np.array(lst_1)
    total_1 = np.sum(arr_1)
    scaled_1 = np.divide(arr_1, total_1)
    for i, v in enumerate(lime_mem_weight_sorted_data):
        lime_mem_weight_sorted_data[i] = {list(v.keys())[0]: scaled_1[i]}

    return mem_weight_sorted_data,lime_mem_weight_sorted_data,shap_mem_weight_sorted_data

def guiyi(importance_dict_sorted):
    sum_0 = 0
    lst = []
    for i in importance_dict_sorted:
        sum_0 = sum_0 + list(i.values())[0]
        lst.append(list(i.values())[0])
    print(sum_0)
    arr = np.array(lst)
    total = np.sum(arr)
    scaled = np.divide(arr, total)
    # print(shap_mem_weight_sorted_data)
    sum_0 = 0
    for i, v in enumerate(importance_dict_sorted):
        importance_dict_sorted[i] = {list(v.keys())[0]: scaled[i]}
        sum_0 = sum_0 + scaled[i]
    #print(sum_0)
    return importance_dict_sorted

def importance_ko(attr,ko_list):
    importance_list = np.mean(np.abs(attr.tolist()),axis=0)
    #importance_list = np.mean(attr.tolist(),axis=0)
    print(len(importance_list))
    importance_dict = []
    for i,ko_name in enumerate(ko_list):
        importance_dict.append({ko_name:np.abs(importance_list[i])})
    return importance_dict

def inpret_all_DL(fi_model,all_feature,x_test):

    x1 = torch.tensor(
        np.array(list(0 if input.dtype is not torch.bool else False for input in x_test[0])).reshape((1, len(all_feature))))
    print(x1.shape)
    explainer = shap.DeepExplainer(fi_model.module.to('cpu'), torch.Tensor(
        np.array(list(0 if input.dtype is not torch.bool else False for input in x_test[0])).reshape((1, len(all_feature)))))
    shap.initjs()
    # shap_values = explainer.shap_values(X_test)
    shap_values = explainer.shap_values(torch.Tensor(x_test))
    shap_mem_weight = []
    ko_all_list = all_feature
    print(len(ko_all_list))
    for i, ko in enumerate(list(ko_all_list)):
        average = np.mean(np.abs((shap_values[:, i])))
        shap_mem_weight.append({ko: average})
    shap_mem_weight_sorted_data = sorted(shap_mem_weight, key=lambda x: list(x.values())[0], reverse=True)
    sum_0 = 0
    lst = []
    for i in shap_mem_weight_sorted_data:
        sum_0 = sum_0 + list(i.values())[0]
        lst.append(list(i.values())[0])
    print(sum_0)
    arr = np.array(lst)
    total = np.sum(arr)
    scaled = np.divide(arr, total)
    sum_0 = 0
    for i, v in enumerate(shap_mem_weight_sorted_data):
        shap_mem_weight_sorted_data[i] = {list(v.keys())[0]: scaled[i]}
        sum_0 = sum_0 + scaled[i]

    torch.cuda.empty_cache()
    fi_model.module.to('cpu')
    fi_model.module.zero_grad()
    # method = Saliency(net.module)
    method_igs = IntegratedGradients(fi_model.module)
    method_i_g = Saliency(fi_model.module)
    x_test = torch.Tensor(x_test)
    x_test.requires_grad_()
    pre = 0
    for i in range(5,len(x_test)+2,5):
        if(i==5):
            attr_igs = method_igs.attribute(inputs=x_test[pre:i,:], target=0)
        else:
            if(i>len(x_test)):i = len(x_test)
            tmp = method_igs.attribute(inputs=x_test[pre:i,:], target=0)
            attr_igs = torch.cat([attr_igs,tmp],dim=0)
        pre = i
    attr_i_g = method_i_g.attribute(inputs=x_test, target=0,abs=False)  # siliency
    importance_dict_igs = importance_ko(attr_igs, all_feature)
    importance_dict_sorted_igs = sorted(importance_dict_igs, key=lambda x: list(x.values())[0], reverse=True)
    importance_dict_i_g = importance_ko(attr_i_g, all_feature)
    importance_dict_sorted_i_g = sorted(importance_dict_i_g, key=lambda x: list(x.values())[0], reverse=True)
    importance_dict_sorted_igs = guiyi(importance_dict_sorted_igs)
    importance_dict_sorted_i_g = guiyi(importance_dict_sorted_i_g)
    return importance_dict_sorted_igs,importance_dict_sorted_i_g,shap_mem_weight_sorted_data

def evaluate(net: BaseEstimator, X: np.ndarray, y: np.ndarray):
    try:
        y_true, y_pred = y, net.predict(X)
        y_prob = net.predict_proba(X)
        # Performance Metrics: AUC, ACC, Recall, Precision, F1_score
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

def compute_five_inpret(inprets):
    print(len(inprets))
    if(len(inprets)<1):first_inpret = 0
    else:first_inpret = inprets[0]
    if(first_inpret==0):return False
    final_inpret = []
    for weight_tmp in first_inpret:
        KO = list(weight_tmp.keys())[0]
        weight_value = list(weight_tmp.values())[0]
        sum_weight = weight_value
        for i,tmp_inpret in enumerate(inprets):
            if(i==0):continue
            for res_tmp in tmp_inpret:
                if(KO==list(res_tmp.keys())[0]):
                    sum_weight = sum_weight+list(res_tmp.values())[0]
                    break
        final_inpret.append({KO:sum_weight/len(inprets)})
    sorted_final_inpret = sorted(final_inpret, key=lambda x: list(x.values())[0], reverse=True)

    return sorted_final_inpret

def compute_five_inpret(inprets):
    # print(len(inprets))
    if(len(inprets)<1):first_inpret = 0
    else:first_inpret = inprets[0]
    if(first_inpret==0):return False
    final_inpret = []
    for weight_tmp in first_inpret:
        weight_tmp = ast.literal_eval(weight_tmp)
        KO = list(weight_tmp.keys())[0]
        weight_value = list(weight_tmp.values())[0]
        sum_weight = weight_value
        for i,tmp_inpret in enumerate(inprets):
            if(i==0):continue
            for res_tmp in tmp_inpret:
                if(KO==list(ast.literal_eval(res_tmp).keys())[0]):
                    sum_weight = sum_weight+list(ast.literal_eval(res_tmp).values())[0]
                    break
        final_inpret.append({KO:sum_weight/len(inprets)})
    sorted_final_inpret = sorted(final_inpret, key=lambda x: list(x.values())[0], reverse=True)

    return sorted_final_inpret

def compute_five_merge_inpret(root_dir,disease):
    files = os.listdir(root_dir)
    files = sorted(files)
    # print(files)
    # print(f'{root_dir}'+'/EW_weight_KO.csv')
    inpret_1 = []
    inpret_2 = []
    inpret_3 = []
    inpret_4 = []
    inpret_5 = []
    inpret_6 = []
    for f in files:
        tmp_f = pd.read_csv(os.path.join(root_dir,f))
        for i,tse in enumerate(list(tmp_f)):
            if(i==0):inpret_1.append(list(tmp_f[tse]))
            elif(i==1):inpret_2.append(list(tmp_f[tse]))
            elif(i==2):inpret_3.append(list(tmp_f[tse]))
            elif(i==3):inpret_4.append(list(tmp_f[tse]))
            elif(i==4):inpret_5.append(list(tmp_f[tse]))
            else:inpret_6.append(list(tmp_f[tse]))
    fi_inpret_1 = compute_five_inpret(inpret_1)
    fi_inpret_2 = compute_five_inpret(inpret_2)
    fi_inpret_3 = compute_five_inpret(inpret_3)
    fi_inpret_4 = compute_five_inpret(inpret_4)
    fi_inpret_5 = compute_five_inpret(inpret_5)
    fi_inpret_6 = compute_five_inpret(inpret_6)

    list_cols = ['cat_importance', 'cat_lime', 'cat_shap','ft_igs', 'ft_saliency', 'ft_shap']
    pd_tmp = pd.DataFrame(columns=list_cols)
    pd_tmp['cat_importance'] = fi_inpret_1
    pd_tmp['cat_lime'] = fi_inpret_2
    pd_tmp['cat_shap'] = fi_inpret_3
    pd_tmp['ft_igs'] = fi_inpret_4
    pd_tmp['ft_saliency'] = fi_inpret_5
    pd_tmp['ft_shap'] = fi_inpret_6
    if(disease=='EW-T2D'):pd_tmp.to_csv(f'{root_dir}'+'/EW_weight_KO.csv', index=False)
    else:pd_tmp.to_csv(f'{root_dir}'+'/C_weight_KO.csv', index=False)    


def main():
    # 初始化 argparse 解析器
    parser = argparse.ArgumentParser(description='Process input file paths and output locations.')

    # 添加必需的参数
    parser.add_argument('--KO_EW_file', type=str, required=True, help='Path to the EW_KO file.')
    parser.add_argument('--KO_C_file', type=str, required=True, help='Path to the C_KO file.')
    parser.add_argument('--config_file', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--EW_T2D_out_file', type=str, required=True, help='Output path_dir for EW weights.')
    parser.add_argument('--C_T2D_out_file', type=str, required=True, help='Output path_dir for C weights.')

    # 解析命令行参数
    args = parser.parse_args()

    # 打印参数
    print(f"KO_EW_file: {args.KO_EW_file}")
    print(f"KO_C_file: {args.KO_C_file}")
    print(f"config_file: {args.config_file}")
    print(f"EW_T2D_out_file: {args.EW_T2D_out_file}")
    print(f"C_T2D_out_file: {args.C_T2D_out_file}")

    KO_EW_file = args.KO_EW_file
    KO_C_file = args.KO_C_file
    config_file = args.config_file
    EW_T2D_out_file = args.EW_T2D_out_file
    C_T2D_out_file = args.C_T2D_out_file

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    paras_cat_C = config["paras_cat_C"]
    paras_cat_EW = config["paras_cat_EW"]
    paras_ft_C = config["paras_ft_C"]
    paras_ft_EW = config["paras_ft_EW"]
    diseases_list = ['EW-T2D','C-T2D']
    models_list = ['Catboost','FT']
    seeds=[412,432,452,472,392]

    # for disease in ['EW-T2D','C-T2D']:
    for disease in diseases_list:
        # for model_type in models_list:
        for model_type in models_list:
            for seed in seeds:
                setup_seed(seed)
                if (disease == 'EW-T2D'):KO_file = KO_EW_file
                else:KO_file = KO_C_file
                x_train, x_test, y_train, y_test, all_feature = acqure_ori_feature(in_file=KO_file,seed=seed)
                if(disease=='EW-T2D' and model_type=='FT'):
                    print('begin: ', model_type, disease)
                    fi_model = acquire_model(model_type,x_train,y_train,disease,seed,**paras_ft_EW)
                elif(disease=='EW-T2D' and model_type=='Catboost'):
                    print('begin: ', model_type, disease)
                    fi_model = acquire_model(model_type, x_train, y_train, disease,seed, **paras_cat_EW)
                elif(disease=='C-T2D' and model_type=='FT'):
                    print('begin: ', model_type, disease)
                    fi_model = acquire_model(model_type, x_train, y_train, disease,seed, **paras_ft_C)
                else:
                    print('begin: ',model_type,disease)
                    fi_model = acquire_model(model_type, x_train, y_train, disease,seed, **paras_cat_C)

                scores = evaluate(fi_model, x_test, y_test)
                print(scores)
                if(model_type=='Catboost'):
                    inpret_1,inpret_2,inpret_3 = inpret_all_ML(fi_model,all_feature,x_train,x_test)
                else:
                    inpret_1,inpret_2,inpret_3 = inpret_all_DL(fi_model,all_feature,x_test)

                if(model_type=='Catboost'):
                    list_cols = ['cat_importance', 'cat_lime', 'cat_shap']
                    pd_tmp = pd.DataFrame(columns=list_cols)
                    pd_tmp['cat_importance'] = inpret_1
                    pd_tmp['cat_lime'] = inpret_2
                    pd_tmp['cat_shap'] = inpret_3
                    if(disease=='EW-T2D'):
                        pd_tmp.to_csv(EW_T2D_out_file+str(seed)+'.csv', index=False)
                    else:
                        pd_tmp.to_csv(C_T2D_out_file+str(seed)+'.csv', index=False)
                else:
                    list_cols = ['ft_igs', 'ft_saliency', 'ft_shap']
                    if(disease=='EW-T2D'):
                        pd_tmp = pd.read_csv(EW_T2D_out_file+str(seed)+'.csv')
                    else:
                        pd_tmp = pd.read_csv(C_T2D_out_file+str(seed)+'.csv')
                    pd_tmp['ft_igs'] = inpret_1
                    pd_tmp['ft_saliency'] = inpret_2
                    pd_tmp['ft_shap'] = inpret_3
                    if(disease=='EW-T2D'):
                        pd_tmp.to_csv(EW_T2D_out_file+str(seed)+'.csv', index=False)
                    else:
                        pd_tmp.to_csv(C_T2D_out_file+str(seed)+'.csv', index=False)
    compute_five_merge_inpret(EW_T2D_out_file,'EW-T2D')
    compute_five_merge_inpret(C_T2D_out_file,'C-T2D')

if __name__ == '__main__':
    main()
